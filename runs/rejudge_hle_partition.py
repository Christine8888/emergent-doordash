from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Literal

from pydantic import BaseModel

from src.datasets import (
    HLE_JUDGE_MAX_COMPLETION_TOKENS,
    build_hle_judge_prompt,
    hle_judge_usage_metadata,
)
from src.storage import (
    build_hinted_grade_path,
    build_hinted_inference_path,
    make_stable_id,
    write_jsonl,
)


DEFAULT_COMPARISON_JUDGE_MODEL = "gpt-5.4-nano-2026-03-17"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_NANO_INPUT_COST_PER_1M = 0.20
DEFAULT_NANO_OUTPUT_COST_PER_1M = 1.25


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    strict: Literal[True] = True


class RejudgeTask(BaseModel):
    index: int
    inference_id: str
    problem_id: str | None
    inference_row: dict[str, Any]
    original_grade: dict[str, Any]
    old_is_correct: Any
    prompt: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _log(message: str) -> None:
    print(f"[{_now_hhmm()}] {message}", flush=True)


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    return cleaned or "unknown"


def _fraction_text(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _iter_json_records(path: Path):
    decoder = json.JSONDecoder()
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            while text:
                try:
                    row, end = decoder.raw_decode(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse {path}:{line_number}: {exc}") from exc
                yield row
                text = text[end:].strip()


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_json_records(path) or []:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _is_llm_judged_grade(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        row.get("grader_type") in {"hle_official_style_llm_judge", "hle_llm_judge_error"}
        or metadata.get("judge_model") is not None
    )


def _answer_type_from_inference(row: dict[str, Any]) -> str:
    hint = row.get("hint")
    hint = hint if isinstance(hint, dict) else {}
    metadata = hint.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    problem_metadata = metadata.get("problem_metadata")
    problem_metadata = problem_metadata if isinstance(problem_metadata, dict) else {}
    answer_type = metadata.get("problem_answer_type") or metadata.get("answer_type")
    answer_type = answer_type or problem_metadata.get("answer_type")
    return str(answer_type or "")


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _build_reused_comparison_row(
    *,
    inference_row: dict[str, Any],
    original_grade: dict[str, Any],
    judge_model: str,
) -> dict[str, Any]:
    inference_id = str(inference_row.get("inference_id"))
    old_is_correct = original_grade.get("is_correct")
    return {
        "comparison_id": make_stable_id(
            inference_id,
            judge_model,
            "reused_original_grade",
            length=16,
        ),
        "inference_id": inference_id,
        "problem_id": inference_row.get("problem_id"),
        "model": inference_row.get("model"),
        "hint_type": inference_row.get("hint_type"),
        "fractioner": inference_row.get("fractioner"),
        "hint_fraction": inference_row.get("hint_fraction"),
        "rejudged": False,
        "old_is_correct": old_is_correct,
        "new_is_correct": old_is_correct,
        "agreement": True if _is_bool(old_is_correct) else None,
        "old_grader_type": original_grade.get("grader_type"),
        "new_grader_type": "reused_original_grade",
        "metadata": {},
        "created_at": _utcnow_iso(),
    }


def _is_completed_rejudge_row(
    row: dict[str, Any],
    *,
    judge_model: str,
    reasoning_effort: str,
) -> bool:
    if row.get("rejudged") is not True:
        return False
    metadata = row.get("new_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        metadata.get("judge_model") == judge_model
        and metadata.get("judge_reasoning_effort") == reasoning_effort
        and _is_bool(row.get("new_is_correct"))
    )


def _default_cost_per_1m(judge_model: str) -> tuple[float | None, float | None]:
    if judge_model == DEFAULT_COMPARISON_JUDGE_MODEL:
        return DEFAULT_NANO_INPUT_COST_PER_1M, DEFAULT_NANO_OUTPUT_COST_PER_1M
    return None, None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_new_metadata_int(rows: list[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        metadata = row.get("new_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        value = _maybe_int(metadata.get(key))
        if value is not None:
            total += value
    return total


def _estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> float | None:
    if input_cost_per_1m is None or output_cost_per_1m is None:
        return None
    return (
        (input_tokens / 1_000_000.0) * input_cost_per_1m
        + (output_tokens / 1_000_000.0) * output_cost_per_1m
    )


def _judge_exact_match(
    *,
    prompt: str,
    judge_model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    request_timeout: float,
) -> dict[str, Any]:
    from openai import LengthFinishReasonError, OpenAI, OpenAIError

    base_metadata = {
        "grader_type": "hle_rejudge_llm",
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort,
        "judge_max_completion_tokens": max_completion_tokens,
        "judge_prompt": prompt,
    }
    try:
        client = OpenAI(timeout=request_timeout)
        completion = client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=max_completion_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
            reasoning_effort=reasoning_effort,
        )
    except (LengthFinishReasonError, OpenAIError) as exc:
        return {
            "is_correct": None,
            "extracted_answer": None,
            "metadata": {
                **base_metadata,
                "grader_type": "hle_rejudge_llm_error",
                "judge_error_type": type(exc).__name__,
                "judge_error": str(exc),
            },
        }

    content = completion.choices[0].message.parsed
    metadata = {
        **base_metadata,
        **hle_judge_usage_metadata(getattr(completion, "usage", None)),
    }
    if content is None:
        return {
            "is_correct": None,
            "extracted_answer": None,
            "metadata": {
                **metadata,
                "grader_type": "hle_rejudge_llm_error",
                "judge_error_type": "NoParsedResponse",
                "judge_error": "HLE comparison judge returned no parsed response.",
            },
        }

    return {
        "is_correct": content.correct == "yes",
        "extracted_answer": content.extracted_final_answer,
        "metadata": {
            **metadata,
            "reasoning": content.reasoning,
        },
    }


def _run_rejudge_task(
    *,
    task: RejudgeTask,
    total_tasks: int,
    judge_model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    request_timeout: float,
) -> dict[str, Any]:
    _log(
        "[rejudge_hle_partition] "
        f"starting={task.index}/{total_tasks} "
        f"inference_id={task.inference_id} problem_id={task.problem_id} "
        f"prompt_chars={len(task.prompt)} approx_prompt_tokens={max(1, round(len(task.prompt) / 4.0))}"
    )
    started = time.monotonic()
    result = _judge_exact_match(
        prompt=task.prompt,
        judge_model=judge_model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        request_timeout=request_timeout,
    )
    elapsed = time.monotonic() - started

    new_is_correct = result.get("is_correct")
    agreement = None
    if _is_bool(task.old_is_correct) and _is_bool(new_is_correct):
        agreement = task.old_is_correct == new_is_correct

    _log(
        "[rejudge_hle_partition] "
        f"finished={task.index}/{total_tasks} elapsed_seconds={elapsed:.1f} "
        f"inference_id={task.inference_id} old_is_correct={task.old_is_correct} "
        f"new_is_correct={new_is_correct} agreement={agreement}"
    )

    return {
        "comparison_id": make_stable_id(
            task.inference_id,
            judge_model,
            reasoning_effort,
            length=16,
        ),
        "inference_id": task.inference_id,
        "problem_id": task.inference_row.get("problem_id"),
        "model": task.inference_row.get("model"),
        "hint_type": task.inference_row.get("hint_type"),
        "fractioner": task.inference_row.get("fractioner"),
        "hint_fraction": task.inference_row.get("hint_fraction"),
        "rejudged": True,
        "old_is_correct": task.old_is_correct,
        "new_is_correct": new_is_correct,
        "agreement": agreement,
        "old_extracted_answer": task.original_grade.get("extracted_answer"),
        "new_extracted_answer": result.get("extracted_answer"),
        "old_grader_type": task.original_grade.get("grader_type"),
        "new_grader_type": (result.get("metadata") or {}).get("grader_type"),
        "old_metadata": task.original_grade.get("metadata", {}),
        "new_metadata": result.get("metadata", {}),
        "elapsed_seconds": elapsed,
        "created_at": _utcnow_iso(),
    }


def _default_output_paths(
    *,
    output_dir: Path,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    judge_model: str,
    reasoning_effort: str,
) -> tuple[Path, Path]:
    partition = (
        f"{_safe_component(model)}__{_safe_component(hint_type)}__"
        f"{_safe_component(fractioner)}__fraction_{_fraction_text(hint_fraction)}"
    )
    if reasoning_effort != DEFAULT_REASONING_EFFORT:
        partition = f"{partition}__reasoning_{_safe_component(reasoning_effort)}"
    judge = _safe_component(judge_model)
    base = output_dir / judge / partition
    return (
        base.parent / f"{base.name}.jsonl",
        base.parent / f"{base.name}.summary.json",
    )


def rejudge_partition(
    *,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    judge_model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    data_root: Path,
    output_dir: Path,
    limit: int | None,
    rejudge_mode: str,
    request_timeout: float,
    concurrency: int,
    write_reused: bool,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    inference_path = build_hinted_inference_path(
        benchmark_name="hle",
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fraction=hint_fraction,
        data_root=data_root,
    )
    grade_path = build_hinted_grade_path(
        benchmark_name="hle",
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fraction=hint_fraction,
        data_root=data_root,
    )
    output_path, summary_path = _default_output_paths(
        output_dir=output_dir,
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fraction=hint_fraction,
        judge_model=judge_model,
        reasoning_effort=reasoning_effort,
    )

    inference_rows = [
        row for row in _read_json_records(inference_path) if not bool(row.get("is_error"))
    ]
    original_grades = _read_json_records(grade_path)
    original_by_inference_id = {
        str(row.get("inference_id")): row
        for row in original_grades
        if isinstance(row.get("inference_id"), str)
    }

    comparison_rows: list[dict[str, Any]] = []
    existing_comparison_rows = _read_json_records(output_path)
    completed_existing_by_inference_id = {
        str(row.get("inference_id")): row
        for row in existing_comparison_rows
        if isinstance(row.get("inference_id"), str)
        and _is_completed_rejudge_row(
            row,
            judge_model=judge_model,
            reasoning_effort=reasoning_effort,
        )
    }
    original_score_values: list[bool] = []
    reused_score_values: list[bool] = []
    rejudge_tasks: list[RejudgeTask] = []

    missing_original_grade = 0
    skipped_not_selected = 0
    already_rejudged = 0

    for inference_row in inference_rows:
        inference_id = str(inference_row.get("inference_id"))
        original_grade = original_by_inference_id.get(inference_id)
        if original_grade is None:
            missing_original_grade += 1
            continue

        old_is_correct = original_grade.get("is_correct")
        if _is_bool(old_is_correct):
            original_score_values.append(old_is_correct)

        should_rejudge = _is_llm_judged_grade(original_grade)
        if rejudge_mode == "all-exact":
            should_rejudge = _answer_type_from_inference(inference_row) == "exactMatch"

        if not should_rejudge:
            skipped_not_selected += 1
            if _is_bool(old_is_correct):
                reused_score_values.append(old_is_correct)
            if write_reused:
                comparison_rows.append(
                    _build_reused_comparison_row(
                        inference_row=inference_row,
                        original_grade=original_grade,
                        judge_model=judge_model,
                    )
                )
            continue

        if limit is not None and len(rejudge_tasks) >= limit:
            if _is_bool(old_is_correct):
                reused_score_values.append(old_is_correct)
            continue

        existing_rejudge = completed_existing_by_inference_id.get(inference_id)
        if existing_rejudge is not None:
            already_rejudged += 1
            comparison_rows.append(existing_rejudge)
            existing_new_is_correct = existing_rejudge.get("new_is_correct")
            if _is_bool(existing_new_is_correct):
                reused_score_values.append(existing_new_is_correct)
            elif _is_bool(old_is_correct):
                reused_score_values.append(old_is_correct)
            continue

        hint = inference_row.get("hint")
        hint = hint if isinstance(hint, dict) else {}
        prompt = build_hle_judge_prompt(
            question=str(hint.get("question") or ""),
            correct_answer=str(hint.get("answer") or ""),
            response=str(inference_row.get("model_output") or ""),
        )
        rejudge_tasks.append(
            RejudgeTask(
                index=len(rejudge_tasks) + 1,
                inference_id=inference_id,
                problem_id=str(inference_row.get("problem_id"))
                if inference_row.get("problem_id") is not None
                else None,
                inference_row=inference_row,
                original_grade=original_grade,
                old_is_correct=old_is_correct,
                prompt=prompt,
            )
        )

    total_tasks = len(rejudge_tasks)
    selected_rejudge_total = total_tasks + already_rejudged
    _log(
        "[rejudge_hle_partition] "
        f"selected_rejudge_total={selected_rejudge_total} "
        f"already_judged={already_rejudged} left={total_tasks} "
        f"concurrency={concurrency} "
        f"skipped_not_selected={skipped_not_selected} missing_original_grade={missing_original_grade}"
    )
    rejudged_rows: list[dict[str, Any]] = []
    if total_tasks:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _run_rejudge_task,
                    task=task,
                    total_tasks=total_tasks,
                    judge_model=judge_model,
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=max_completion_tokens,
                    request_timeout=request_timeout,
                )
                for task in rejudge_tasks
            ]
            for future in as_completed(futures):
                rejudged_rows.append(future.result())

    row_order = {task.inference_id: task.index for task in rejudge_tasks}
    rejudged_rows.sort(
        key=lambda row: row_order.get(str(row.get("inference_id")), len(row_order) + 1)
    )
    comparison_rows.extend(rejudged_rows)

    rejudge_errors = sum(
        1
        for row in rejudged_rows
        if not _is_bool(row.get("new_is_correct"))
    )
    agreement_values = [
        bool(row["agreement"])
        for row in rejudged_rows
        if _is_bool(row.get("agreement"))
    ]
    rejudged_old_values = [
        bool(row["old_is_correct"])
        for row in rejudged_rows
        if _is_bool(row.get("old_is_correct")) and _is_bool(row.get("new_is_correct"))
    ]
    rejudged_new_values = [
        bool(row["new_is_correct"])
        for row in rejudged_rows
        if _is_bool(row.get("old_is_correct")) and _is_bool(row.get("new_is_correct"))
    ]
    comparison_score_values = list(reused_score_values)
    for row in rejudged_rows:
        new_is_correct = row.get("new_is_correct")
        old_is_correct = row.get("old_is_correct")
        if _is_bool(new_is_correct):
            comparison_score_values.append(new_is_correct)
        elif _is_bool(old_is_correct):
            comparison_score_values.append(old_is_correct)

    if total_tasks:
        _log(
            "[rejudge_hle_partition] "
            f"finished_rejudge_tasks={total_tasks} rejudge_errors={rejudge_errors} "
            f"agreement={_mean_bool(agreement_values)}"
        )

    cost_rows = [
        row
        for row in comparison_rows
        if row.get("rejudged") is True
        and isinstance(row.get("new_metadata"), dict)
        and (row.get("new_metadata") or {}).get("judge_model") == judge_model
        and (row.get("new_metadata") or {}).get("judge_reasoning_effort") == reasoning_effort
    ]
    judge_input_tokens = _sum_new_metadata_int(cost_rows, "judge_input_token_count")
    judge_output_tokens = _sum_new_metadata_int(cost_rows, "judge_output_token_count")
    judge_total_tokens = _sum_new_metadata_int(cost_rows, "judge_total_token_count")
    judge_cached_input_tokens = _sum_new_metadata_int(cost_rows, "judge_cached_input_token_count")
    judge_reasoning_output_tokens = _sum_new_metadata_int(
        cost_rows,
        "judge_reasoning_output_token_count",
    )
    total_cost_usd = _estimate_cost_usd(
        input_tokens=judge_input_tokens,
        output_tokens=judge_output_tokens,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
    )

    summary = {
        "model": model,
        "hint_type": hint_type,
        "fractioner": fractioner,
        "hint_fraction": hint_fraction,
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort,
        "judge_max_completion_tokens": max_completion_tokens,
        "request_timeout_seconds": request_timeout,
        "concurrency": concurrency,
        "rejudge_mode": rejudge_mode,
        "inference_path": str(inference_path),
        "grade_path": str(grade_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "inference_rows": len(inference_rows),
        "original_grade_rows": len(original_grades),
        "missing_original_grade": missing_original_grade,
        "skipped_not_selected": skipped_not_selected,
        "rejudged": total_tasks,
        "rejudge_errors": rejudge_errors,
        "agreement_n": len(agreement_values),
        "agreement": _mean_bool(agreement_values),
        "original_partition_score": _mean_bool(original_score_values),
        "comparison_partition_score": _mean_bool(comparison_score_values),
        "original_rejudged_score": _mean_bool(rejudged_old_values),
        "comparison_rejudged_score": _mean_bool(rejudged_new_values),
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
        "judge_total_tokens": judge_total_tokens,
        "judge_cached_input_tokens": judge_cached_input_tokens,
        "judge_reasoning_output_tokens": judge_reasoning_output_tokens,
        "input_cost_per_1m_tokens_usd": input_cost_per_1m,
        "output_cost_per_1m_tokens_usd": output_cost_per_1m,
        "total_cost_usd": total_cost_usd,
        "created_at": _utcnow_iso(),
    }

    write_jsonl(output_path, comparison_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if total_cost_usd is None:
        _log(
            "[rejudge_hle_partition] "
            "total_cost_usd=unknown "
            "set --input-cost-per-1m and --output-cost-per-1m to price this judge model"
        )
    else:
        _log(
            "[rejudge_hle_partition] "
            f"total_cost_usd={total_cost_usd:.6f} "
            f"input_tokens={judge_input_tokens} output_tokens={judge_output_tokens} "
            f"input_cost_per_1m={input_cost_per_1m} output_cost_per_1m={output_cost_per_1m}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rejudge one HLE hinted grade partition with another judge model and "
            "write agreement/score-drift sidecars."
        )
    )
    parser.add_argument("--model", required=True, help="Answered model partition to rejudge.")
    parser.add_argument("--hint-type", default="answer_not_revealed")
    parser.add_argument("--fractioner", required=True)
    parser.add_argument("--hint-fraction", type=float, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "hinted_judge_comparisons" / "hle",
    )
    parser.add_argument("--judge-model", default=DEFAULT_COMPARISON_JUDGE_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-completion-tokens", type=int, default=HLE_JUDGE_MAX_COMPLETION_TOKENS)
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=None,
        help="Input-token price in USD per 1M tokens. Defaults to known nano pricing when applicable.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=None,
        help="Output-token price in USD per 1M tokens. Defaults to known nano pricing when applicable.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request OpenAI timeout in seconds.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent judge requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of selected rows to rejudge. Reused local rows still count in score.",
    )
    parser.add_argument(
        "--rejudge-mode",
        choices=["original-llm", "all-exact"],
        default="original-llm",
        help="original-llm rejudges only rows previously sent to the LLM judge.",
    )
    parser.add_argument(
        "--write-reused",
        action="store_true",
        help="Also write rows for non-rejudged local grades. Defaults to writing only rejudged rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_input_cost, default_output_cost = _default_cost_per_1m(args.judge_model)
    input_cost_per_1m = (
        args.input_cost_per_1m
        if args.input_cost_per_1m is not None
        else default_input_cost
    )
    output_cost_per_1m = (
        args.output_cost_per_1m
        if args.output_cost_per_1m is not None
        else default_output_cost
    )
    summary = rejudge_partition(
        model=args.model,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        hint_fraction=args.hint_fraction,
        judge_model=args.judge_model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        data_root=args.data_root,
        output_dir=args.output_dir,
        limit=args.limit,
        rejudge_mode=args.rejudge_mode,
        request_timeout=args.request_timeout,
        concurrency=args.concurrency,
        write_reused=args.write_reused,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
    )
    _log(f"[rejudge_hle_partition] summary={summary}")


if __name__ == "__main__":
    main()


"""
Example:

python -m runs.rejudge_hle_partition \
    --model Qwen/Qwen2.5-14B-Instruct \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --hint-fraction 0.9 \
    --judge-model gpt-5.4-nano-2026-03-17 \
    --reasoning-effort medium \
    --limit 100
"""
