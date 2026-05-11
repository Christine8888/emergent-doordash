from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
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


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    strict: Literal[True] = True


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


def _judge_exact_match(
    *,
    question: str,
    correct_answer: str,
    response: str,
    judge_model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    from openai import LengthFinishReasonError, OpenAI, OpenAIError

    prompt = build_hle_judge_prompt(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    base_metadata = {
        "grader_type": "hle_rejudge_llm",
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort,
        "judge_max_completion_tokens": max_completion_tokens,
        "judge_prompt": prompt,
    }
    try:
        client = OpenAI()
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


def _default_output_paths(
    *,
    output_dir: Path,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    judge_model: str,
) -> tuple[Path, Path]:
    partition = (
        f"{_safe_component(model)}__{_safe_component(hint_type)}__"
        f"{_safe_component(fractioner)}__fraction_{_fraction_text(hint_fraction)}"
    )
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
    write_reused: bool,
) -> dict[str, Any]:
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
    original_score_values: list[bool] = []
    comparison_score_values: list[bool] = []
    rejudged_old_values: list[bool] = []
    rejudged_new_values: list[bool] = []
    agreement_values: list[bool] = []

    attempted = 0
    missing_original_grade = 0
    skipped_not_selected = 0
    rejudge_errors = 0

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
                comparison_score_values.append(old_is_correct)
            if write_reused:
                comparison_rows.append(
                    {
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
                )
            continue

        if limit is not None and attempted >= limit:
            if _is_bool(old_is_correct):
                comparison_score_values.append(old_is_correct)
            continue

        hint = inference_row.get("hint")
        hint = hint if isinstance(hint, dict) else {}
        result = _judge_exact_match(
            question=str(hint.get("question") or ""),
            correct_answer=str(hint.get("answer") or ""),
            response=str(inference_row.get("model_output") or ""),
            judge_model=judge_model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
        )
        attempted += 1

        new_is_correct = result.get("is_correct")
        if not _is_bool(new_is_correct):
            rejudge_errors += 1
            if _is_bool(old_is_correct):
                comparison_score_values.append(old_is_correct)
        else:
            comparison_score_values.append(new_is_correct)

        agreement = None
        if _is_bool(old_is_correct) and _is_bool(new_is_correct):
            agreement = old_is_correct == new_is_correct
            agreement_values.append(agreement)
            rejudged_old_values.append(old_is_correct)
            rejudged_new_values.append(new_is_correct)

        comparison_rows.append(
            {
                "comparison_id": make_stable_id(
                    inference_id,
                    judge_model,
                    reasoning_effort,
                    length=16,
                ),
                "inference_id": inference_id,
                "problem_id": inference_row.get("problem_id"),
                "model": inference_row.get("model"),
                "hint_type": inference_row.get("hint_type"),
                "fractioner": inference_row.get("fractioner"),
                "hint_fraction": inference_row.get("hint_fraction"),
                "rejudged": True,
                "old_is_correct": old_is_correct,
                "new_is_correct": new_is_correct,
                "agreement": agreement,
                "old_extracted_answer": original_grade.get("extracted_answer"),
                "new_extracted_answer": result.get("extracted_answer"),
                "old_grader_type": original_grade.get("grader_type"),
                "new_grader_type": (result.get("metadata") or {}).get("grader_type"),
                "old_metadata": original_grade.get("metadata", {}),
                "new_metadata": result.get("metadata", {}),
                "created_at": _utcnow_iso(),
            }
        )

    summary = {
        "model": model,
        "hint_type": hint_type,
        "fractioner": fractioner,
        "hint_fraction": hint_fraction,
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort,
        "judge_max_completion_tokens": max_completion_tokens,
        "rejudge_mode": rejudge_mode,
        "inference_path": str(inference_path),
        "grade_path": str(grade_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "inference_rows": len(inference_rows),
        "original_grade_rows": len(original_grades),
        "missing_original_grade": missing_original_grade,
        "skipped_not_selected": skipped_not_selected,
        "rejudged": attempted,
        "rejudge_errors": rejudge_errors,
        "agreement_n": len(agreement_values),
        "agreement": _mean_bool(agreement_values),
        "original_partition_score": _mean_bool(original_score_values),
        "comparison_partition_score": _mean_bool(comparison_score_values),
        "original_rejudged_score": _mean_bool(rejudged_old_values),
        "comparison_rejudged_score": _mean_bool(rejudged_new_values),
        "created_at": _utcnow_iso(),
    }

    write_jsonl(output_path, comparison_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
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
        write_reused=args.write_reused,
    )
    _log(f"[rejudge_hle_partition] summary={summary}")


if __name__ == "__main__":
    main()


"""
Example:

python -m runs.rejudge_hle_partition \
    --model Qwen3.5-9B \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --hint-fraction 0.1 \
    --judge-model gpt-5.4-nano-2026-03-17 \
    --limit 100
"""
