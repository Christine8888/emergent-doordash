from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.datasets import Problem, get_dataset_spec
from src.hint_fractioners import fraction_hint
from src.storage import (
    append_jsonl,
    build_expanded_hinted_prompt_path,
    build_hint_generation_path,
    build_hinted_inference_path,
    make_stable_id,
    read_jsonl,
    write_jsonl,
)
from src.types import (
    ExpandedHintedPromptRecord,
    GraderResult,
    HintGenerationRecord,
    HintedInferenceRecord,
)

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_fraction(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError("hint_fraction must be in [0.0, 1.0]")
    return float(f"{value:.6f}")


def _default_prompt(*, question: str, hint_text: str) -> str:
    instructions = (
        "You will be given a problem and a hint to the problem.\n"
        "Put your final answer within \\boxed{}.\n"
        "The answer is an integer between 0 and 999 inclusive."
    )
    prompt = f"{instructions}\n\nProblem:\n{question.strip()}"
    if hint_text.strip():
        prompt += f"\n\nHint:\n{hint_text.strip()}"
    return prompt


def _inference_id(
    *,
    benchmark_name: str,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    hint_id: str,
) -> str:
    return make_stable_id(
        benchmark_name,
        model,
        hint_type,
        fractioner,
        f"{hint_fraction:.6f}",
        hint_id,
        length=24,
    )


def _prompt_id(
    *,
    benchmark_name: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    hint_id: str,
) -> str:
    return make_stable_id(
        benchmark_name,
        hint_type,
        fractioner,
        f"{hint_fraction:.6f}",
        hint_id,
        length=24,
    )


def _safe_usage_tokens(usage: Any, *keys: str) -> int:
    if usage is None:
        return 0

    for key in keys:
        if hasattr(usage, key):
            value = getattr(usage, key)
            if isinstance(value, int):
                return value
        if isinstance(usage, dict):
            value = usage.get(key)
            if isinstance(value, int):
                return value
    return 0


def _extract_stop_reason(response: Any) -> str | None:
    for attr in ("stop_reason", "finish_reason", "reason"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first = choices[0]
        finish_reason = getattr(first, "finish_reason", None)
        if isinstance(finish_reason, str):
            return finish_reason
    return None


def _extract_last_boxed_expression(text: str) -> str | None:
    marker = "\\boxed"
    start = text.rfind(marker)
    while start != -1:
        i = start + len(marker)
        n = len(text)
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            start = text.rfind(marker, 0, start)
            continue

        if text[i] == "{":
            depth = 0
            j = i
            while j < n:
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        inner = text[i + 1 : j].strip()
                        return inner or None
                j += 1
        else:
            j = i
            while j < n and not text[j].isspace():
                j += 1
            inner = text[i:j].strip()
            if inner:
                return inner

        start = text.rfind(marker, 0, start)
    return None


def _read_existing_inference_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            inference_id = row.get("inference_id")
            if isinstance(inference_id, str):
                ids.add(inference_id)
    return ids


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _checkpoint_path_for_output(output_path: Path) -> Path:
    if output_path.suffix == ".jsonl":
        return output_path.with_suffix(".ckpt.json")
    return output_path.with_name(output_path.name + ".ckpt.json")


def _save_checkpoint(
    *,
    ckpt_path: Path,
    benchmark_name: str,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    output_path: Path,
    total_candidates: int,
    existing_records: int,
    skipped_existing: int,
    processed_this_run: int,
    written_success: int,
    written_error: int,
    remaining: int,
    run_id: str,
    last_inference_id: str | None,
    run_metadata: dict[str, Any] | None = None,
    retry_count: int | None = None,
    total_input_tokens: int | None = None,
    total_output_tokens: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    payload = {
        "updated_at": _utcnow_iso(),
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "model": model,
        "hint_type": hint_type,
        "fractioner": fractioner,
        "hint_fraction": hint_fraction,
        "output_path": str(output_path),
        "total_candidates": total_candidates,
        "existing_records": existing_records,
        "skipped_existing": skipped_existing,
        "processed_this_run": processed_this_run,
        "written_success": written_success,
        "written_error": written_error,
        "remaining": remaining,
        "last_inference_id": last_inference_id,
    }
    if retry_count is not None:
        payload["retry_count"] = retry_count
    if total_input_tokens is not None:
        payload["total_input_tokens"] = total_input_tokens
    if total_output_tokens is not None:
        payload["total_output_tokens"] = total_output_tokens
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    if run_metadata is not None:
        payload["run_metadata"] = run_metadata
    _atomic_write_json(ckpt_path, payload)


def _load_hints(benchmark_name: str, hint_type: str, *, data_root: str | Path) -> list[HintGenerationRecord]:
    path = build_hint_generation_path(
        benchmark_name=benchmark_name,
        hint_type=hint_type,
        data_root=data_root,
    )
    rows = read_jsonl(path, model_cls=HintGenerationRecord)
    rows = [row for row in rows if isinstance(row, HintGenerationRecord)]
    rows.sort(
        key=lambda row: (
            row.problem_id,
            row.rollout_id,
            row.created_at,
            row.hint_id,
        )
    )
    return rows


def _build_success_grader(
    *,
    extracted_answer: str | None,
    is_correct: bool | None,
    dataset_name: str,
) -> GraderResult:
    return GraderResult(
        extractor_grader_type="dataset_extract_and_match",
        extracted_answer=extracted_answer,
        is_correct=is_correct,
        metadata={"dataset_spec": dataset_name},
    )


@dataclass(frozen=True)
class FractionRunSummary:
    output_path: str
    checkpoint_path: str
    hint_fraction: float
    total_candidates: int
    skipped_existing: int
    written_success: int
    written_error: int
    retry_count: int
    total_input_tokens: int
    total_output_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True)
class PromptCandidate:
    hint: HintGenerationRecord
    hint_fraction: float
    inference_id: str
    prompt_id: str
    hint_text_used: str
    prompt: str
    fraction_metadata: dict[str, Any]


@dataclass
class FractionRunState:
    hint_fraction: float
    expanded_prompt_path: Path
    output_path: Path
    ckpt_path: Path
    total_candidates: int
    existing_records: int
    skipped_existing: int
    pending_candidates: list[PromptCandidate]
    processed_this_run: int = 0
    written_success: int = 0
    written_error: int = 0
    retry_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_inference_id: str | None = None


def _build_expanded_prompt_records_for_fraction(
    *,
    benchmark_name: str,
    hint_type: str,
    fractioner: str,
    hint_fraction: float,
    hints: list[HintGenerationRecord],
    source_hint_path: Path,
) -> list[ExpandedHintedPromptRecord]:
    records: list[ExpandedHintedPromptRecord] = []
    for hint in hints:
        hint_text_used, fraction_meta = fraction_hint(
            hint_record=hint,
            fractioner_name=fractioner,
            hint_fraction=hint_fraction,
        )
        prompt = _default_prompt(question=hint.question, hint_text=hint_text_used)
        prompt_id = _prompt_id(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            hint_id=hint.hint_id,
        )
        records.append(
            ExpandedHintedPromptRecord(
                prompt_id=prompt_id,
                problem_id=hint.problem_id,
                benchmark_name=benchmark_name,
                hint_type=hint_type,
                fractioner=fractioner,
                hint_fraction=hint_fraction,
                hint_id=hint.hint_id,
                question=hint.question,
                answer=hint.answer,
                hint_text_used=hint_text_used,
                prompt=prompt,
                fraction_metadata=fraction_meta,
                hint=hint,
                metadata={
                    "prompt_version": "hinted_inference_v1",
                    "source_hint_generation_path": str(source_hint_path),
                    "source_hint_created_at": hint.created_at,
                },
            )
        )
    return records


def build_expanded_hinted_prompt_dataset(
    *,
    benchmark_name: str,
    hint_type: str,
    fractioner: str,
    hint_fractions: list[float],
    data_root: str | Path = "data",
) -> dict[float, Path]:
    normalized_fractions = [_normalize_fraction(v) for v in hint_fractions]
    normalized_fractions = sorted(set(normalized_fractions))
    if not normalized_fractions:
        raise ValueError("hint_fractions cannot be empty")

    source_hint_path = build_hint_generation_path(
        benchmark_name=benchmark_name,
        hint_type=hint_type,
        data_root=data_root,
    )
    hints = _load_hints(benchmark_name, hint_type, data_root=data_root)
    if not hints:
        raise ValueError(
            f"No hints found for benchmark={benchmark_name!r}, hint_type={hint_type!r}. "
            f"Expected file: {source_hint_path}"
        )

    fraction_paths: dict[float, Path] = {}
    per_fraction_counts: dict[str, int] = {}
    for hint_fraction in normalized_fractions:
        output_path = build_expanded_hinted_prompt_path(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            data_root=data_root,
        )
        records = _build_expanded_prompt_records_for_fraction(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            hints=hints,
            source_hint_path=source_hint_path,
        )
        write_jsonl(output_path, records)
        fraction_paths[hint_fraction] = output_path
        per_fraction_counts[f"{hint_fraction:.6f}"] = len(records)
        print(
            f"[hinted_inference] expanded prompts built fraction={hint_fraction} rows={len(records)} "
            f"path={output_path}",
            flush=True,
        )

    first_path = next(iter(fraction_paths.values()))
    manifest_path = first_path.parent / "manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "created_at": _utcnow_iso(),
            "benchmark_name": benchmark_name,
            "hint_type": hint_type,
            "fractioner": fractioner,
            "source_hint_path": str(source_hint_path),
            "hint_fractions": normalized_fractions,
            "fraction_files": {
                f"{fraction:.6f}": str(path) for fraction, path in fraction_paths.items()
            },
            "per_fraction_counts": per_fraction_counts,
        },
    )
    return fraction_paths


def _load_prompt_candidates_from_expanded(
    *,
    benchmark_name: str,
    model: str,
    hint_type: str,
    fractioner: str,
    fraction_paths: dict[float, Path],
) -> dict[float, list[PromptCandidate]]:
    candidates_by_fraction: dict[float, list[PromptCandidate]] = {}
    for hint_fraction, path in fraction_paths.items():
        rows = read_jsonl(path, model_cls=ExpandedHintedPromptRecord)
        typed_rows = [row for row in rows if isinstance(row, ExpandedHintedPromptRecord)]
        candidates: list[PromptCandidate] = []
        for row in typed_rows:
            inference_id = _inference_id(
                benchmark_name=benchmark_name,
                model=model,
                hint_type=hint_type,
                fractioner=fractioner,
                hint_fraction=hint_fraction,
                hint_id=row.hint_id,
            )
            candidates.append(
                PromptCandidate(
                    hint=row.hint,
                    hint_fraction=hint_fraction,
                    inference_id=inference_id,
                    prompt_id=row.prompt_id,
                    hint_text_used=row.hint_text_used,
                    prompt=row.prompt,
                    fraction_metadata=row.fraction_metadata,
                )
            )
        candidates_by_fraction[hint_fraction] = candidates
    return candidates_by_fraction


def _parse_prometheus_value(text: str, metric_names: list[str], *, is_counter: bool = True) -> float | None:
    for metric_name in metric_names:
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#") or not line.startswith(metric_name):
                continue
            rest = line[len(metric_name):]
            if rest and rest[0] not in ("{", " "):
                continue
            value_str = line.rsplit("}", 1)[-1].strip() if "{" in line else line.split()[-1]
            try:
                value = float(value_str)
            except ValueError:
                continue
            if is_counter:
                total += value
                found = True
            else:
                return value
        if found:
            return total
    return None


def _read_vllm_metrics(metrics_url: str) -> dict[str, float | None] | None:
    gen_names = ["vllm:generation_tokens_total", "vllm_generation_tokens_total"]
    prompt_names = ["vllm:prompt_tokens_total", "vllm_prompt_tokens_total"]
    running_names = ["vllm:num_requests_running", "vllm_num_requests_running"]
    waiting_names = ["vllm:num_requests_waiting", "vllm_num_requests_waiting"]
    try:
        with urllib.request.urlopen(metrics_url, timeout=1.0) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    return {
        "gen_tokens_total": _parse_prometheus_value(text, gen_names, is_counter=True),
        "prompt_tokens_total": _parse_prometheus_value(text, prompt_names, is_counter=True),
        "running": _parse_prometheus_value(text, running_names, is_counter=False),
        "waiting": _parse_prometheus_value(text, waiting_names, is_counter=False),
    }


def _save_fraction_checkpoint(
    *,
    state: FractionRunState,
    benchmark_name: str,
    model: str,
    hint_type: str,
    fractioner: str,
    run_id: str,
    run_metadata: dict[str, Any] | None,
    elapsed_seconds: float,
) -> None:
    remaining = len(state.pending_candidates) - state.processed_this_run
    _save_checkpoint(
        ckpt_path=state.ckpt_path,
        benchmark_name=benchmark_name,
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fraction=state.hint_fraction,
        output_path=state.output_path,
        total_candidates=state.total_candidates,
        existing_records=state.existing_records,
        skipped_existing=state.skipped_existing,
        processed_this_run=state.processed_this_run,
        written_success=state.written_success,
        written_error=state.written_error,
        remaining=remaining,
        run_id=run_id,
        last_inference_id=state.last_inference_id,
        run_metadata=run_metadata,
        retry_count=state.retry_count,
        total_input_tokens=state.total_input_tokens,
        total_output_tokens=state.total_output_tokens,
        elapsed_seconds=elapsed_seconds,
    )


def _to_fraction_summary(state: FractionRunState, *, elapsed_seconds: float) -> FractionRunSummary:
    return FractionRunSummary(
        output_path=str(state.output_path),
        checkpoint_path=str(state.ckpt_path),
        hint_fraction=state.hint_fraction,
        total_candidates=state.total_candidates,
        skipped_existing=state.skipped_existing,
        written_success=state.written_success,
        written_error=state.written_error,
        retry_count=state.retry_count,
        total_input_tokens=state.total_input_tokens,
        total_output_tokens=state.total_output_tokens,
        elapsed_seconds=elapsed_seconds,
    )


async def _run_all_candidates(
    *,
    benchmark_name: str,
    model: str,
    inspect_model_id: str,
    hint_type: str,
    fractioner: str,
    states: list[FractionRunState],
    dataset_spec: Any,
    do_sample: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
    max_tokens: int,
    max_connections: int,
    timeout_seconds: int,
    max_retries: int,
    vllm_metrics_url: str | None,
    run_id: str,
    checkpoint_every: int,
    run_metadata: dict[str, Any] | None = None,
) -> float:
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    work_items: list[tuple[FractionRunState, PromptCandidate]] = []
    for state in states:
        for candidate in state.pending_candidates:
            work_items.append((state, candidate))
    total_pending = len(work_items)
    if total_pending == 0:
        return 0.0

    started_at = time.monotonic()
    print(
        f"[hinted_inference] start model={model} pending_total={total_pending} "
        f"checkpoint_every={checkpoint_every} max_connections={max_connections} max_retries={max_retries}",
        flush=True,
    )
    progress = (
        tqdm(total=total_pending, desc=f"{model} hinted", leave=True)
        if tqdm is not None
        else None
    )
    last_metrics_poll = 0.0
    metrics_poll_interval_seconds = 5.0
    prev_metrics_gen_tokens: float | None = None
    prev_metrics_prompt_tokens: float | None = None
    prev_metrics_time: float | None = None
    latest_gen_tps: float | None = None
    latest_prompt_tps: float | None = None
    latest_waiting: float | None = None
    latest_running: float | None = None

    config_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "max_connections": max_connections,
    }
    if do_sample is not None:
        config_kwargs["do_sample"] = do_sample
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if top_p is not None:
        config_kwargs["top_p"] = top_p
    if top_k is not None:
        config_kwargs["top_k"] = top_k
    if repetition_penalty is not None:
        config_kwargs["repetition_penalty"] = repetition_penalty
    config = GenerateConfig(**config_kwargs)

    host = socket.gethostname()
    pid = os.getpid()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    global_processed = 0
    global_retry_count = 0
    global_output_tokens = 0

    async with get_model(inspect_model_id, config=config) as model_client:
        for state, candidate in work_items:
            hint = candidate.hint
            inference_id = candidate.inference_id
            state.last_inference_id = inference_id
            hint_fraction = state.hint_fraction
            fraction_meta = candidate.fraction_metadata
            hint_text_used = candidate.hint_text_used
            prompt = candidate.prompt
            model_output = ""
            input_token_count = 0
            output_token_count = 0
            stop_reason: str | None = None
            graders: list[GraderResult]
            is_error = False
            error_text: str | None = None
            attempts = 0

            for attempt_idx in range(max_retries + 1):
                attempts = attempt_idx + 1
                try:
                    response = await asyncio.wait_for(
                        model_client.generate(input=[ChatMessageUser(content=prompt)]),
                        timeout=timeout_seconds,
                    )
                    model_output = response.completion
                    usage = getattr(response, "usage", None)
                    input_token_count = _safe_usage_tokens(usage, "input_tokens", "prompt_tokens")
                    output_token_count = _safe_usage_tokens(usage, "output_tokens", "completion_tokens")
                    stop_reason = _extract_stop_reason(response)

                    extracted_answer = _extract_last_boxed_expression(model_output)
                    problem = Problem(
                        problem_id=hint.problem_id,
                        question=hint.question,
                        answer=hint.answer,
                        source=str(hint.metadata.get("problem_source", "")),
                    )
                    is_correct = dataset_spec.is_correct(extracted_answer, problem)
                    graders = [
                        _build_success_grader(
                            extracted_answer=extracted_answer,
                            is_correct=is_correct,
                            dataset_name=dataset_spec.name,
                        )
                    ]
                    state.written_success += 1
                    break
                except Exception as exc:
                    error_text = str(exc)
                    if attempt_idx < max_retries:
                        state.retry_count += 1
                        global_retry_count += 1
                        print(
                            f"[hinted_inference] retry model={model} fraction={hint_fraction} "
                            f"inference_id={inference_id} attempt={attempt_idx + 1}/{max_retries + 1} "
                            f"error={error_text}",
                            flush=True,
                        )
                        continue
                    is_error = True
                    graders = [
                        GraderResult(
                            extractor_grader_type="dataset_extract_and_match",
                            extracted_answer=None,
                            is_correct=None,
                            metadata={
                                "dataset_spec": dataset_spec.name,
                                "error": error_text,
                            },
                        )
                    ]
                    state.written_error += 1

            record = HintedInferenceRecord(
                inference_id=inference_id,
                problem_id=hint.problem_id,
                benchmark_name=benchmark_name,
                model=model,
                hint_type=hint.hint_type,
                fractioner=fractioner,
                hint_fraction=hint_fraction,
                hint_text_used=hint_text_used,
                model_output=model_output,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                cost=0.0,
                is_error=is_error,
                graders=graders,
                hint=hint,
                metadata={
                    "run_id": run_id,
                    "backend": "inspect_vllm",
                    "inspect_model_id": inspect_model_id,
                    "prompt_version": "hinted_inference_v1",
                    "prompt_id": candidate.prompt_id,
                    "expanded_prompt_path": str(state.expanded_prompt_path),
                    "prompt": prompt,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "max_tokens": max_tokens,
                    "max_connections": max_connections,
                    "timeout_seconds": timeout_seconds,
                    "vllm_metrics_url": vllm_metrics_url,
                    "fraction_metadata": fraction_meta,
                    "stop_reason": stop_reason,
                    "host": host,
                    "pid": pid,
                    "slurm_job_id": slurm_job_id,
                    "error": error_text,
                    "attempts": attempts,
                    "max_retries": max_retries,
                    "retries_used": max(0, attempts - 1),
                    "answer_extractor": "last_boxed_expression",
                    "run_metadata": run_metadata,
                },
            )
            append_jsonl(state.output_path, record)

            state.processed_this_run += 1
            state.total_input_tokens += input_token_count
            state.total_output_tokens += output_token_count
            global_processed += 1
            global_output_tokens += output_token_count

            now = time.monotonic()
            if vllm_metrics_url and (now - last_metrics_poll >= metrics_poll_interval_seconds):
                last_metrics_poll = now
                snapshot = _read_vllm_metrics(vllm_metrics_url)
                if snapshot is not None:
                    latest_waiting = snapshot.get("waiting")
                    latest_running = snapshot.get("running")
                    gen_tokens = snapshot.get("gen_tokens_total")
                    prompt_tokens = snapshot.get("prompt_tokens_total")
                    if (
                        isinstance(gen_tokens, (float, int))
                        and prev_metrics_gen_tokens is not None
                        and prev_metrics_time is not None
                    ):
                        dt = now - prev_metrics_time
                        if dt > 0:
                            latest_gen_tps = (float(gen_tokens) - prev_metrics_gen_tokens) / dt
                            if isinstance(prompt_tokens, (float, int)) and prev_metrics_prompt_tokens is not None:
                                latest_prompt_tps = (
                                    float(prompt_tokens) - prev_metrics_prompt_tokens
                                ) / dt
                    if isinstance(gen_tokens, (float, int)):
                        prev_metrics_gen_tokens = float(gen_tokens)
                    if isinstance(prompt_tokens, (float, int)):
                        prev_metrics_prompt_tokens = float(prompt_tokens)
                    prev_metrics_time = now

            elapsed_seconds = max(0.001, time.monotonic() - started_at)
            overall_output_tps = global_output_tokens / elapsed_seconds
            rate = global_processed / elapsed_seconds if global_processed > 0 else 0.0
            remaining_total = total_pending - global_processed
            eta_seconds = int(remaining_total / rate) if rate > 0 else -1
            expected_total_seconds = int(total_pending / rate) if rate > 0 else -1
            eta_text = f"{eta_seconds}s" if eta_seconds >= 0 else "?"
            expected_total_text = f"{expected_total_seconds}s" if expected_total_seconds >= 0 else "?"

            if progress is not None:
                postfix: dict[str, Any] = {
                    "done": f"{global_processed}/{total_pending}",
                    "retry": global_retry_count,
                    "out_tok/s": f"{overall_output_tps:.1f}",
                    "eta": eta_text,
                    "exp_total": expected_total_text,
                }
                if latest_gen_tps is not None:
                    postfix["gen_tok/s"] = f"{latest_gen_tps:.1f}"
                if latest_prompt_tps is not None:
                    postfix["prompt_tok/s"] = f"{latest_prompt_tps:.1f}"
                if latest_running is not None:
                    postfix["running"] = int(latest_running)
                if latest_waiting is not None:
                    postfix["waiting"] = int(latest_waiting)
                progress.update(1)
                progress.set_postfix(postfix)
            else:
                print(
                    f"[hinted_inference] processed={global_processed}/{total_pending} "
                    f"retry={global_retry_count} out_tok/s={overall_output_tps:.1f} eta={eta_text}",
                    flush=True,
                )

            if global_processed % checkpoint_every == 0 or remaining_total == 0:
                for checkpoint_state in states:
                    _save_fraction_checkpoint(
                        state=checkpoint_state,
                        benchmark_name=benchmark_name,
                        model=model,
                        hint_type=hint_type,
                        fractioner=fractioner,
                        run_id=run_id,
                        run_metadata=run_metadata,
                        elapsed_seconds=elapsed_seconds,
                    )
                print(
                    f"[hinted_inference] checkpoint chunk processed={global_processed}/{total_pending} "
                    f"eta={eta_text} expected_total={expected_total_text}",
                    flush=True,
                )

    if progress is not None:
        progress.close()
    return max(0.001, time.monotonic() - started_at)


def run_hinted_inference(
    *,
    benchmark_name: str,
    hint_type: str,
    model: str,
    inspect_model_id: str,
    fractioner: str,
    hint_fractions: list[float],
    data_root: str | Path = "data",
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    max_tokens: int = 32768,
    max_connections: int = 32,
    timeout_seconds: int = 3600,
    max_retries: int = 2,
    checkpoint_every: int = 25,
    vllm_metrics_url: str | None = None,
    build_only: bool = False,
    run_metadata: dict[str, Any] | None = None,
) -> list[FractionRunSummary]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if max_connections < 1:
        raise ValueError("max_connections must be >= 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    normalized_fractions = [_normalize_fraction(v) for v in hint_fractions]
    normalized_fractions = sorted(set(normalized_fractions))
    if not normalized_fractions:
        raise ValueError("hint_fractions cannot be empty")

    dataset_spec = get_dataset_spec(benchmark_name)
    fraction_paths = build_expanded_hinted_prompt_dataset(
        benchmark_name=benchmark_name,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fractions=normalized_fractions,
        data_root=data_root,
    )

    run_id = make_stable_id(
        benchmark_name,
        hint_type,
        model,
        _utcnow_iso(),
        length=12,
    )

    candidates_by_fraction = _load_prompt_candidates_from_expanded(
        benchmark_name=benchmark_name,
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        fraction_paths=fraction_paths,
    )
    effective_run_metadata: dict[str, Any] = dict(run_metadata) if run_metadata is not None else {}
    effective_run_metadata["expanded_prompt_dataset"] = {
        "data_root": str(data_root),
        "benchmark_name": benchmark_name,
        "hint_type": hint_type,
        "fractioner": fractioner,
        "hint_fractions": normalized_fractions,
        "fraction_files": {
            f"{fraction:.6f}": str(fraction_paths[fraction]) for fraction in normalized_fractions
        },
    }
    effective_run_metadata["build_only"] = build_only

    states: list[FractionRunState] = []
    for hint_fraction in normalized_fractions:
        output_path = build_hinted_inference_path(
            benchmark_name=benchmark_name,
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            data_root=data_root,
        )
        ckpt_path = _checkpoint_path_for_output(output_path)
        existing_ids = _read_existing_inference_ids(output_path)
        candidates = candidates_by_fraction[hint_fraction]
        pending_candidates = [
            candidate for candidate in candidates if candidate.inference_id not in existing_ids
        ]
        state = FractionRunState(
            hint_fraction=hint_fraction,
            expanded_prompt_path=fraction_paths[hint_fraction],
            output_path=output_path,
            ckpt_path=ckpt_path,
            total_candidates=len(candidates),
            existing_records=len(existing_ids),
            skipped_existing=len(candidates) - len(pending_candidates),
            pending_candidates=pending_candidates,
        )
        _save_fraction_checkpoint(
            state=state,
            benchmark_name=benchmark_name,
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            run_id=run_id,
            run_metadata=effective_run_metadata,
            elapsed_seconds=0.0,
        )
        if not pending_candidates:
            print(
                f"[hinted_inference] fraction={hint_fraction} model={model} "
                f"skip(all existing) total={len(candidates)} output={output_path}",
                flush=True,
            )
        states.append(state)

    if build_only:
        print(
            f"[hinted_inference] build_only=true: expanded prompts prepared for model={model}; "
            "skipping generation",
            flush=True,
        )
        return [
            _to_fraction_summary(
                state,
                elapsed_seconds=0.0,
            )
            for state in states
        ]

    elapsed_seconds = asyncio.run(
        _run_all_candidates(
            benchmark_name=benchmark_name,
            model=model,
            inspect_model_id=inspect_model_id,
            hint_type=hint_type,
            fractioner=fractioner,
            states=states,
            dataset_spec=dataset_spec,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            max_connections=max_connections,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            vllm_metrics_url=vllm_metrics_url,
            run_id=run_id,
            checkpoint_every=checkpoint_every,
            run_metadata=effective_run_metadata,
        )
    )

    summaries = [
        _to_fraction_summary(
            state,
            elapsed_seconds=elapsed_seconds if state.pending_candidates else 0.0,
        )
        for state in states
    ]
    return summaries
