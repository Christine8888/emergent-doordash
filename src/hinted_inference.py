from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.datasets import Problem, get_dataset_spec
from src.hint_fractioners import fraction_hint
from src.storage import (
    _model_storage_component,
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


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


_load_project_env()

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_fraction(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError("hint_fraction must be in [0.0, 1.0]")
    return float(f"{value:.6f}")


def _extract_allowed_max_tokens_from_error(error_text: str) -> int | None:
    patterns = [
        r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)",
        r"maximum context length is\s*(\d+)\s*and your request has\s*(\d+)\s*input tokens",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text)
        if match is None:
            continue
        numbers = [int(group) for group in match.groups()]
        if len(numbers) == 3:
            _, context_limit, input_tokens = numbers
            return max(1, context_limit - input_tokens)
        if len(numbers) == 2:
            context_limit, input_tokens = numbers
            return max(1, context_limit - input_tokens)
    return None


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
    canonical_model = _model_storage_component(model)
    return make_stable_id(
        benchmark_name,
        canonical_model,
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
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                finish_reason = first.get("finish_reason")
                if isinstance(finish_reason, str):
                    return finish_reason
        for key in ("stop_reason", "finish_reason", "reason"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        return None

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


def _extract_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)

    text = first.get("text")
    if isinstance(text, str):
        return text
    return ""


def _extract_reasoning_text(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None

    message = first.get("message")
    if not isinstance(message, dict):
        return None

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        parts: list[str] = []
        for part in reasoning:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)

    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "reasoning_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    return None


def _vllm_chat_completion(
    *,
    chat_completions_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    do_sample: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if do_sample is False:
        payload["temperature"] = 0.0
        payload["top_p"] = 1.0
    else:
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        chat_completions_url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from vLLM: {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse vLLM response as JSON: {body[:800]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected vLLM response type: {type(parsed).__name__}")
    return parsed


async def _chat_completion_request(
    *,
    client: httpx.AsyncClient,
    chat_completions_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    do_sample: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if do_sample is False:
        payload["temperature"] = 0.0
        payload["top_p"] = 1.0
    else:
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = await client.post(
            chat_completions_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"HTTP {exc.response.status_code} from chat completions: {exc.response.text[:800]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"chat completions request failed: {exc}") from exc

    try:
        parsed = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse chat completions response as JSON: {response.text[:800]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected chat completions response type: {type(parsed).__name__}")
    return parsed


def _extract_last_boxed_expression(text: str) -> str | None:
    markers = ["\\boxed", "\\fbox"]
    search_end = len(text)
    while True:
        best_start = -1
        best_marker = ""
        for marker in markers:
            idx = text.rfind(marker, 0, search_end)
            if idx > best_start:
                best_start = idx
                best_marker = marker
        if best_start == -1:
            return None

        i = best_start + len(best_marker)
        n = len(text)
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            search_end = best_start
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

        search_end = best_start


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


@dataclass(frozen=True)
class InferenceBackendConfig:
    backend_label: str
    provider_label: str
    chat_completions_url: str
    api_key: str | None
    metrics_url: str | None
    base_url: str


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
    out_names = ["vllm:generation_tokens_total", "vllm_generation_tokens_total"]
    prompt_names = ["vllm:prompt_tokens_total", "vllm_prompt_tokens_total"]
    running_names = ["vllm:num_requests_running", "vllm_num_requests_running"]
    waiting_names = ["vllm:num_requests_waiting", "vllm_num_requests_waiting"]
    try:
        with urllib.request.urlopen(metrics_url, timeout=1.0) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    return {
        "out_tokens_total": _parse_prometheus_value(text, out_names, is_counter=True),
        "prompt_tokens_total": _parse_prometheus_value(text, prompt_names, is_counter=True),
        "running": _parse_prometheus_value(text, running_names, is_counter=False),
        "waiting": _parse_prometheus_value(text, waiting_names, is_counter=False),
    }


def _format_eta_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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


def _resolve_backend_config(
    *,
    backend: str,
    vllm_metrics_url: str | None,
) -> InferenceBackendConfig:
    if backend == "local-vllm":
        vllm_base_url = os.environ.get("VLLM_BASE_URL")
        if not vllm_base_url:
            raise RuntimeError(
                "VLLM_BASE_URL is required for local-vllm hinted inference. "
                "Expected format like http://localhost:8000/v1"
            )
        return InferenceBackendConfig(
            backend_label="local_vllm",
            provider_label="vllm_openai_compat",
            chat_completions_url=f"{vllm_base_url.rstrip('/')}/chat/completions",
            api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            metrics_url=vllm_metrics_url,
            base_url=vllm_base_url,
        )
    if backend == "together-serverless":
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is required for together-serverless hinted inference.")
        base_url = "https://api.together.xyz/v1"
        return InferenceBackendConfig(
            backend_label="together_serverless",
            provider_label="together_openai_compat",
            chat_completions_url=f"{base_url}/chat/completions",
            api_key=api_key,
            metrics_url=None,
            base_url=base_url,
        )
    raise ValueError(f"Unsupported backend: {backend!r}")


async def _run_all_candidates(
    *,
    benchmark_name: str,
    model: str,
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
    backend: str,
    run_id: str,
    checkpoint_every: int,
    run_metadata: dict[str, Any] | None = None,
) -> float:
    canonical_model = _model_storage_component(model)
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
        tqdm(
            total=total_pending,
            desc=f"{model} hinted",
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
        if tqdm is not None
        else None
    )
    last_metrics_poll = 0.0
    metrics_poll_interval_seconds = 1.0
    prev_metrics_out_tokens: float | None = None
    prev_metrics_prompt_tokens: float | None = None
    prev_metrics_time: float | None = None
    latest_out_tps: float | None = None
    latest_prompt_tps: float | None = None
    latest_waiting: float | None = None
    latest_running: float | None = None
    smoothed_eta_seconds: float | None = None
    eta_ewma_alpha = 0.2

    backend_config = _resolve_backend_config(
        backend=backend,
        vllm_metrics_url=vllm_metrics_url,
    )

    host = socket.gethostname()
    pid = os.getpid()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    global_processed = 0
    global_retry_count = 0
    global_input_tokens = 0
    global_output_tokens = 0

    semaphore = asyncio.Semaphore(max_connections)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds + 5, connect=min(timeout_seconds, 30)),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
    )

    async def _generate_one(
        item_state: FractionRunState, item_candidate: PromptCandidate
    ) -> dict[str, Any]:
        async with semaphore:
            hint = item_candidate.hint
            hint_fraction = item_state.hint_fraction
            prompt = item_candidate.prompt
            model_output = ""
            input_token_count = 0
            output_token_count = 0
            stop_reason: str | None = None
            reasoning_text: str | None = None
            graders: list[GraderResult]
            is_error = False
            error_text: str | None = None
            attempts = 0
            request_max_tokens = max_tokens
            effective_max_tokens_used = max_tokens

            for attempt_idx in range(max_retries + 1):
                attempts = attempt_idx + 1
                try:
                    response = await _chat_completion_request(
                        client=client,
                        chat_completions_url=backend_config.chat_completions_url,
                        api_key=backend_config.api_key,
                        model=model,
                        prompt=prompt,
                        max_tokens=request_max_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                    effective_max_tokens_used = request_max_tokens
                    model_output = _extract_completion_text(response)
                    reasoning_text = _extract_reasoning_text(response)
                    usage = response.get("usage") if isinstance(response, dict) else None
                    input_token_count = _safe_usage_tokens(usage, "prompt_tokens", "input_tokens")
                    output_token_count = _safe_usage_tokens(
                        usage, "completion_tokens", "output_tokens"
                    )
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
                    break
                except Exception as exc:
                    error_text = str(exc)
                    allowed_max_tokens = _extract_allowed_max_tokens_from_error(error_text)
                    if (
                        allowed_max_tokens is not None
                        and allowed_max_tokens < request_max_tokens
                        and attempt_idx < max_retries
                    ):
                        request_max_tokens = allowed_max_tokens
                        print(
                            f"[hinted_inference] clip_max_tokens model={model} "
                            f"fraction={hint_fraction} inference_id={item_candidate.inference_id} "
                            f"attempt={attempt_idx + 1}/{max_retries + 1} "
                            f"new_max_tokens={request_max_tokens}",
                            flush=True,
                        )
                        continue
                    if attempt_idx < max_retries:
                        print(
                            f"[hinted_inference] retry model={model} fraction={hint_fraction} "
                            f"inference_id={item_candidate.inference_id} attempt={attempt_idx + 1}/{max_retries + 1} "
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

            return {
                "state": item_state,
                "candidate": item_candidate,
                "model_output": model_output,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "stop_reason": stop_reason,
                "reasoning_text": reasoning_text,
                "is_error": is_error,
                "error_text": error_text,
                "attempts": attempts,
                "retries_used": max(0, attempts - 1),
                "effective_max_tokens_used": effective_max_tokens_used,
                "graders": graders,
            }

    work_iter = iter(work_items)
    pending_tasks: set[asyncio.Task[dict[str, Any]]] = set()

    def _enqueue_next() -> bool:
        try:
            next_state, next_candidate = next(work_iter)
        except StopIteration:
            return False
        pending_tasks.add(asyncio.create_task(_generate_one(next_state, next_candidate)))
        return True

    initial = min(max_connections, total_pending)
    for _ in range(initial):
        _enqueue_next()

    try:
        while pending_tasks:
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                result = task.result()
                state = result["state"]
                candidate = result["candidate"]
                hint = candidate.hint
                inference_id = candidate.inference_id
                state.last_inference_id = inference_id
                hint_fraction = state.hint_fraction
                fraction_meta = candidate.fraction_metadata
                hint_text_used = candidate.hint_text_used

                model_output = str(result["model_output"])
                input_token_count = int(result["input_token_count"])
                output_token_count = int(result["output_token_count"])
                stop_reason = result["stop_reason"]
                reasoning_text = result["reasoning_text"]
                is_error = bool(result["is_error"])
                error_text = result["error_text"]
                attempts = int(result["attempts"])
                retries_used = int(result["retries_used"])
                effective_max_tokens_used = int(result["effective_max_tokens_used"])
                graders = result["graders"]

                state.retry_count += retries_used
                global_retry_count += retries_used
                if is_error:
                    state.written_error += 1
                else:
                    state.written_success += 1

                record = HintedInferenceRecord(
                    inference_id=inference_id,
                    problem_id=hint.problem_id,
                    benchmark_name=benchmark_name,
                    model=canonical_model,
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
                        "backend": backend_config.backend_label,
                        "prompt_version": "hinted_inference_v1",
                        "prompt_id": candidate.prompt_id,
                        "expanded_prompt_path": str(state.expanded_prompt_path),
                        "prompt": candidate.prompt,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "max_tokens": max_tokens,
                        "effective_max_tokens_used": effective_max_tokens_used,
                        "max_connections": max_connections,
                        "timeout_seconds": timeout_seconds,
                        "provider": backend_config.provider_label,
                        "provider_base_url": backend_config.base_url,
                        "provider_metrics_url": backend_config.metrics_url,
                        "provider_model_id": model,
                        "fraction_metadata": fraction_meta,
                        "stop_reason": stop_reason,
                        "provider_reasoning": reasoning_text,
                        "host": host,
                        "pid": pid,
                        "slurm_job_id": slurm_job_id,
                        "error": error_text,
                        "attempts": attempts,
                        "max_retries": max_retries,
                        "retries_used": retries_used,
                        "answer_extractor": "last_boxed_expression",
                        "run_metadata": run_metadata,
                    },
                )
                append_jsonl(state.output_path, record)

                state.processed_this_run += 1
                state.total_input_tokens += input_token_count
                state.total_output_tokens += output_token_count
                global_processed += 1
                global_input_tokens += input_token_count
                global_output_tokens += output_token_count

                now = time.monotonic()
                if backend_config.metrics_url and (now - last_metrics_poll >= metrics_poll_interval_seconds):
                    last_metrics_poll = now
                    snapshot = _read_vllm_metrics(backend_config.metrics_url)
                    if snapshot is not None:
                        latest_waiting = snapshot.get("waiting")
                        latest_running = snapshot.get("running")
                        out_tokens = snapshot.get("out_tokens_total")
                        prompt_tokens = snapshot.get("prompt_tokens_total")
                        if (
                            isinstance(out_tokens, (float, int))
                            and prev_metrics_out_tokens is not None
                            and prev_metrics_time is not None
                        ):
                            dt = now - prev_metrics_time
                            if dt > 0:
                                latest_out_tps = (float(out_tokens) - prev_metrics_out_tokens) / dt
                                if (
                                    isinstance(prompt_tokens, (float, int))
                                    and prev_metrics_prompt_tokens is not None
                                ):
                                    latest_prompt_tps = (
                                        float(prompt_tokens) - prev_metrics_prompt_tokens
                                    ) / dt
                        if isinstance(out_tokens, (float, int)):
                            prev_metrics_out_tokens = float(out_tokens)
                        if isinstance(prompt_tokens, (float, int)):
                            prev_metrics_prompt_tokens = float(prompt_tokens)
                        prev_metrics_time = now

                remaining_total = total_pending - global_processed
                eta_seconds: float | None = None
                if (
                    global_processed > 0
                    and remaining_total > 0
                    and latest_out_tps is not None
                    and latest_prompt_tps is not None
                    and latest_out_tps > 0
                    and latest_prompt_tps > 0
                ):
                    avg_input_tokens = global_input_tokens / global_processed
                    avg_output_tokens = global_output_tokens / global_processed
                    remaining_prompt_tokens = remaining_total * avg_input_tokens
                    remaining_output_tokens = remaining_total * avg_output_tokens
                    eta_seconds = (
                        remaining_prompt_tokens / latest_prompt_tps
                        + remaining_output_tokens / latest_out_tps
                    )
                    if smoothed_eta_seconds is None:
                        smoothed_eta_seconds = eta_seconds
                    else:
                        smoothed_eta_seconds = (
                            eta_ewma_alpha * eta_seconds
                            + (1.0 - eta_ewma_alpha) * smoothed_eta_seconds
                        )
                elif remaining_total == 0:
                    smoothed_eta_seconds = 0.0

                if progress is not None:
                    postfix: dict[str, Any] = {"retry": global_retry_count}
                    if smoothed_eta_seconds is not None:
                        postfix["eta"] = _format_eta_seconds(smoothed_eta_seconds)
                    if latest_out_tps is not None:
                        postfix["out_tok/s"] = f"{latest_out_tps:.1f}"
                    if latest_prompt_tps is not None:
                        postfix["prompt_tok/s"] = f"{latest_prompt_tps:.1f}"
                    if latest_running is not None:
                        postfix["running"] = int(latest_running)
                    if latest_waiting is not None:
                        postfix["waiting"] = int(latest_waiting)
                    progress.update(1)
                    progress.set_postfix(postfix)
                else:
                    status_parts = [
                        f"[hinted_inference] processed={global_processed}/{total_pending}",
                        f"retry={global_retry_count}",
                    ]
                    if smoothed_eta_seconds is not None:
                        status_parts.append(f"eta={_format_eta_seconds(smoothed_eta_seconds)}")
                    if latest_out_tps is not None:
                        status_parts.append(f"out_tok/s={latest_out_tps:.1f}")
                    if latest_prompt_tps is not None:
                        status_parts.append(f"prompt_tok/s={latest_prompt_tps:.1f}")
                    if latest_running is not None:
                        status_parts.append(f"running={int(latest_running)}")
                    if latest_waiting is not None:
                        status_parts.append(f"waiting={int(latest_waiting)}")
                    print(
                        " ".join(status_parts),
                        flush=True,
                    )

                elapsed_seconds = max(0.001, time.monotonic() - started_at)
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
                        f"[hinted_inference] checkpoint chunk processed={global_processed}/{total_pending}",
                        flush=True,
                    )

                _enqueue_next()
    finally:
        for task in pending_tasks:
            task.cancel()
        with contextlib.suppress(Exception):
            await client.aclose()
        if progress is not None:
            progress.close()
    return max(0.001, time.monotonic() - started_at)


def run_hinted_inference(
    *,
    benchmark_name: str,
    hint_type: str,
    model: str,
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
    max_requests: int | None = None,
    timeout_seconds: int = 3600,
    max_retries: int = 2,
    checkpoint_every: int = 25,
    vllm_metrics_url: str | None = None,
    backend: str = "local-vllm",
    build_only: bool = False,
    run_metadata: dict[str, Any] | None = None,
) -> list[FractionRunSummary]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if max_connections < 1:
        raise ValueError("max_connections must be >= 1")
    if max_requests is not None and max_requests < 1:
        raise ValueError("max_requests must be >= 1")
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
    effective_run_metadata["max_requests"] = max_requests

    states: list[FractionRunState] = []
    remaining_request_budget = max_requests
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
        full_pending_candidates = [
            candidate for candidate in candidates if candidate.inference_id not in existing_ids
        ]
        pending_candidates = full_pending_candidates
        if remaining_request_budget is not None:
            pending_candidates = full_pending_candidates[:remaining_request_budget]
            remaining_request_budget -= len(pending_candidates)
        state = FractionRunState(
            hint_fraction=hint_fraction,
            expanded_prompt_path=fraction_paths[hint_fraction],
            output_path=output_path,
            ckpt_path=ckpt_path,
            total_candidates=len(candidates),
            existing_records=len(existing_ids),
            skipped_existing=len(candidates) - len(full_pending_candidates),
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
        elif len(pending_candidates) != len(full_pending_candidates):
            print(
                f"[hinted_inference] fraction={hint_fraction} model={model} "
                f"limited pending={len(pending_candidates)}/{len(full_pending_candidates)} "
                f"max_requests={max_requests}",
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
            backend=backend,
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
