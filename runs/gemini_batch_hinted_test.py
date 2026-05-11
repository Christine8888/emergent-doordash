from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

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


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_BATCH_ROOT = DEFAULT_DATA_ROOT / "gemini_batch_hinted"
DEFAULT_MANIFEST_ROOT = DEFAULT_BATCH_ROOT / "manifests"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_MAX_OUTPUT_TOKENS = 32000
DEFAULT_TEMPERATURE = 1.0
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# Paid tier Batch API prices per 1M tokens. These were copied from the Gemini API
# pricing page on 2026-05-11; override from the CLI if Google changes prices.
GEMINI_BATCH_PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    "gemini-3.1-pro-preview": {"input": 1.00, "output": 6.00},
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _parse_fraction_values(values: list[str] | None) -> list[float]:
    if values is None:
        return [0.0]
    fractions = [float(value) for value in values]
    for fraction in fractions:
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("hint fractions must be in [0.0, 1.0]")
    return sorted({float(f"{fraction:.6f}") for fraction in fractions})


def _safe_rate_lookup(
    *,
    model: str,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> dict[str, float]:
    default = GEMINI_BATCH_PRICING_PER_MILLION.get(model)
    input_rate = input_price_per_million
    output_rate = output_price_per_million
    if input_rate is None and default is not None:
        input_rate = default["input"]
    if output_rate is None and default is not None:
        output_rate = default["output"]
    if input_rate is None or output_rate is None:
        raise ValueError(
            f"No built-in Gemini Batch pricing for model={model!r}. "
            "Pass --input-price-per-million and --output-price-per-million."
        )
    return {"input": float(input_rate), "output": float(output_rate)}


def _validate_supported_model(model: str) -> None:
    if model != DEFAULT_MODEL:
        raise ValueError(
            f"Unsupported Gemini batch test model: {model!r}. "
            f"This harness currently supports only {DEFAULT_MODEL!r}."
        )


def _estimate_text_tokens(text: str) -> int:
    # Gemini docs describe one token as roughly four characters for text.
    return max(1, math.ceil(len(text) / 4.0))


def _gemini_count_text_tokens(*, client: Any, model: str, text: str) -> int:
    response = client.models.count_tokens(model=model, contents=text)
    for attr in ("total_tokens", "totalTokens"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    plain = _to_plain(response)
    if isinstance(plain, dict):
        for key in ("total_tokens", "totalTokens"):
            value = plain.get(key)
            if isinstance(value, int):
                return value
    raise RuntimeError(f"Could not read token count from count_tokens response: {plain!r}")


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


def _hle_prompt(*, question: str, hint_text: str) -> str:
    prompt = (
        "You will be given a problem and a hint to the problem. Use the hint if helpful.\n\n"
        f"Problem:\n{question.strip()}"
    )
    if hint_text.strip():
        prompt += f"\n\nHint:\n{hint_text.strip()}"
    prompt += "\n\nGive your final answer in the following format:\nAnswer: {your answer}"
    return prompt


def _build_hinted_prompt(*, benchmark_name: str, question: str, hint_text: str) -> str:
    if benchmark_name == "hle":
        return _hle_prompt(question=question, hint_text=hint_text)
    return _default_prompt(question=question, hint_text=hint_text)


def _hint_is_text_only_hle(hint: HintGenerationRecord) -> bool:
    if hint.benchmark_name != "hle":
        return True
    problem_text_only = hint.metadata.get("problem_text_only")
    if isinstance(problem_text_only, bool):
        return problem_text_only
    problem_metadata = hint.metadata.get("problem_metadata")
    if isinstance(problem_metadata, dict) and isinstance(problem_metadata.get("text_only"), bool):
        return bool(problem_metadata["text_only"])
    return False


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


def _load_hints(
    benchmark_name: str,
    hint_type: str,
    *,
    data_root: str | Path,
) -> list[HintGenerationRecord]:
    path = build_hint_generation_path(
        benchmark_name=benchmark_name,
        hint_type=hint_type,
        data_root=data_root,
    )
    rows = read_jsonl(path, model_cls=HintGenerationRecord)
    typed_rows = [row for row in rows if isinstance(row, HintGenerationRecord)]
    typed_rows.sort(
        key=lambda row: (
            row.problem_id,
            row.rollout_id,
            row.created_at,
            row.hint_id,
        )
    )
    return typed_rows


def _build_expanded_hinted_prompt_dataset(
    *,
    benchmark_name: str,
    hint_type: str,
    fractioner: str,
    hint_fractions: list[float],
    data_root: str | Path,
) -> dict[float, Path]:
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
    for hint_fraction in sorted({float(f"{value:.6f}") for value in hint_fractions}):
        output_path = build_expanded_hinted_prompt_path(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            data_root=data_root,
        )
        records: list[ExpandedHintedPromptRecord] = []
        for hint in hints:
            if benchmark_name == "hle" and not _hint_is_text_only_hle(hint):
                continue
            hint_text_used, fraction_meta = fraction_hint(
                hint_record=hint,
                fractioner_name=fractioner,
                hint_fraction=hint_fraction,
            )
            prompt = _build_hinted_prompt(
                benchmark_name=benchmark_name,
                question=hint.question,
                hint_text=hint_text_used,
            )
            records.append(
                ExpandedHintedPromptRecord(
                    prompt_id=_prompt_id(
                        benchmark_name=benchmark_name,
                        hint_type=hint_type,
                        fractioner=fractioner,
                        hint_fraction=hint_fraction,
                        hint_id=hint.hint_id,
                    ),
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
        write_jsonl(output_path, records)
        fraction_paths[hint_fraction] = output_path
    return fraction_paths


def _extract_state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(state, str):
        return state
    if state is None:
        return "JOB_STATE_UNSPECIFIED"
    return str(state)


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if hasattr(value, "to_json_dict"):
        return _to_plain(value.to_json_dict())
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if hasattr(value, "__dict__"):
        return {
            key: _to_plain(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_filename_component(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text.strip())
    return cleaned.strip("._-") or "unknown"


def _submitted_manifest_path(
    *,
    submitted_at: datetime,
    job_name: str,
) -> Path:
    timestamp = submitted_at.strftime("%Y%m%d_%H%M%S")
    job_component = _safe_filename_component(job_name)
    return DEFAULT_MANIFEST_ROOT / f"{timestamp}__{job_component}.json"


def _iter_jsonl(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def _default_thinking_config(model: str) -> dict[str, Any] | None:
    _validate_supported_model(model)
    return {"thinkingLevel": "high", "includeThoughts": True}


def _request_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "maxOutputTokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
    }
    if args.top_p is not None:
        config["topP"] = float(args.top_p)
    if args.top_k is not None:
        config["topK"] = int(args.top_k)
    thinking_config = _default_thinking_config(args.model)
    if thinking_config is not None:
        config["thinkingConfig"] = thinking_config
    return config


def _read_existing_inference_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for row in _iter_jsonl(path):
        if isinstance(row, dict) and isinstance(row.get("inference_id"), str):
            ids.add(row["inference_id"])
    return ids


def _build_batch_paths(
    *,
    batch_root: Path,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    model: str,
    run_id: str,
) -> dict[str, Path]:
    root = (
        batch_root
        / benchmark
        / hint_type
        / fractioner
        / _model_storage_component(model)
        / run_id
    )
    return {
        "root": root,
        "input_jsonl": root / "input.jsonl",
        "request_metadata_jsonl": root / "request_metadata.jsonl",
        "manifest": root / "manifest.json",
        "downloaded_results_jsonl": root / "results.jsonl",
        "status_json": root / "status.json",
    }


def _build_jsonl(args: argparse.Namespace) -> dict[str, Any]:
    _validate_supported_model(args.model)
    data_root = Path(args.data_root)
    batch_root = Path(args.batch_root)
    hint_fractions = _parse_fraction_values(args.hint_fractions)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    generation_config = _request_generation_config(args)
    token_count_client = (
        genai.Client() if args.token_estimate_method == "gemini-count-tokens" else None
    )
    paths = _build_batch_paths(
        batch_root=batch_root,
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        model=args.model,
        run_id=run_id,
    )

    fraction_paths = _build_expanded_hinted_prompt_dataset(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        hint_fractions=hint_fractions,
        data_root=data_root,
    )

    total_rows = 0
    skipped_existing = 0
    total_prompt_token_estimate = 0
    per_fraction: dict[str, dict[str, Any]] = {}
    remaining = args.max_requests
    paths["root"].mkdir(parents=True, exist_ok=True)

    with open(paths["input_jsonl"], "w", encoding="utf-8") as input_f, open(
        paths["request_metadata_jsonl"], "w", encoding="utf-8"
    ) as metadata_f:
        for hint_fraction in hint_fractions:
            expanded_path = fraction_paths[hint_fraction]
            output_path = build_hinted_inference_path(
                benchmark_name=args.benchmark,
                model=args.model,
                hint_type=args.hint_type,
                fractioner=args.fractioner,
                hint_fraction=hint_fraction,
                data_root=data_root,
            )
            existing_ids = (
                _read_existing_inference_ids(output_path) if args.skip_existing else set()
            )
            fraction_count = 0
            fraction_skipped = 0
            rows = read_jsonl(expanded_path, model_cls=ExpandedHintedPromptRecord)
            typed_rows = [
                row for row in rows if isinstance(row, ExpandedHintedPromptRecord)
            ]
            for row in typed_rows:
                inference_id = _inference_id(
                    benchmark_name=args.benchmark,
                    model=args.model,
                    hint_type=args.hint_type,
                    fractioner=args.fractioner,
                    hint_fraction=hint_fraction,
                    hint_id=row.hint_id,
                )
                if inference_id in existing_ids:
                    skipped_existing += 1
                    fraction_skipped += 1
                    continue
                if remaining is not None and remaining <= 0:
                    continue

                key = inference_id
                if token_count_client is None:
                    prompt_token_estimate = _estimate_text_tokens(row.prompt)
                else:
                    prompt_token_estimate = _gemini_count_text_tokens(
                        client=token_count_client,
                        model=args.model,
                        text=row.prompt,
                    )
                total_prompt_token_estimate += prompt_token_estimate
                request = {
                    "key": key,
                    "request": {
                        "contents": [
                            {
                                "parts": [{"text": row.prompt}],
                            }
                        ],
                    },
                }
                metadata = {
                    "key": key,
                    "inference_id": inference_id,
                    "prompt_id": row.prompt_id,
                    "problem_id": row.problem_id,
                    "benchmark_name": row.benchmark_name,
                    "model": args.model,
                    "canonical_model": _model_storage_component(args.model),
                    "hint_type": row.hint_type,
                    "fractioner": row.fractioner,
                    "hint_fraction": row.hint_fraction,
                    "hint_id": row.hint_id,
                    "hint_text_used": row.hint_text_used,
                    "prompt": row.prompt,
                    "answer": row.answer,
                    "fraction_metadata": row.fraction_metadata,
                    "hint": row.hint.model_dump(),
                    "expanded_prompt_path": str(expanded_path),
                    "output_path": str(output_path),
                    "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                    "prompt_token_estimate": prompt_token_estimate,
                }
                input_f.write(json.dumps(request, ensure_ascii=False) + "\n")
                metadata_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                total_rows += 1
                fraction_count += 1
                if remaining is not None:
                    remaining -= 1

            per_fraction[f"{hint_fraction:.6f}"] = {
                "expanded_prompt_path": str(expanded_path),
                "output_path": str(output_path),
                "requests": fraction_count,
                "skipped_existing": fraction_skipped,
            }

    pricing = _safe_rate_lookup(
        model=args.model,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    max_output_tokens = total_rows * DEFAULT_MAX_OUTPUT_TOKENS
    estimated_cost = _calculate_cost(
        input_tokens=total_prompt_token_estimate,
        output_tokens=max_output_tokens,
        pricing=pricing,
    )
    manifest = {
        "created_at": _utcnow_iso(),
        "script": "runs.gemini_batch_hinted_test",
        "run_id": run_id,
        "model": args.model,
        "benchmark": args.benchmark,
        "hint_type": args.hint_type,
        "fractioner": args.fractioner,
        "hint_fractions": hint_fractions,
        "data_root": str(data_root),
        "batch_root": str(batch_root),
        "input_jsonl": str(paths["input_jsonl"]),
        "request_metadata_jsonl": str(paths["request_metadata_jsonl"]),
        "downloaded_results_jsonl": str(paths["downloaded_results_jsonl"]),
        "request_count": total_rows,
        "skipped_existing": skipped_existing,
        "generation_config": generation_config,
        "pricing_per_million": pricing,
        "token_estimate_method": args.token_estimate_method,
        "estimated_input_tokens": total_prompt_token_estimate,
        "max_output_tokens": max_output_tokens,
        "estimated_max_cost_usd": estimated_cost,
        "per_fraction": per_fraction,
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def _submit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = _read_json(manifest_path)
    _validate_supported_model(str(manifest["model"]))
    input_path = Path(manifest["input_jsonl"])
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_path}")
    if int(manifest.get("request_count", 0)) < 1:
        raise ValueError("Refusing to submit an empty batch JSONL.")

    _load_project_env()
    client = genai.Client()
    display_name = args.display_name or (
        f"hinted-{manifest['benchmark']}-{manifest['hint_type']}-"
        f"{manifest['fractioner']}-{manifest['run_id']}"
    )
    uploaded_file = client.files.upload(
        file=str(input_path),
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )
    batch_job = client.batches.create(
        model=str(manifest["model"]),
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    submitted_at = datetime.now(timezone.utc)
    submitted_manifest_path = _submitted_manifest_path(
        submitted_at=submitted_at,
        job_name=batch_job.name,
    )
    manifest.update(
        {
            "submitted_at": submitted_at.isoformat(),
            "display_name": display_name,
            "source_manifest_path": str(manifest_path),
            "manifest_path": str(submitted_manifest_path),
            "submitted_manifest_path": str(submitted_manifest_path),
            "uploaded_file_name": uploaded_file.name,
            "uploaded_file": _to_plain(uploaded_file),
            "batch_job_name": batch_job.name,
            "batch_job": _to_plain(batch_job),
            "batch_job_state": _extract_state_name(batch_job),
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(submitted_manifest_path, manifest)
    return manifest


def _resolve_job_name(args: argparse.Namespace) -> tuple[str, Path | None, dict[str, Any] | None]:
    if args.job_name:
        return args.job_name, None, None
    if not args.manifest:
        raise ValueError("Pass either --job-name or --manifest.")
    manifest_path = Path(args.manifest)
    manifest = _read_json(manifest_path)
    job_name = manifest.get("batch_job_name")
    if not isinstance(job_name, str) or not job_name:
        raise ValueError(f"Manifest does not contain batch_job_name: {manifest_path}")
    return job_name, manifest_path, manifest


def _status(args: argparse.Namespace) -> dict[str, Any]:
    job_name, manifest_path, manifest = _resolve_job_name(args)
    _load_project_env()
    client = genai.Client()
    batch_job = client.batches.get(name=job_name)
    state = _extract_state_name(batch_job)
    payload = {
        "checked_at": _utcnow_iso(),
        "batch_job_name": job_name,
        "state": state,
        "batch_job": _to_plain(batch_job),
    }
    if manifest_path is not None and manifest is not None:
        manifest["last_status"] = payload
        manifest["batch_job_state"] = state
        _write_json(manifest_path, manifest)
        status_path = Path(manifest_path).with_name("status.json")
        _write_json(status_path, payload)
    return payload


def _wait(args: argparse.Namespace) -> dict[str, Any]:
    interval_seconds = max(1, int(args.poll_interval_seconds))
    while True:
        payload = _status(args)
        state = str(payload["state"])
        checked_at = datetime.now().strftime("%H:%M")
        print(
            f"[gemini_batch] {checked_at} state={state} job={payload['batch_job_name']}",
            flush=True,
        )
        if state in COMPLETED_STATES:
            return payload
        time.sleep(interval_seconds)


def _download(args: argparse.Namespace) -> dict[str, Any]:
    job_name, manifest_path, manifest = _resolve_job_name(args)
    _load_project_env()
    client = genai.Client()
    batch_job = client.batches.get(name=job_name)
    state = _extract_state_name(batch_job)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Job is not succeeded; state={state}")
    dest = getattr(batch_job, "dest", None)
    result_file_name = getattr(dest, "file_name", None)
    if not isinstance(result_file_name, str) or not result_file_name:
        raise RuntimeError("Succeeded job does not expose dest.file_name.")

    output_path = Path(args.output_path) if args.output_path else None
    if output_path is None:
        if manifest is None:
            output_path = Path("gemini_batch_results.jsonl")
        else:
            output_path = Path(manifest["downloaded_results_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_content = client.files.download(file=result_file_name)
    if isinstance(file_content, str):
        result_bytes = file_content.encode("utf-8")
    else:
        result_bytes = bytes(file_content)
    with open(output_path, "wb") as f:
        f.write(result_bytes)

    payload = {
        "downloaded_at": _utcnow_iso(),
        "batch_job_name": job_name,
        "state": state,
        "result_file_name": result_file_name,
        "downloaded_results_jsonl": str(output_path),
        "bytes": len(result_bytes),
    }
    if manifest_path is not None and manifest is not None:
        manifest.update(payload)
        manifest["batch_job"] = _to_plain(batch_job)
        _write_json(manifest_path, manifest)
    return payload


def _candidate_text_parts(response: dict[str, Any]) -> tuple[str, str]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "", ""
    first = candidates[0]
    if not isinstance(first, dict):
        return "", ""
    content = first.get("content")
    if not isinstance(content, dict):
        return "", ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return "", ""
    answer_chunks: list[str] = []
    thought_chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            if part.get("thought") is True:
                thought_chunks.append(part["text"])
            else:
                answer_chunks.append(part["text"])
    return "".join(answer_chunks), "".join(thought_chunks)


def _candidate_finish_reason(response: dict[str, Any]) -> str | None:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    for key in ("finishReason", "finish_reason"):
        value = first.get(key)
        if isinstance(value, str):
            return value
    return None


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def _extract_usage_counts(response: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _usage_int(usage, "promptTokenCount", "prompt_token_count")
    candidate_tokens = _usage_int(
        usage, "candidatesTokenCount", "candidates_token_count"
    )
    thought_tokens = _usage_int(usage, "thoughtsTokenCount", "thoughts_token_count")
    output_tokens = candidate_tokens + thought_tokens
    if output_tokens == 0:
        total_tokens = _usage_int(usage, "totalTokenCount", "total_token_count")
        if total_tokens and input_tokens:
            output_tokens = max(0, total_tokens - input_tokens)
    return input_tokens, output_tokens, usage


def _calculate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, float],
) -> float:
    return (
        input_tokens / 1_000_000.0 * pricing["input"]
        + output_tokens / 1_000_000.0 * pricing["output"]
    )


def _build_grader(
    *,
    benchmark_name: str,
    model_output: str,
    metadata: dict[str, Any],
) -> GraderResult:
    dataset_spec = get_dataset_spec(benchmark_name)
    hint = metadata["hint"]
    if dataset_spec.name == "hle":
        problem_metadata = hint.get("metadata")
        if not isinstance(problem_metadata, dict):
            problem_metadata = {}
        nested_problem_metadata = problem_metadata.get("problem_metadata")
        if not isinstance(nested_problem_metadata, dict):
            nested_problem_metadata = {}
        return GraderResult(
            extractor_grader_type="hle_pending_grader",
            extracted_answer=dataset_spec.extract_answer(model_output),
            is_correct=None,
            metadata={
                "dataset_spec": dataset_spec.name,
                "answer_type": nested_problem_metadata.get(
                    "answer_type",
                    problem_metadata.get("problem_answer_type"),
                ),
                "stage": "pending_hle_sidecar_grading",
            },
        )

    extracted_answer = _extract_last_boxed_expression(model_output)
    problem = Problem(
        problem_id=str(metadata["problem_id"]),
        question=str(hint["question"]),
        answer=str(hint["answer"]),
        source=str(hint.get("metadata", {}).get("problem_source", "")),
    )
    return GraderResult(
        extractor_grader_type="dataset_extract_and_match",
        extracted_answer=extracted_answer,
        is_correct=dataset_spec.is_correct(extracted_answer, problem),
        metadata={"dataset_spec": dataset_spec.name},
    )


def _load_request_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            continue
        rows[row["key"]] = row
    return rows


def _process_results(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(Path(args.manifest))
    _validate_supported_model(str(manifest["model"]))
    result_path = Path(args.results_jsonl or manifest["downloaded_results_jsonl"])
    metadata_path = Path(manifest["request_metadata_jsonl"])
    request_metadata = _load_request_metadata(metadata_path)
    pricing = _safe_rate_lookup(
        model=str(manifest["model"]),
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    run_id = str(manifest["run_id"])

    written_success = 0
    written_error = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    missing_metadata = 0
    duplicate_existing = 0

    for result in _iter_jsonl(result_path):
        if not isinstance(result, dict):
            written_error += 1
            continue
        key = result.get("key")
        if not isinstance(key, str) or key not in request_metadata:
            missing_metadata += 1
            continue
        metadata = request_metadata[key]
        output_path = Path(metadata["output_path"])
        inference_id = str(metadata["inference_id"])
        if args.skip_existing and inference_id in _read_existing_inference_ids(output_path):
            duplicate_existing += 1
            continue

        response = result.get("response")
        error = result.get("error")
        is_error = not isinstance(response, dict)
        if isinstance(response, dict):
            model_output, thought_summary = _candidate_text_parts(response)
        else:
            model_output = ""
            thought_summary = ""
        input_tokens = 0
        output_tokens = 0
        usage_metadata: dict[str, Any] = {}
        finish_reason = None
        graders: list[GraderResult]
        if is_error:
            graders = [
                GraderResult(
                    extractor_grader_type="gemini_batch_error",
                    extracted_answer=None,
                    is_correct=None,
                    metadata={"error": error},
                )
            ]
            written_error += 1
        else:
            input_tokens, output_tokens, usage_metadata = _extract_usage_counts(response)
            finish_reason = _candidate_finish_reason(response)
            graders = [
                _build_grader(
                    benchmark_name=str(metadata["benchmark_name"]),
                    model_output=model_output,
                    metadata=metadata,
                )
            ]
            written_success += 1

        cost = _calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        )
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += cost
        record = HintedInferenceRecord(
            inference_id=inference_id,
            problem_id=str(metadata["problem_id"]),
            benchmark_name=str(metadata["benchmark_name"]),
            model=str(metadata["canonical_model"]),
            hint_type=str(metadata["hint_type"]),
            fractioner=str(metadata["fractioner"]),
            hint_fraction=float(metadata["hint_fraction"]),
            hint_text_used=str(metadata["hint_text_used"]),
            model_output=model_output,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            cost=cost,
            is_error=is_error,
            graders=graders,
            hint=metadata["hint"],
            metadata={
                "run_id": run_id,
                "backend": "gemini_batch",
                "provider": "google_gemini_batch_api",
                "provider_model_id": manifest["model"],
                "prompt_version": "hinted_inference_v1",
                "prompt_id": metadata["prompt_id"],
                "expanded_prompt_path": metadata["expanded_prompt_path"],
                "prompt": metadata["prompt"],
                "max_tokens": metadata["max_output_tokens"],
                "generation_config": manifest["generation_config"],
                "cost_method": "calculated_from_gemini_usage_metadata",
                "token_pricing_per_million": pricing,
                "usage_metadata": usage_metadata,
                "fraction_metadata": metadata["fraction_metadata"],
                "stop_reason": finish_reason,
                "thought_summary": thought_summary,
                "thought_summary_chars": len(thought_summary),
                "gemini_batch_job_name": manifest.get("batch_job_name"),
                "gemini_batch_result_key": key,
                "gemini_batch_error": error,
            },
        )
        append_jsonl(output_path, record)

    summary = {
        "processed_at": _utcnow_iso(),
        "results_jsonl": str(result_path),
        "request_metadata_jsonl": str(metadata_path),
        "written_success": written_success,
        "written_error": written_error,
        "missing_metadata": missing_metadata,
        "duplicate_existing": duplicate_existing,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost,
        "pricing_per_million": pricing,
    }
    manifest["processed_results"] = summary
    _write_json(Path(args.manifest), manifest)
    return summary


def _estimate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(Path(args.manifest))
    _validate_supported_model(str(manifest["model"]))
    pricing = _safe_rate_lookup(
        model=str(manifest["model"]),
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    metadata_path = Path(manifest["request_metadata_jsonl"])
    request_count = 0
    input_tokens = 0
    max_output_tokens = 0
    assumed_output_tokens = args.assumed_output_tokens
    for row in _iter_jsonl(metadata_path):
        if not isinstance(row, dict):
            continue
        request_count += 1
        input_tokens += int(row.get("prompt_token_estimate", 0))
        if assumed_output_tokens is None:
            max_output_tokens += int(row.get("max_output_tokens", 0))
        else:
            max_output_tokens += int(assumed_output_tokens)
    estimate = {
        "request_count": request_count,
        "pricing_per_million": pricing,
        "input_token_estimate": input_tokens,
        "output_token_estimate": max_output_tokens,
        "estimated_cost_usd": _calculate_cost(
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
            pricing=pricing,
        ),
        "input_token_method": "ceil(prompt_chars / 4)",
        "output_token_method": (
            "per-request max_output_tokens"
            if assumed_output_tokens is None
            else f"assumed_output_tokens={assumed_output_tokens}"
        ),
    }
    return estimate


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def _add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", default="aime2025_2026")
    parser.add_argument("--hint-type", default="answer_not_revealed")
    parser.add_argument("--fractioner", default="mask_word")
    parser.add_argument("--hint-fractions", nargs="*", default=["0.0"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--skip-existing", type=_parse_bool, default=True)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--token-estimate-method",
        choices=["chars", "gemini-count-tokens"],
        default="chars",
        help=(
            "Use chars for offline estimates, or gemini-count-tokens for Gemini's "
            "count_tokens endpoint before writing the manifest."
        ),
    )
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Small Gemini Batch API harness for hinted inference experiments. "
            "Start with `build-jsonl`, then `submit`, `status`/`wait`, "
            "`download`, and `process-results`."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-jsonl")
    _add_common_build_args(build_parser)
    build_parser.set_defaults(func=_build_jsonl)

    estimate_parser = subparsers.add_parser("estimate")
    estimate_parser.add_argument("--manifest", required=True)
    estimate_parser.add_argument("--assumed-output-tokens", type=int, default=None)
    estimate_parser.add_argument("--input-price-per-million", type=float, default=None)
    estimate_parser.add_argument("--output-price-per-million", type=float, default=None)
    estimate_parser.set_defaults(func=_estimate)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--manifest", required=True)
    submit_parser.add_argument("--display-name", default=None)
    submit_parser.set_defaults(func=_submit)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--manifest", default=None)
    status_parser.add_argument("--job-name", default=None)
    status_parser.set_defaults(func=_status)

    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--manifest", default=None)
    wait_parser.add_argument("--job-name", default=None)
    wait_parser.add_argument("--poll-interval-seconds", type=int, default=30)
    wait_parser.set_defaults(func=_wait)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--manifest", default=None)
    download_parser.add_argument("--job-name", default=None)
    download_parser.add_argument("--output-path", default=None)
    download_parser.set_defaults(func=_download)

    process_parser = subparsers.add_parser("process-results")
    process_parser.add_argument("--manifest", required=True)
    process_parser.add_argument("--results-jsonl", default=None)
    process_parser.add_argument("--skip-existing", type=_parse_bool, default=True)
    process_parser.add_argument("--input-price-per-million", type=float, default=None)
    process_parser.add_argument("--output-price-per-million", type=float, default=None)
    process_parser.set_defaults(func=_process_results)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = args.func(args)
    _print_payload(payload)


if __name__ == "__main__":
    main()


"""
  Run this first:

  python -m runs.gemini_batch_hinted_test build-jsonl \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --hint-fractions 0.0 \
    --max-requests 3

  It prints a JSON payload. Copy the "input_jsonl" directory’s sibling manifest path, which will look like:

  data/gemini_batch_hinted/aime2025_2026/answer_not_revealed/mask_word/gemini-3.1-pro-preview/20260511_123456/manifest.json

  Then submit:

  python -m runs.gemini_batch_hinted_test submit \
    --manifest data/gemini_batch_hinted/aime2025_2026/answer_not_revealed/mask_word/gemini-3.1-pro-preview/20260511_193322/manifest.json

  That prints the canonical submitted manifest path, like:

  data/gemini_batch_hinted/manifests/20260511_123501__batches_123456789.json

  Use that path for the rest:

  python -m runs.gemini_batch_hinted_test status \
    --manifest data/gemini_batch_hinted/manifests/20260511_193849__batches_7sr3ba1jmg7qj6c24j5sr3rj9z4rm9zfomez.json

  python -m runs.gemini_batch_hinted_test wait \
    --manifest data/gemini_batch_hinted/manifests/20260511_193849__batches_7sr3ba1jmg7qj6c24j5sr3rj9z4rm9zfomez.json

  python -m runs.gemini_batch_hinted_test download \
    --manifest data/gemini_batch_hinted/manifests/20260511_123501__batches_123456789.json

  python -m runs.gemini_batch_hinted_test process-results \
    --manifest data/gemini_batch_hinted/manifests/20260511_123501__batches_123456789.json




hle uses "gemini-3.1-pro-preview-high"
"""
