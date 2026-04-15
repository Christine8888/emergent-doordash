from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.model_config import ALL_MODEL_PATHS, ModelSpec, get_model_spec
from src.storage import (
    _model_storage_component,
    append_jsonl,
    build_eci_score_path,
    make_stable_id,
    read_jsonl,
    write_jsonl,
)
from src.types import ECIScoreRecord, GraderResult, _utcnow_iso
from src.vllm_server import VLLMServer, VLLMServerConfig

BENCHMARK_CONFIGS: dict[str, "BenchmarkConfig"] = {}
BENCHMARKS = [
    "mmlu_5_shot__language_en_us__cot_false", # 14,042
    "bbh__prompt_type_answer_only", # 6,511
    "arc_challenge", # 1,172 questions
    "math__levels_5__fewshot_0", # 1,324 questions
    "hellaswag__split_validation", # 10,042 questions
    "piqa", # 1,838 questions
    "winogrande__dataset_name_winogrande_xl__fewshot_5", # 1,267 questions
    # without HumanEval: 36,196 scored samples per model for 1 epoch...


    # "humaneval__pass_at_1", 
]
MODELS_TO_RUN = list(ALL_MODEL_PATHS)
INSPECT_ENV_NAME = "ed_inspect"
INSPECT_ENV_PREFIX = Path("/sphinx/u/suzeva/miniconda3/envs") / INSPECT_ENV_NAME
INSPECT_BIN = INSPECT_ENV_PREFIX / "bin" / "inspect"
INSPECT_PYTHON = INSPECT_ENV_PREFIX / "bin" / "python"
XDG_DATA_HOME_ROOT = Path("/nlp/scr/suzeva/xdg_data")
XDG_CACHE_HOME_ROOT = Path("/nlp/scr/suzeva/xdg_cache")
EPOCHS = 1
MAX_TOKENS = 32768
MAX_RETRIES = 2
MAX_NUM_BATCHED_TOKENS = 32768
DEFAULT_SLURM_CPUS_PER_TASK = 16
DEFAULT_SLURM_MEM_GB = 64
EIGHT_GPU_SLURM_CPUS_PER_TASK = 120
EIGHT_GPU_SLURM_MEM_GB = 1000
SLURM_TIME_HOURS_OVERRIDE: int | None = None
NLP_SLURM_ACCOUNT = "nlp"
NLP_SLURM_PARTITION = "sphinx,jag-standard"
SPHINX_SLURM_ACCOUNT = "nlp"
SPHINX_SLURM_PARTITION = "sphinx"
MISO_SLURM_ACCOUNT = "miso"
MISO_SLURM_PARTITION = "miso"
HF_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    inspect_task: str
    task_args: dict[str, bool | int | str]
    source_metric_names: tuple[str, ...]
    sandbox: str | None = None


@dataclass(frozen=True)
class BenchmarkRunSummary:
    benchmark_id: str
    output_path: str
    inspect_log_path: str
    source_metric: str
    accuracy: float
    rollout_count: int
    elapsed_seconds: float


def _register(config: BenchmarkConfig) -> BenchmarkConfig:
    BENCHMARK_CONFIGS[config.benchmark_id] = config
    return config


_register(
    BenchmarkConfig(
        benchmark_id="mmlu_5_shot__language_en_us__cot_false",
        inspect_task="inspect_evals/mmlu_5_shot",
        task_args={"language": "EN_US", "cot": False},
        source_metric_names=("accuracy",),
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="bbh__prompt_type_answer_only",
        inspect_task="inspect_evals/bbh",
        task_args={"prompt_type": "answer_only"},
        source_metric_names=("accuracy",),
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="arc_challenge",
        inspect_task="inspect_evals/arc_challenge",
        task_args={},
        source_metric_names=("accuracy",),
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="math__levels_5__fewshot_0",
        inspect_task="inspect_evals/math",
        task_args={"levels": "5", "fewshot": 0},
        source_metric_names=("accuracy",),
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="humaneval__pass_at_1",
        inspect_task="inspect_evals/humaneval",
        task_args={},
        source_metric_names=("pass@1", "pass_at_1"),
        sandbox="docker",
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="hellaswag__split_validation",
        inspect_task="inspect_evals/hellaswag",
        task_args={"split": "validation"},
        source_metric_names=("accuracy",),
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="piqa",
        inspect_task="inspect_evals/piqa",
        task_args={},
        source_metric_names=("accuracy",),
    )
)
_register(
    BenchmarkConfig(
        benchmark_id="winogrande__dataset_name_winogrande_xl__fewshot_5",
        inspect_task="inspect_evals/winogrande",
        task_args={"dataset_name": "winogrande_xl", "fewshot": 5},
        source_metric_names=("accuracy",),
    )
)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _format_task_value(value: bool | int | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _selected_models(model: str) -> list[ModelSpec]:
    if not MODELS_TO_RUN:
        raise ValueError("MODELS_TO_RUN cannot be empty")
    if model == "all":
        return [get_model_spec(model_path) for model_path in MODELS_TO_RUN]
    if model not in MODELS_TO_RUN:
        raise ValueError(f"Model {model!r} is not in MODELS_TO_RUN")
    return [get_model_spec(model)]


def _resolve_parallelism(spec: ModelSpec, num_gpus: int | None) -> tuple[int, int, int]:
    tp = spec.tp
    if num_gpus is None:
        return tp, 1, tp
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if num_gpus < tp:
        raise ValueError(f"num_gpus={num_gpus} is smaller than model tp={tp} for model={spec.path}")
    if num_gpus % tp != 0:
        raise ValueError(
            f"num_gpus={num_gpus} must be divisible by model tp={tp} for model={spec.path}"
        )
    dp = num_gpus // tp
    return tp, dp, num_gpus


def _resolve_slurm_account(*, cluster: str) -> tuple[str, str, str]:
    if cluster == "miso":
        return "miso", MISO_SLURM_ACCOUNT, MISO_SLURM_PARTITION
    if cluster == "sphinx":
        return "sphinx", SPHINX_SLURM_ACCOUNT, SPHINX_SLURM_PARTITION
    if cluster == "nlp":
        return "nlp", NLP_SLURM_ACCOUNT, NLP_SLURM_PARTITION
    raise ValueError(f"Unsupported cluster: {cluster!r}")


def _resolve_slurm_time_hours(*, slurm_account: str) -> int:
    if SLURM_TIME_HOURS_OVERRIDE is not None:
        return SLURM_TIME_HOURS_OVERRIDE
    if slurm_account == "miso":
        return 6
    return 60


def _resolve_slurm_resources(*, requested_gpus: int) -> tuple[int, int]:
    if requested_gpus == 8:
        return EIGHT_GPU_SLURM_CPUS_PER_TASK, EIGHT_GPU_SLURM_MEM_GB
    return DEFAULT_SLURM_CPUS_PER_TASK, DEFAULT_SLURM_MEM_GB


def _build_log_dir(*, benchmark_id: str, model: str, root: str | Path = "data/inspect_logs") -> Path:
    model_name = _model_storage_component(model)
    return Path(root) / benchmark_id / model_name / time.strftime("%Y%m%d_%H%M%S")


def _build_inspect_env(*, log_dir: Path) -> dict[str, str]:
    xdg_data_home = XDG_DATA_HOME_ROOT
    xdg_cache_home = XDG_CACHE_HOME_ROOT
    xdg_data_home.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    trace_name = f"{log_dir.parent.name}__{log_dir.name}.trace.log"
    return {
        "XDG_DATA_HOME": str(xdg_data_home.resolve()),
        "XDG_CACHE_HOME": str(xdg_cache_home.resolve()),
        "INSPECT_TRACE_FILE": str((log_dir / trace_name).resolve()),
    }


def _relevant_hf_env() -> dict[str, str]:
    return {
        key: value
        for key in HF_ENV_KEYS
        if (value := os.environ.get(key))
    }


def _print_runtime_env_summary() -> None:
    summary = {
        "hostname": socket.gethostname(),
        "inspect_bin": str(INSPECT_BIN),
        "hf_env": _relevant_hf_env(),
    }
    print(f"[eci_inference] runtime_env {json.dumps(summary, sort_keys=True)}", flush=True)


def _preflight_huggingface_access() -> None:
    cache_paths = []
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key)
        if value:
            cache_paths.append({"env": key, "path": value, "exists": Path(value).exists()})

    try:
        socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
    except OSError as ex:
        print(
            "[eci_inference] huggingface_dns_check "
            f"status=failed error={ex!r} cache_paths={json.dumps(cache_paths)}",
            flush=True,
        )
        raise RuntimeError(
            "Unable to resolve huggingface.co from the worker node. "
            "The node network/DNS setup or propagated environment is incomplete."
        ) from ex

    print(
        "[eci_inference] huggingface_dns_check "
        f"status=ok cache_paths={json.dumps(cache_paths)}",
        flush=True,
    )


def _preflight_inspect_env() -> None:
    if not INSPECT_PYTHON.exists():
        raise RuntimeError(f"Inspect python not found at {INSPECT_PYTHON}")

    check = subprocess.run(
        [
            str(INSPECT_PYTHON),
            "-c",
            "import inspect_ai, inspect_evals, openai; print('inspect_env_ok')",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        details = (check.stderr or check.stdout).strip()
        raise RuntimeError(
            "Inspect environment preflight failed. "
            f"Expected {INSPECT_PYTHON} to import inspect_ai, inspect_evals, and openai. "
            f"Details: {details}"
        )


def _normalize_metric_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _coerce_metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        nested_value = value.get("value")
        if isinstance(nested_value, (int, float)) and not isinstance(nested_value, bool):
            return float(nested_value)
    return None


def _coerce_sample_score_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"c", "correct", "pass", "passed", "true"}:
            return 1.0
        if normalized in {"i", "incorrect", "fail", "failed", "false"}:
            return 0.0
        if normalized in {"p", "partial", "partially_correct"}:
            return 0.5
        if normalized in {"n", "none", "no_answer", "unanswered"}:
            return 0.0
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        return _coerce_sample_score_value(value.get("value"))
    return None


def _extract_metrics_map(payload: dict[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    metrics: dict[str, float] = {}
    if isinstance(results, dict):
        scores = results.get("scores")
        if isinstance(scores, list):
            for score in scores:
                if not isinstance(score, dict):
                    continue
                score_metrics = score.get("metrics")
                if isinstance(score_metrics, dict):
                    for metric_name, metric_payload in score_metrics.items():
                        metric_value = _coerce_metric_value(metric_payload)
                        if metric_value is not None and metric_name not in metrics:
                            metrics[metric_name] = metric_value

    if metrics:
        return metrics

    found: dict[str, float] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            value = _coerce_metric_value(node.get("value"))
            if isinstance(name, str) and value is not None and name not in found:
                found[name] = value
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(results if isinstance(results, dict) else payload)
    return found


def _extract_scalar_metric(
    *,
    payload: dict[str, Any],
    source_metric_names: tuple[str, ...],
) -> tuple[str, float, dict[str, float]]:
    metrics = _extract_metrics_map(payload)
    normalized_to_original = {_normalize_metric_name(name): name for name in metrics}
    for source_metric_name in source_metric_names:
        original_name = normalized_to_original.get(_normalize_metric_name(source_metric_name))
        if original_name is not None:
            return original_name, metrics[original_name], metrics
    raise ValueError(
        "Unable to find any of the requested metric names "
        f"{source_metric_names!r} in inspect log metrics {sorted(metrics.keys())!r}"
    )


def _find_single_log_file(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No inspect log files found under {log_dir}")
    return candidates[-1]


def _tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _extract_score_metrics(node: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            metric_value = _coerce_sample_score_value(value)
            if metric_value is not None and key not in metrics:
                metrics[key] = metric_value
    elif isinstance(node, list):
        for item in node:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            metric_value = _coerce_sample_score_value(item.get("value"))
            if isinstance(name, str) and metric_value is not None and name not in metrics:
                metrics[name] = metric_value
    return metrics


def _eci_inference_id(*, benchmark_name: str, model: str, problem_id: str, rollout_id: int) -> str:
    canonical_model = _model_storage_component(model)
    return make_stable_id(benchmark_name, canonical_model, problem_id, rollout_id, length=24)


def _read_existing_rollout_rows(path: Path) -> list[ECIScoreRecord]:
    rows = read_jsonl(path, model_cls=ECIScoreRecord)
    return [row for row in rows if isinstance(row, ECIScoreRecord)]


def _resolve_existing_rollout_count(path: Path) -> tuple[int, list[ECIScoreRecord]]:
    rows = _read_existing_rollout_rows(path)
    if not rows:
        return 0, []
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.problem_id] = counts.get(row.problem_id, 0) + 1
    distinct_counts = sorted(set(counts.values()))
    if len(distinct_counts) != 1:
        raise ValueError(
            f"Inconsistent existing rollout counts in {path}: "
            f"{distinct_counts}. Refusing to guess how many new epochs to run."
        )
    return distinct_counts[0], rows


def _row_has_numeric_score(row: ECIScoreRecord) -> bool:
    if not row.graders:
        return False
    score = row.graders[0].metadata.get("score")
    return isinstance(score, (int, float))


def _repair_existing_rows_if_needed(
    *,
    output_path: Path,
    config: BenchmarkConfig,
    model_path: str,
    rows: list[ECIScoreRecord],
) -> list[ECIScoreRecord]:
    if not rows or any(_row_has_numeric_score(row) for row in rows):
        return rows

    rows_by_log_path: dict[Path, list[ECIScoreRecord]] = {}
    for row in rows:
        inspect_log_path = row.metadata.get("inspect_log_path")
        if not isinstance(inspect_log_path, str) or not inspect_log_path:
            return rows
        rows_by_log_path.setdefault(Path(inspect_log_path), []).append(row)

    repaired_rows: list[ECIScoreRecord] = []
    for inspect_log_path, log_rows in sorted(rows_by_log_path.items(), key=lambda item: str(item[0])):
        if not inspect_log_path.exists():
            return rows
        payload = json.loads(inspect_log_path.read_text(encoding="utf-8"))
        min_rollout_id = min(row.rollout_id for row in log_rows)
        epoch_values = [
            row.metadata.get("epoch_in_run")
            for row in log_rows
            if isinstance(row.metadata.get("epoch_in_run"), int)
        ]
        min_epoch = min(epoch_values) if epoch_values else 1
        rollout_offset = min_rollout_id - min_epoch
        run_metadata = log_rows[0].metadata.get("run_metadata")
        repaired_rows.extend(
            _extract_sample_rows(
                payload=payload,
                benchmark_id=config.benchmark_id,
                model_path=model_path,
                source_metric_names=config.source_metric_names,
                rollout_offset=rollout_offset,
                inspect_log_path=inspect_log_path,
                run_metadata=run_metadata if isinstance(run_metadata, dict) else {},
            )
        )

    repaired_rows.sort(key=lambda row: (row.rollout_id, row.problem_id))
    write_jsonl(output_path, repaired_rows)
    print(
        f"[eci_inference] repaired existing output benchmark={config.benchmark_id} "
        f"model={model_path} path={output_path}",
        flush=True,
    )
    return repaired_rows


def _stringify_sample_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _extract_model_output(sample: dict[str, Any]) -> str:
    for key in ("output", "completion", "answer", "response"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nested_key in ("completion", "content", "text"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested

    scores = sample.get("scores")
    if isinstance(scores, dict):
        for score_payload in scores.values():
            if isinstance(score_payload, dict):
                explanation = score_payload.get("explanation")
                if isinstance(explanation, str):
                    return explanation
    if isinstance(scores, list):
        for score_payload in scores:
            if isinstance(score_payload, dict):
                explanation = score_payload.get("explanation")
                if isinstance(explanation, str):
                    return explanation
    return ""


def _extract_rendered_prompt(sample: dict[str, Any]) -> str | None:
    prompt_candidates = [
        sample.get("prompt"),
        sample.get("state", {}).get("prompt") if isinstance(sample.get("state"), dict) else None,
    ]
    for candidate in prompt_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _extract_prompt_messages(sample: dict[str, Any]) -> list[dict[str, Any]] | None:
    candidate_paths = [
        sample.get("messages"),
        sample.get("state", {}).get("messages") if isinstance(sample.get("state"), dict) else None,
        sample.get("input") if isinstance(sample.get("input"), list) else None,
    ]
    for candidate in candidate_paths:
        if not isinstance(candidate, list):
            continue
        messages: list[dict[str, Any]] = []
        for item in candidate:
            if isinstance(item, dict):
                messages.append(item)
        if messages:
            return messages
    return None


def _extract_token_count(sample: dict[str, Any], *keys: str) -> int:
    for container_key in ("model_usage", "usage"):
        container = sample.get(container_key)
        if isinstance(container, dict):
            for key in keys:
                value = container.get(key)
                if isinstance(value, int):
                    return value
    for key in keys:
        value = sample.get(key)
        if isinstance(value, int):
            return value
    return 0


def _find_logged_generation_config(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    target_keys = {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "max_retries",
        "max_connections",
    }

    def _walk(node: Any, path: str) -> tuple[str | None, dict[str, Any]]:
        if isinstance(node, dict):
            subset = {key: node[key] for key in target_keys if key in node}
            if subset:
                return path, subset
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                found_path, found_config = _walk(value, child_path)
                if found_config:
                    return found_path, found_config
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                child_path = f"{path}[{idx}]"
                found_path, found_config = _walk(value, child_path)
                if found_config:
                    return found_path, found_config
        return None, {}

    return _walk(payload, "")


def _coerce_logged_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _expected_generation_config(
    *,
    sampling_params: dict[str, bool | float | int],
    max_connections: int,
) -> dict[str, float]:
    expected: dict[str, float] = {
        "max_tokens": float(MAX_TOKENS),
        "max_retries": float(MAX_RETRIES),
        "max_connections": float(max_connections),
    }
    do_sample = bool(sampling_params.get("do_sample", True))
    if do_sample:
        if "temperature" in sampling_params:
            expected["temperature"] = float(sampling_params["temperature"])
        if "top_p" in sampling_params:
            expected["top_p"] = float(sampling_params["top_p"])
        if "top_k" in sampling_params:
            expected["top_k"] = float(sampling_params["top_k"])
    else:
        expected["temperature"] = 0.0
    return expected


def _validate_logged_generation_config(
    *,
    benchmark_id: str,
    model_path: str,
    config_path: str | None,
    logged_generation_config: dict[str, Any],
    sampling_params: dict[str, bool | float | int],
    max_connections: int,
) -> None:
    expected = _expected_generation_config(
        sampling_params=sampling_params,
        max_connections=max_connections,
    )
    missing = [key for key in expected if key not in logged_generation_config]
    if missing:
        raise RuntimeError(
            "Inspect log did not record expected generation settings "
            f"for benchmark={benchmark_id} model={model_path}: missing={missing} "
            f"path={config_path or '<missing>'} "
            f"logged={json.dumps(logged_generation_config, sort_keys=True)}"
        )

    mismatches: list[str] = []
    for key, expected_value in expected.items():
        logged_value = _coerce_logged_number(logged_generation_config.get(key))
        if logged_value is None:
            mismatches.append(f"{key}=<non-numeric:{logged_generation_config.get(key)!r}> expected={expected_value}")
            continue
        if abs(logged_value - expected_value) > 1e-9:
            mismatches.append(f"{key}={logged_value} expected={expected_value}")

    if mismatches:
        raise RuntimeError(
            "Inspect logged generation settings did not match expected values "
            f"for benchmark={benchmark_id} model={model_path}: "
            + "; ".join(mismatches)
            + f" path={config_path or '<missing>'}"
        )


def _extract_sample_rows(
    *,
    payload: dict[str, Any],
    benchmark_id: str,
    model_path: str,
    source_metric_names: tuple[str, ...],
    rollout_offset: int,
    inspect_log_path: Path,
    run_metadata: dict[str, Any],
) -> list[ECIScoreRecord]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"Inspect log {inspect_log_path} did not contain a top-level `samples` list.")

    rows: list[ECIScoreRecord] = []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            continue
        problem_id = sample.get("id")
        epoch = sample.get("epoch")
        if not isinstance(problem_id, (str, int)):
            raise ValueError(f"Inspect sample missing `id` in {inspect_log_path}")
        if not isinstance(epoch, int):
            raise ValueError(f"Inspect sample missing integer `epoch` in {inspect_log_path}")

        metrics = _extract_score_metrics(sample.get("scores"))
        score_metric_name = None
        score_value = None
        normalized_to_original = {_normalize_metric_name(name): name for name in metrics}
        for source_metric_name in source_metric_names:
            original_name = normalized_to_original.get(_normalize_metric_name(source_metric_name))
            if original_name is not None:
                score_metric_name = original_name
                score_value = metrics[original_name]
                break
        if score_metric_name is None and metrics:
            score_metric_name = next(iter(metrics))
            score_value = metrics[score_metric_name]

        is_correct = None if score_value is None else bool(float(score_value) >= 0.5)
        rollout_id = rollout_offset + epoch
        rendered_prompt = _extract_rendered_prompt(sample)
        prompt_messages = _extract_prompt_messages(sample)
        rows.append(
            ECIScoreRecord(
                inference_id=_eci_inference_id(
                    benchmark_name=benchmark_id,
                    model=model_path,
                    problem_id=str(problem_id),
                    rollout_id=rollout_id,
                ),
                problem_id=str(problem_id),
                benchmark_name=benchmark_id,
                model=_model_storage_component(model_path),
                rollout_id=rollout_id,
                question=_stringify_sample_text(sample.get("input")),
                answer=_stringify_sample_text(sample.get("target")),
                model_output=_extract_model_output(sample),
                input_token_count=_extract_token_count(sample, "input_tokens", "prompt_tokens"),
                output_token_count=_extract_token_count(sample, "output_tokens", "completion_tokens"),
                cost=0.0,
                is_error=sample.get("error") is not None,
                graders=[
                    GraderResult(
                        extractor_grader_type="inspect_score",
                        extracted_answer=None,
                        is_correct=is_correct,
                        metadata={
                            "score_metric": score_metric_name,
                            "score": score_value,
                            "scores": metrics,
                            "sample_metadata": sample.get("metadata"),
                            "sample_error": sample.get("error"),
                            "sample_retries": sample.get("retries"),
                            "epoch_in_run": epoch,
                            "inspect_log_path": str(inspect_log_path),
                        },
                    )
                ],
                metadata={
                    "inspect_scores": metrics,
                    "score_metric": score_metric_name,
                    "score": score_value,
                    "rendered_prompt": rendered_prompt,
                    "prompt_messages": prompt_messages,
                    "sample_metadata": sample.get("metadata"),
                    "sample_error": sample.get("error"),
                    "sample_retries": sample.get("retries"),
                    "epoch_in_run": epoch,
                    "inspect_log_path": str(inspect_log_path),
                    "run_metadata": run_metadata,
                },
            )
        )
    return rows


def _summarize_rollout_rows(
    *,
    rows: list[ECIScoreRecord],
    source_metric_names: tuple[str, ...],
) -> tuple[str, float, int]:
    normalized_target_names = {_normalize_metric_name(name) for name in source_metric_names}
    values: list[float] = []
    fallback_values: list[float] = []
    source_metric = source_metric_names[0]
    fallback_metric = source_metric
    for row in rows:
        if not row.graders:
            continue
        grader = row.graders[0]
        score_metric = grader.metadata.get("score_metric")
        score = grader.metadata.get("score")
        if not isinstance(score_metric, str) or not isinstance(score, (int, float)):
            continue
        fallback_metric = score_metric
        fallback_values.append(float(score))
        if _normalize_metric_name(score_metric) not in normalized_target_names:
            continue
        source_metric = score_metric
        values.append(float(score))
    if not values and fallback_values:
        return fallback_metric, sum(fallback_values) / len(fallback_values), len(fallback_values)
    if not values:
        raise ValueError(f"No per-sample scores found for metrics {source_metric_names!r}")
    return source_metric, sum(values) / len(values), len(values)


def _resolve_inspect_command() -> list[str]:
    if not INSPECT_BIN.exists():
        raise RuntimeError(
            f"Inspect binary not found at {INSPECT_BIN}. "
            f"Create/fix the `{INSPECT_ENV_NAME}` env so it provides that executable."
        )
    return [str(INSPECT_BIN)]


def _build_inspect_command(
    *,
    config: BenchmarkConfig,
    inspect_model: str,
    model_base_url: str | None,
    log_dir: Path,
    limit: int | None,
    max_connections: int,
    sampling_params: dict[str, bool | float | int],
) -> list[str]:
    cmd = list(_resolve_inspect_command())
    cmd.extend(
        [
            "eval",
            config.inspect_task,
            "--model",
            inspect_model,
            "--log-format",
            "json",
            "--log-dir",
            str(log_dir),
            "--display",
            "none",
            "--log-level",
            "info",
            "--debug-errors",
            "--max-connections",
            str(max_connections),
        ]
    )
    if config.sandbox is not None:
        cmd.extend(["--sandbox", config.sandbox])
    if model_base_url is not None:
        cmd.extend(["--model-base-url", model_base_url])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    cmd.extend(["--max-tokens", str(MAX_TOKENS)])
    cmd.extend(["--max-retries", str(MAX_RETRIES)])

    do_sample = bool(sampling_params.get("do_sample", True))
    if do_sample:
        if "temperature" in sampling_params:
            cmd.extend(["--temperature", str(sampling_params["temperature"])])
        if "top_p" in sampling_params:
            cmd.extend(["--top-p", str(sampling_params["top_p"])])
        if "top_k" in sampling_params:
            cmd.extend(["--top-k", str(sampling_params["top_k"])])
    else:
        cmd.extend(["--temperature", "0"])

    for name, value in config.task_args.items():
        cmd.extend(["-T", f"{name}={_format_task_value(value)}"])
    return cmd


def _run_inspect_eval(
    *,
    config: BenchmarkConfig,
    model_path: str,
    backend: str,
    limit: int | None,
    max_connections: int,
    sampling_params: dict[str, bool | float | int],
    run_metadata: dict[str, Any],
    model_base_url: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> BenchmarkRunSummary:
    if backend == "local-vllm":
        inspect_model = f"vllm/{model_path}"
    elif backend == "together-serverless":
        inspect_model = f"together/{model_path}"
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    output_path = build_eci_score_path(
        benchmark_name=config.benchmark_id,
        model=model_path,
        data_root="data",
    )
    existing_rollout_count, existing_rows = _resolve_existing_rollout_count(output_path)
    existing_rows = _repair_existing_rows_if_needed(
        output_path=output_path,
        config=config,
        model_path=model_path,
        rows=existing_rows,
    )
    if existing_rollout_count > EPOCHS:
        print(
            f"[eci_inference] skip benchmark={config.benchmark_id} model={model_path} "
            f"existing_rollouts={existing_rollout_count} target_epochs={EPOCHS}",
            flush=True,
        )
        source_metric, accuracy, rollout_count = _summarize_rollout_rows(
            rows=existing_rows,
            source_metric_names=config.source_metric_names,
        )
        return BenchmarkRunSummary(
            benchmark_id=config.benchmark_id,
            output_path=str(output_path),
            inspect_log_path="",
            source_metric=source_metric,
            accuracy=accuracy,
            rollout_count=rollout_count,
            elapsed_seconds=0.0,
        )

    additional_epochs = EPOCHS - existing_rollout_count
    if additional_epochs == 0:
        source_metric, accuracy, rollout_count = _summarize_rollout_rows(
            rows=existing_rows,
            source_metric_names=config.source_metric_names,
        )
        return BenchmarkRunSummary(
            benchmark_id=config.benchmark_id,
            output_path=str(output_path),
            inspect_log_path="",
            source_metric=source_metric,
            accuracy=accuracy,
            rollout_count=rollout_count,
            elapsed_seconds=0.0,
        )

    log_dir = _build_log_dir(benchmark_id=config.benchmark_id, model=model_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_inspect_command(
        config=config,
        inspect_model=inspect_model,
        model_base_url=model_base_url,
        log_dir=log_dir,
        limit=limit,
        max_connections=max_connections,
        sampling_params=sampling_params,
    )
    cmd.extend(["--epochs", str(additional_epochs)])

    env = os.environ.copy()
    env.update(_build_inspect_env(log_dir=log_dir))
    if extra_env is not None:
        env.update(extra_env)

    print(
        f"[eci_inference] running benchmark={config.benchmark_id} model={model_path} "
        f"task={config.inspect_task}",
        flush=True,
    )
    started_at = time.monotonic()
    completed = subprocess.run(
        cmd,
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed_seconds = time.monotonic() - started_at
    inspect_stdout_path = log_dir / "inspect.stdout.log"
    inspect_stderr_path = log_dir / "inspect.stderr.log"
    inspect_stdout_path.write_text(completed.stdout, encoding="utf-8")
    inspect_stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        stdout_tail = _tail_text(completed.stdout.strip())
        stderr_tail = _tail_text(completed.stderr.strip())
        if stdout_tail:
            print(
                f"[eci_inference] inspect_stdout_tail benchmark={config.benchmark_id} "
                f"model={model_path}\n{stdout_tail}",
                flush=True,
            )
        if stderr_tail:
            print(
                f"[eci_inference] inspect_stderr_tail benchmark={config.benchmark_id} "
                f"model={model_path}\n{stderr_tail}",
                flush=True,
            )
        raise RuntimeError(
            "Inspect eval failed "
            f"(exit_code={completed.returncode}, "
            f"stdout_log={inspect_stdout_path}, stderr_log={inspect_stderr_path})"
        )

    inspect_log_path = _find_single_log_file(log_dir)
    with open(inspect_log_path, "r", encoding="utf-8") as f:
        inspect_payload = json.load(f)
    config_path, logged_generation_config = _find_logged_generation_config(inspect_payload)
    if logged_generation_config:
        print(
            "[eci_inference] inspect_logged_generation_config "
            f"benchmark={config.benchmark_id} model={model_path} "
            f"path={config_path or '<root>'} "
            f"config={json.dumps(logged_generation_config, sort_keys=True)}",
            flush=True,
        )
    else:
        print(
            "[eci_inference] inspect_logged_generation_config "
            f"benchmark={config.benchmark_id} model={model_path} path=<missing> config={{}}",
            flush=True,
        )
    _validate_logged_generation_config(
        benchmark_id=config.benchmark_id,
        model_path=model_path,
        config_path=config_path,
        logged_generation_config=logged_generation_config,
        sampling_params=sampling_params,
        max_connections=max_connections,
    )
    new_rows = _extract_sample_rows(
        payload=inspect_payload,
        benchmark_id=config.benchmark_id,
        model_path=model_path,
        source_metric_names=config.source_metric_names,
        rollout_offset=existing_rollout_count,
        inspect_log_path=inspect_log_path,
        run_metadata=run_metadata,
    )
    for row in new_rows:
        append_jsonl(output_path, row)
    all_rows = existing_rows + new_rows
    source_metric, accuracy, rollout_count = _summarize_rollout_rows(
        rows=all_rows,
        source_metric_names=config.source_metric_names,
    )

    return BenchmarkRunSummary(
        benchmark_id=config.benchmark_id,
        output_path=str(output_path),
        inspect_log_path=str(inspect_log_path),
        source_metric=source_metric,
        accuracy=accuracy,
        rollout_count=rollout_count,
        elapsed_seconds=elapsed_seconds,
    )


def _run_single_model_job(
    *,
    benchmark_names: list[str],
    model_path: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    sampling_params: dict[str, bool | float | int],
    run_metadata: dict[str, Any],
    limit: int | None,
    max_connections: int,
    gpu_memory_utilization: float,
    dtype: str,
    backend: str,
) -> dict[str, Any]:
    summaries: list[BenchmarkRunSummary] = []
    _print_runtime_env_summary()
    _preflight_inspect_env()
    _preflight_huggingface_access()

    def _run_all(model_base_url: str | None, extra_env: dict[str, str] | None) -> None:
        for benchmark_name in benchmark_names:
            config = BENCHMARK_CONFIGS[benchmark_name]
            summary = _run_inspect_eval(
                config=config,
                model_path=model_path,
                backend=backend,
                limit=limit,
                max_connections=max_connections,
                sampling_params=sampling_params,
                run_metadata=run_metadata,
                model_base_url=model_base_url,
                extra_env=extra_env,
            )
            summaries.append(summary)

    if backend == "local-vllm":
        server_config = VLLMServerConfig(
            model_path=model_path,
            served_model_name=model_path,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            dtype=dtype,
        )
        with VLLMServer(server_config) as server:
            base_url = f"http://localhost:{server.port}/v1"
            _run_all(
                model_base_url=base_url,
                extra_env={
                    "VLLM_BASE_URL": base_url,
                    "VLLM_API_KEY": "local",
                },
            )
    elif backend == "together-serverless":
        _run_all(model_base_url=None, extra_env=None)
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    return {
        "model": model_path,
        "model_path": model_path,
        "run_metadata": run_metadata,
        "summaries": [asdict(summary) for summary in summaries],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Inspect-backed ECI benchmark scoring.")
    parser.add_argument("--model", type=str, choices=["all"] + MODELS_TO_RUN, default="all")
    parser.add_argument("--backend", choices=["local-vllm", "together-serverless"], default="local-vllm")
    parser.add_argument(
        "--cluster",
        choices=["nlp", "sphinx", "miso"],
        default="nlp",
        help="Submit target cluster/account routing (no auto-inference).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--executor", choices=["local", "submitit"], default="local")
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.91)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Requested GPUs per model job. "
            "Must be divisible by the model's tp from src/model_config.py."
        ),
    )
    parser.add_argument("--dry-run", type=_parse_bool, default=False)
    return parser


def _load_sampling_params(models: list[ModelSpec]) -> dict[str, dict[str, Any]]:
    return {spec.path: dict(spec.sampling_params) for spec in models}


def _build_run_metadata(
    *,
    args: argparse.Namespace,
    benchmark_names: list[str],
    spec: ModelSpec,
    tp: int,
    dp: int,
    requested_gpus: int,
    sampling_params: dict[str, bool | float | int],
    resolved_cluster: str | None = None,
    slurm_account: str | None = None,
    slurm_partition: str | None = None,
    slurm_time_hours: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem_gb: int | None = None,
) -> dict[str, Any]:
    benchmark_details = {
        name: {
            "inspect_task": BENCHMARK_CONFIGS[name].inspect_task,
            "task_args": BENCHMARK_CONFIGS[name].task_args,
            "source_metric_names": list(BENCHMARK_CONFIGS[name].source_metric_names),
            "sandbox": BENCHMARK_CONFIGS[name].sandbox,
        }
        for name in benchmark_names
    }
    return {
        "launcher": "src.eci_inference",
        "cli_args": dict(vars(args)),
        "job": {
            "executor": args.executor,
            "backend": args.backend,
            "benchmarks": benchmark_names,
            "benchmark_details": benchmark_details,
            "limit": args.limit,
        },
        "model_spec": {
            "path": spec.path,
            "name": spec.name,
            "tp": spec.tp,
            "constraint": spec.constraint,
            "sampling_params": sampling_params,
        },
        "parallelism": {
            "num_gpus_arg": args.num_gpus,
            "requested_gpus": requested_gpus,
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
        },
        "inspect": {
            "command": _resolve_inspect_command(),
            "max_connections": args.max_connections,
            "max_tokens": MAX_TOKENS,
            "max_retries": MAX_RETRIES,
            "epochs": EPOCHS,
        },
        "vllm_server": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "dtype": args.dtype,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "generation": {
            "do_sample": sampling_params.get("do_sample"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "top_k": sampling_params.get("top_k"),
            "repetition_penalty": sampling_params.get("repetition_penalty"),
            "max_tokens": MAX_TOKENS,
        },
        "slurm": {
            "cluster_arg": args.cluster,
            "resolved_cluster": resolved_cluster,
            "account": slurm_account,
            "partition": slurm_partition,
            "time_hours": slurm_time_hours,
            "cpus_per_task": slurm_cpus_per_task,
            "mem_gb": slurm_mem_gb,
        },
    }


def _print_plan(args: argparse.Namespace, benchmark_names: list[str], models: list[ModelSpec]) -> None:
    print("[eci_inference] plan", flush=True)
    print(
        json.dumps(
            {
                "executor": args.executor,
                "backend": args.backend,
                "benchmarks": benchmark_names,
                "models": [m.path for m in models],
                "limit": args.limit,
                "epochs": EPOCHS,
                "max_retries": MAX_RETRIES,
            },
            indent=2,
        ),
        flush=True,
    )
    for spec in models:
        for benchmark_name in benchmark_names:
            path = build_eci_score_path(
                benchmark_name=benchmark_name,
                model=spec.path,
                data_root="data",
            )
            print(f"  output -> {path}", flush=True)


def _run_local(
    args: argparse.Namespace,
    benchmark_names: list[str],
    models: list[ModelSpec],
    sampling_params_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in models:
        tp, dp, requested_gpus = _resolve_parallelism(spec, args.num_gpus)
        sampling_params = sampling_params_by_model.get(spec.path, {})
        run_metadata = _build_run_metadata(
            args=args,
            benchmark_names=benchmark_names,
            spec=spec,
            tp=tp,
            dp=dp,
            requested_gpus=requested_gpus,
            sampling_params=sampling_params,
        )
        print(f"[eci_inference] local model={spec.path}", flush=True)
        result = _run_single_model_job(
            benchmark_names=benchmark_names,
            model_path=spec.path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            sampling_params=sampling_params,
            run_metadata=run_metadata,
            limit=args.limit,
            max_connections=args.max_connections,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            backend=args.backend,
        )
        results.append(result)
    return results


def _run_submitit(
    args: argparse.Namespace,
    benchmark_names: list[str],
    models: list[ModelSpec],
    sampling_params_by_model: dict[str, dict[str, Any]],
) -> list[Any]:
    import submitit

    submitit_dir = Path("data/submitit_logs/eci_scores")
    submitit_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(submitit_dir))

    jobs = []
    for spec in models:
        tp, dp, requested_gpus = _resolve_parallelism(spec, args.num_gpus)
        model_name = spec.path.split("/")[-1]
        resolved_cluster, account, partition = _resolve_slurm_account(cluster=args.cluster)
        time_hours = _resolve_slurm_time_hours(slurm_account=account)
        cpus_per_task, mem_gb = _resolve_slurm_resources(requested_gpus=requested_gpus)
        sampling_params = sampling_params_by_model.get(spec.path, {})
        run_metadata = _build_run_metadata(
            args=args,
            benchmark_names=benchmark_names,
            spec=spec,
            tp=tp,
            dp=dp,
            requested_gpus=requested_gpus,
            sampling_params=sampling_params,
            resolved_cluster=resolved_cluster,
            slurm_account=account,
            slurm_partition=partition,
            slurm_time_hours=time_hours,
            slurm_cpus_per_task=cpus_per_task,
            slurm_mem_gb=mem_gb,
        )
        params = {
            "name": f"eci_{model_name}",
            "slurm_account": account,
            "slurm_partition": partition,
            "slurm_gpus_per_node": requested_gpus,
            "slurm_cpus_per_task": cpus_per_task,
            "slurm_mem": f"{mem_gb}GB",
            "slurm_time": time_hours * 60,
            "timeout_min": time_hours * 60,
        }
        executor.update_parameters(**params)
        job = executor.submit(
            _run_single_model_job,
            benchmark_names=benchmark_names,
            model_path=spec.path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            sampling_params=sampling_params,
            run_metadata=run_metadata,
            limit=args.limit,
            max_connections=args.max_connections,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            backend=args.backend,
        )
        jobs.append(job)
        print(f"[eci_inference] submitted job_id={job.job_id} model={spec.path}", flush=True)
    return jobs


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    benchmark_names = list(BENCHMARKS)
    models = _selected_models(args.model)
    sampling_params_by_model = _load_sampling_params(models)

    _print_plan(args, benchmark_names, models)
    if args.dry_run:
        return

    if args.executor == "local":
        results = _run_local(args, benchmark_names, models, sampling_params_by_model)
        print(json.dumps(results, indent=2), flush=True)
        return

    jobs = _run_submitit(args, benchmark_names, models, sampling_params_by_model)
    print(
        json.dumps(
            {
                "submitted_jobs": [job.job_id for job in jobs],
                "models": [spec.path for spec in models],
                "benchmarks": benchmark_names,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()



"""
MISO
python -m src.eci_inference \
    --backend local-vllm \
    --model Qwen/Qwen3-0.6B \
    --limit 5 \
    --dry-run  true\
    --executor submitit \
    --cluster miso \
    --num-gpus 8 \
    --max-connections 300

NLP
python -m src.eci_inference \
    --backend local-vllm \
    --model Qwen/Qwen3-0.6B \
    --limit 20 \
    --executor submitit \
    --cluster nlp \
    --num-gpus 1 \
    --max-connections 48
"""
