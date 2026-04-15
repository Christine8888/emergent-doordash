from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.storage import _model_storage_component, _safe_component, append_jsonl, build_eci_score_path, make_stable_id
from src.types import ECIScoreRecord, GraderResult
from src.vllm_server import VLLMServer, VLLMServerConfig

BENCHMARK_CONFIGS: dict[str, "BenchmarkConfig"] = {}
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
DEFAULT_CHECKPOINT_EVERY = 2000
TASK_SAMPLE_ID_CACHE: dict[str, list[str]] = {}
SAMPLE_IDS_PREFIX = "__ECI_SAMPLE_IDS__="


def _timestamp() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


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
    existing_count: int
    new_count: int
    elapsed_seconds: float


def _register(config: BenchmarkConfig) -> BenchmarkConfig:
    BENCHMARK_CONFIGS[config.benchmark_id] = config
    return config


_register(
    BenchmarkConfig(
        benchmark_id="mmlu_5_shot__language_en_us__cot_true",
        inspect_task="inspect_evals/mmlu_5_shot",
        task_args={"language": "EN_US", "cot": True},
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


def _format_task_value(value: bool | int | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _normalize_metric_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


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


def _read_existing_rows(path: Path) -> list[ECIScoreRecord]:
    if not path.exists():
        return []

    rows: list[ECIScoreRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                rows.append(ECIScoreRecord.model_validate(payload))
            except Exception:
                continue
    return rows


def _read_existing_inference_ids(path: Path) -> set[str]:
    return {row.inference_id for row in _read_existing_rows(path)}


def _dedupe_rows(rows: list[ECIScoreRecord]) -> list[ECIScoreRecord]:
    deduped: dict[str, ECIScoreRecord] = {}
    for row in rows:
        deduped.setdefault(row.inference_id, row)
    return sorted(deduped.values(), key=lambda row: (row.problem_id, row.rollout_id, row.created_at))


def _read_eci_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _infer_total_samples_from_checkpoints(
    *,
    benchmark_name: str,
    data_root: str | Path,
) -> int:
    benchmark_dir = Path(data_root) / "eci_scores" / _safe_component(benchmark_name)
    if not benchmark_dir.exists():
        return 0
    totals: list[int] = []
    for ckpt_path in benchmark_dir.glob("*.ckpt.json"):
        payload = _read_eci_checkpoint(ckpt_path)
        if payload is None:
            continue
        total_value = payload.get("total_samples")
        if isinstance(total_value, int) and total_value > 0:
            totals.append(total_value)
    return max(totals) if totals else 0


def is_eci_benchmark_complete(
    *,
    benchmark_name: str,
    model_path: str,
    limit: int | None,
    data_root: str | Path = "data",
) -> bool:
    config = BENCHMARK_CONFIGS[benchmark_name]
    output_path = build_eci_score_path(
        benchmark_name=benchmark_name,
        model=model_path,
        data_root=data_root,
    )
    existing_rows = _dedupe_rows(_read_existing_rows(output_path))
    ckpt_path = _checkpoint_path_for_output(output_path)
    ckpt_payload = _read_eci_checkpoint(ckpt_path)
    existing_ids = {
        row.inference_id
        for row in existing_rows
        if 1 <= row.rollout_id <= EPOCHS
    }
    if limit is None:
        total_samples = 0
        if ckpt_payload is not None:
            total_value = ckpt_payload.get("total_samples")
            if isinstance(total_value, int) and total_value > 0:
                total_samples = total_value
        if total_samples <= 0:
            total_samples = _infer_total_samples_from_checkpoints(
                benchmark_name=benchmark_name,
                data_root=data_root,
            )
        if total_samples > 0 and len(existing_ids) >= total_samples:
            return True

    try:
        dataset_problem_ids = _load_task_sample_ids(config)
    except Exception:
        return False

    target_problem_ids = dataset_problem_ids[:limit] if limit is not None else dataset_problem_ids
    if not target_problem_ids:
        return True

    target_problem_id_set = set(target_problem_ids)
    existing_ids = {
        row.inference_id
        for row in existing_rows
        if row.problem_id in target_problem_id_set and 1 <= row.rollout_id <= EPOCHS
    }
    for rollout_id in range(1, EPOCHS + 1):
        for problem_id in target_problem_ids:
            if _eci_inference_id(
                benchmark_name=benchmark_name,
                model=model_path,
                problem_id=problem_id,
                rollout_id=rollout_id,
            ) not in existing_ids:
                return False
    return True


def is_eci_model_complete(
    *,
    benchmark_names: list[str],
    model_path: str,
    limit: int | None,
    data_root: str | Path = "data",
) -> bool:
    return all(
        is_eci_benchmark_complete(
            benchmark_name=benchmark_name,
            model_path=model_path,
            limit=limit,
            data_root=data_root,
        )
        for benchmark_name in benchmark_names
    )


def _chunk_sample_ids(sample_ids: list[str], *, max_chars: int = 24000) -> list[list[str]]:
    if not sample_ids:
        return []
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for sample_id in sample_ids:
        sample_chars = len(sample_id) + (1 if current else 0)
        if current and current_chars + sample_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(sample_id)
        current_chars += len(sample_id) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def _chunk_sample_ids_with_limits(
    sample_ids: list[str],
    *,
    max_items: int,
    max_chars: int = 24000,
) -> list[list[str]]:
    if max_items < 1:
        raise ValueError("max_items must be >= 1")
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for sample_id in sample_ids:
        sample_chars = len(sample_id) + (1 if current else 0)
        if current and (len(current) >= max_items or current_chars + sample_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(sample_id)
        current_chars += len(sample_id) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def _shared_xdg_env() -> dict[str, str]:
    XDG_DATA_HOME_ROOT.mkdir(parents=True, exist_ok=True)
    XDG_CACHE_HOME_ROOT.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache = XDG_CACHE_HOME_ROOT / "inspect_ai" / "hf_datasets"
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    return {
        "XDG_DATA_HOME": str(XDG_DATA_HOME_ROOT.resolve()),
        "XDG_CACHE_HOME": str(XDG_CACHE_HOME_ROOT.resolve()),
        "HF_DATASETS_CACHE": str(hf_datasets_cache.resolve()),
    }


def _build_log_dir(*, benchmark_id: str, model: str, root: str | Path = "data/inspect_logs") -> Path:
    model_name = _model_storage_component(model)
    return Path(root) / benchmark_id / model_name / time.strftime("%Y%m%d_%H%M%S")


def _build_inspect_env(*, log_dir: Path) -> dict[str, str]:
    env = _shared_xdg_env()
    trace_name = f"{log_dir.parent.name}__{log_dir.name}.trace.log"
    env["INSPECT_TRACE_FILE"] = str((log_dir / trace_name).resolve())
    return env


def _load_task_sample_ids(config: BenchmarkConfig) -> list[str]:
    cached = TASK_SAMPLE_ID_CACHE.get(config.benchmark_id)
    if cached is not None:
        return cached

    helper = (
        "import json, sys\n"
        "from inspect_ai._eval.loader import load_tasks\n"
        "task_spec = sys.argv[1]\n"
        "task_args = json.loads(sys.argv[2])\n"
        "tasks = load_tasks([task_spec], task_args)\n"
        "if len(tasks) != 1:\n"
        "    raise RuntimeError(f'Expected exactly one task for {task_spec}, found {len(tasks)}')\n"
        "task = tasks[0]\n"
        "sample_ids = []\n"
        "for idx in range(len(task.dataset)):\n"
        "    sample = task.dataset[idx]\n"
        "    sample_id = sample.id\n"
        "    if sample_id is None:\n"
        "        raise RuntimeError(f'Sample at index {idx} in {task_spec} did not have an id')\n"
        "    sample_ids.append(str(sample_id))\n"
        f"print('{SAMPLE_IDS_PREFIX}' + json.dumps(sample_ids))\n"
    )
    env = os.environ.copy()
    env.update(_shared_xdg_env())
    completed = subprocess.run(
        [
            str(INSPECT_PYTHON),
            "-c",
            helper,
            config.inspect_task,
            json.dumps(config.task_args, sort_keys=True),
        ],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"Failed to load sample ids for benchmark={config.benchmark_id} task={config.inspect_task}: "
            f"{details}"
        )

    sample_ids_payload: str | None = None
    for stream_text in (completed.stdout, completed.stderr):
        for raw_line in reversed(stream_text.splitlines()):
            line = raw_line.strip()
            if line.startswith(SAMPLE_IDS_PREFIX):
                sample_ids_payload = line.removeprefix(SAMPLE_IDS_PREFIX)
                break
        if sample_ids_payload is not None:
            break
    if sample_ids_payload is None:
        details = _tail_text((completed.stdout + "\n" + completed.stderr).strip(), max_chars=8000)
        raise RuntimeError(
            f"Did not find sample-id payload for benchmark={config.benchmark_id} task={config.inspect_task}. "
            f"Captured output:\n{details}"
        )

    sample_ids = json.loads(sample_ids_payload)
    if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
        raise RuntimeError(
            f"Invalid sample id payload for benchmark={config.benchmark_id}: {sample_ids!r}"
        )
    TASK_SAMPLE_ID_CACHE[config.benchmark_id] = sample_ids
    return sample_ids


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
    sample_ids: list[str] | None,
    epochs: int,
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
            "--epochs",
            str(epochs),
            "--max-tokens",
            str(MAX_TOKENS),
            "--max-retries",
            str(MAX_RETRIES),
        ]
    )
    if config.sandbox is not None:
        cmd.extend(["--sandbox", config.sandbox])
    if model_base_url is not None:
        cmd.extend(["--model-base-url", model_base_url])
    if sample_ids:
        cmd.extend(["--sample-id", ",".join(sample_ids)])

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


def _find_single_log_file(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No inspect log files found under {log_dir}")
    return candidates[-1]


def _tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _samplebuffer_root() -> Path:
    return XDG_DATA_HOME_ROOT / "inspect_ai" / "samplebuffer"


def _find_recent_samplebuffer_db(*, started_wall_time: float) -> Path | None:
    root = _samplebuffer_root()
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("*.db")
        if path.is_file() and path.stat().st_mtime >= (started_wall_time - 5.0)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_samplebuffer_progress(db_path: Path) -> tuple[int, int] | None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.2)
    except sqlite3.Error:
        return None
    try:
        cursor = conn.cursor()
        row_count = cursor.execute("select count(*) from samples").fetchone()
        if row_count is None:
            return None
        completed_count = int(row_count[0])
        retry_count = 0
        for (data_text,) in cursor.execute("select data from samples"):
            try:
                payload = json.loads(data_text)
            except Exception:
                continue
            retries = payload.get("retries")
            if isinstance(retries, int):
                retry_count += retries
        return completed_count, retry_count
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _format_eta_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_prometheus_value(text: str, metric_names: list[str], *, is_counter: bool) -> float | None:
    for metric_name in metric_names:
        total = 0.0
        found = False
        prefix = f"{metric_name}"
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith(prefix):
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


def _metrics_poller(metrics_url: str, interval: float, stop_event: threading.Event) -> None:
    prev_gen_tokens: float | None = None
    prev_prompt_tokens: float | None = None
    prev_time: float | None = None
    saw_activity = False
    gen_names = ["vllm:generation_tokens_total", "vllm_generation_tokens_total"]
    prompt_names = ["vllm:prompt_tokens_total", "vllm_prompt_tokens_total"]
    running_names = ["vllm:num_requests_running", "vllm_num_requests_running"]
    waiting_names = ["vllm:num_requests_waiting", "vllm_num_requests_waiting"]

    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        try:
            with urllib.request.urlopen(metrics_url, timeout=5) as resp:
                text = resp.read().decode()
        except Exception as exc:
            if saw_activity:
                _log(f"[metrics-poller] fetch failed: {exc}")
            continue

        now = time.time()
        gen_tokens = _parse_prometheus_value(text, gen_names, is_counter=True)
        prompt_tokens = _parse_prometheus_value(text, prompt_names, is_counter=True)
        running = _parse_prometheus_value(text, running_names, is_counter=False)
        waiting = _parse_prometheus_value(text, waiting_names, is_counter=False)
        active_requests = (
            (running is not None and running > 0)
            or (waiting is not None and waiting > 0)
        )

        if gen_tokens is not None and prev_gen_tokens is not None and prev_time is not None:
            dt = now - prev_time
            if dt > 0:
                gen_rate = (gen_tokens - prev_gen_tokens) / dt
                prompt_rate = ((prompt_tokens or 0.0) - (prev_prompt_tokens or 0.0)) / dt
                if active_requests or gen_rate > 0 or prompt_rate > 0 or saw_activity:
                    saw_activity = True
                    parts = [
                        f"gen_tok/s={gen_rate:.1f}",
                        f"prompt_tok/s={prompt_rate:.1f}",
                    ]
                    if running is not None:
                        parts.append(f"running={int(running)}")
                    if waiting is not None:
                        parts.append(f"waiting={int(waiting)}")
                    _log(f"[metrics-poller] {' '.join(parts)}")
        elif active_requests:
            saw_activity = True

        prev_gen_tokens = gen_tokens
        prev_prompt_tokens = prompt_tokens
        prev_time = now


def _checkpoint_path_for_output(output_path: Path) -> Path:
    if output_path.suffix == ".jsonl":
        return output_path.with_suffix(".ckpt.json")
    return output_path.with_name(output_path.name + ".ckpt.json")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _save_eci_checkpoint(
    *,
    ckpt_path: Path,
    benchmark_name: str,
    model: str,
    output_path: Path,
    total_samples: int,
    completed_samples: int,
    existing_records: int,
    new_records: int,
    checkpoint: int | None,
    elapsed_seconds: float,
    run_metadata: dict[str, Any],
    inspect_log_path: str | None = None,
) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_name": benchmark_name,
        "model": model,
        "output_path": str(output_path),
        "inspect_log_path": inspect_log_path,
        "total_samples": int(total_samples),
        "completed_samples": int(completed_samples),
        "remaining_samples": max(0, int(total_samples) - int(completed_samples)),
        "existing_records": int(existing_records),
        "new_records": int(new_records),
        "checkpoint": None if checkpoint is None else int(checkpoint),
        "elapsed_seconds": float(elapsed_seconds),
        "run_metadata": run_metadata,
    }
    _atomic_write_json(ckpt_path, payload)


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


def _extract_stop_reason(sample: dict[str, Any]) -> str | None:
    output = sample.get("output")
    if not isinstance(output, dict):
        return None

    choices = output.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            stop_reason = choice.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                return stop_reason

    stop_reason = output.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason:
        return stop_reason
    return None


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


def _extract_token_count_from_container(container: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = container.get(key)
        if isinstance(value, int):
            return value
    for nested in container.values():
        if not isinstance(nested, dict):
            continue
        for key in keys:
            value = nested.get(key)
            if isinstance(value, int):
                return value
    return 0


def _extract_token_count(sample: dict[str, Any], *keys: str) -> int:
    containers: list[dict[str, Any]] = []
    for container_key in ("model_usage", "usage", "role_usage"):
        container = sample.get(container_key)
        if isinstance(container, dict):
            containers.append(container)
    output = sample.get("output")
    if isinstance(output, dict):
        containers.append(output)
        output_usage = output.get("usage")
        if isinstance(output_usage, dict):
            containers.append(output_usage)

    for container in containers:
        value = _extract_token_count_from_container(container, *keys)
        if value > 0:
            return value

    for key in keys:
        value = sample.get(key)
        if isinstance(value, int):
            return value
    return 0


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
                stop_reason=_extract_stop_reason(sample),
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
                    "rendered_prompt": _extract_rendered_prompt(sample),
                    "prompt_messages": _extract_prompt_messages(sample),
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


def _run_inspect_command_once(
    *,
    config: BenchmarkConfig,
    model_path: str,
    inspect_model: str,
    sample_ids: list[str] | None,
    total_samples: int,
    epochs: int,
    rollout_offset: int,
    max_connections: int,
    sampling_params: dict[str, bool | float | int],
    run_metadata: dict[str, Any],
    model_base_url: str | None,
    extra_env: dict[str, str] | None,
    show_progress: bool = True,
) -> tuple[list[ECIScoreRecord], str, float]:
    log_dir = _build_log_dir(benchmark_id=config.benchmark_id, model=model_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_inspect_command(
        config=config,
        inspect_model=inspect_model,
        model_base_url=model_base_url,
        log_dir=log_dir,
        sample_ids=sample_ids,
        epochs=epochs,
        max_connections=max_connections,
        sampling_params=sampling_params,
    )
    env = os.environ.copy()
    env.update(_build_inspect_env(log_dir=log_dir))
    if extra_env is not None:
        env.update(extra_env)

    _log(
        f"[eci] running benchmark={config.benchmark_id} model={model_path} "
        f"epochs={epochs}",
    )
    inspect_stdout_path = log_dir / "inspect.stdout.log"
    inspect_stderr_path = log_dir / "inspect.stderr.log"
    started_at = time.monotonic()
    started_wall_time = time.time()
    samplebuffer_db_path: Path | None = None
    completed_count = 0
    retry_count = 0
    last_completed_count = 0
    last_retry_count = 0
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    progress = (
        tqdm(
            total=total_samples,
            desc=f"{model_path} eci",
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
        if show_progress and tqdm is not None and total_samples > 0
        else None
    )
    with open(inspect_stdout_path, "w", encoding="utf-8") as stdout_file, open(
        inspect_stderr_path, "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            last_status_print = 0.0
            poll_interval_seconds = 0.1
            while True:
                returncode = process.poll()
                now = time.monotonic()

                if samplebuffer_db_path is None:
                    samplebuffer_db_path = _find_recent_samplebuffer_db(started_wall_time=started_wall_time)
                if samplebuffer_db_path is not None:
                    progress_snapshot = _read_samplebuffer_progress(samplebuffer_db_path)
                    if progress_snapshot is not None:
                        completed_count, retry_count = progress_snapshot

                if total_samples > 0:
                    elapsed = max(0.001, now - started_at)
                    observed_completed = min(completed_count, total_samples)
                    postfix: dict[str, Any] = {"retry": retry_count}
                    if observed_completed > 0 and observed_completed < total_samples:
                        remaining = total_samples - observed_completed
                        eta_seconds = remaining * (elapsed / observed_completed)
                        postfix["eta"] = _format_eta_seconds(eta_seconds)
                    if progress is not None:
                        completed_delta = observed_completed - last_completed_count
                        if completed_delta > 0:
                            progress.update(completed_delta)
                        if completed_delta > 0 or retry_count != last_retry_count:
                            progress.set_postfix(postfix)
                            progress.refresh()
                    elif show_progress and (
                        observed_completed != last_completed_count
                        or retry_count != last_retry_count
                        or now - last_status_print >= 5.0
                    ):
                        parts = [
                            f"[eci] progress benchmark={config.benchmark_id}",
                            f"model={model_path}",
                            f"processed={observed_completed}/{total_samples}",
                            f"retry={retry_count}",
                        ]
                        if observed_completed > 0 and observed_completed < total_samples:
                            remaining = total_samples - observed_completed
                            eta_seconds = remaining * (elapsed / observed_completed)
                            parts.append(f"eta={_format_eta_seconds(eta_seconds)}")
                        _log(" ".join(parts))
                        last_status_print = now
                    last_completed_count = observed_completed
                    last_retry_count = retry_count

                if returncode is not None:
                    break
                time.sleep(poll_interval_seconds)
        finally:
            if progress is not None:
                observed_completed = min(completed_count if total_samples > 0 else 0, total_samples)
                completed_delta = observed_completed - last_completed_count
                if completed_delta > 0:
                    progress.update(completed_delta)
                progress.set_postfix({"retry": retry_count})
                progress.refresh()
                progress.close()

    completed = subprocess.CompletedProcess(cmd, process.returncode)
    elapsed_seconds = time.monotonic() - started_at
    if completed.returncode != 0:
        stdout_tail = _tail_text(inspect_stdout_path.read_text(encoding="utf-8").strip())
        stderr_tail = _tail_text(inspect_stderr_path.read_text(encoding="utf-8").strip())
        if stdout_tail:
            _log(
                f"[eci] inspect_stdout_tail benchmark={config.benchmark_id} model={model_path}\n{stdout_tail}",
            )
        if stderr_tail:
            _log(
                f"[eci] inspect_stderr_tail benchmark={config.benchmark_id} model={model_path}\n{stderr_tail}",
            )
        extra_hint = ""
        if model_base_url is not None and "APIConnectionError" in stderr_tail:
            extra_hint = (
                " local-vllm became unavailable during the run; check the submitit stderr/vLLM log "
                "for the underlying backend crash (often CUDA OOM)."
            )
        raise RuntimeError(
            "Inspect eval failed "
            f"(exit_code={completed.returncode}, stdout_log={inspect_stdout_path}, stderr_log={inspect_stderr_path})."
            f"{extra_hint}"
        )

    inspect_log_path = _find_single_log_file(log_dir)
    with open(inspect_log_path, "r", encoding="utf-8") as f:
        inspect_payload = json.load(f)
    rows = _extract_sample_rows(
        payload=inspect_payload,
        benchmark_id=config.benchmark_id,
        model_path=model_path,
        source_metric_names=config.source_metric_names,
        rollout_offset=rollout_offset,
        inspect_log_path=inspect_log_path,
        run_metadata=run_metadata,
    )
    return rows, str(inspect_log_path), elapsed_seconds


def run_eci_benchmark(
    *,
    config: BenchmarkConfig,
    model_path: str,
    backend: str,
    limit: int | None,
    max_connections: int,
    sampling_params: dict[str, bool | float | int],
    run_metadata: dict[str, Any],
    checkpoint: int | None,
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
    existing_rows = _dedupe_rows(_read_existing_rows(output_path))
    existing_ids = {row.inference_id for row in existing_rows}
    dataset_problem_ids = _load_task_sample_ids(config)
    target_problem_ids = dataset_problem_ids[:limit] if limit is not None else dataset_problem_ids
    target_problem_id_set = set(target_problem_ids)
    scoped_existing_rows = [
        row
        for row in existing_rows
        if row.problem_id in target_problem_id_set and 1 <= row.rollout_id <= EPOCHS
    ]
    total_target_samples = len(target_problem_ids) * EPOCHS
    inspect_log_path = ""
    elapsed_seconds = 0.0
    new_count = 0
    ckpt_path = _checkpoint_path_for_output(output_path)
    chunk_sample_count = (
        max(1, (checkpoint + max(1, EPOCHS) - 1) // max(1, EPOCHS))
        if checkpoint is not None
        else None
    )
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    benchmark_progress = (
        tqdm(
            total=total_target_samples,
            initial=len(scoped_existing_rows),
            desc=f"{config.benchmark_id} {model_path}",
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
        if tqdm is not None and total_target_samples > 0 and sys.stdout.isatty()
        else None
    )

    def _update_checkpoint_and_progress(batch_added: int) -> None:
        completed_samples = len(scoped_existing_rows) + new_count
        if benchmark_progress is not None and batch_added > 0:
            benchmark_progress.update(batch_added)
            benchmark_progress.set_postfix({"saved": new_count})
            benchmark_progress.refresh()
        elif batch_added > 0:
            _log(
                f"[eci] checkpoint benchmark={config.benchmark_id} model={model_path} "
                f"completed={completed_samples}/{total_target_samples} saved={new_count}",
            )
        _save_eci_checkpoint(
            ckpt_path=ckpt_path,
            benchmark_name=config.benchmark_id,
            model=model_path,
            output_path=output_path,
            total_samples=total_target_samples,
            completed_samples=completed_samples,
            existing_records=len(scoped_existing_rows),
            new_records=new_count,
            checkpoint=checkpoint,
            elapsed_seconds=elapsed_seconds,
            run_metadata=run_metadata,
            inspect_log_path=inspect_log_path or None,
        )

    try:
        _save_eci_checkpoint(
            ckpt_path=ckpt_path,
            benchmark_name=config.benchmark_id,
            model=model_path,
            output_path=output_path,
            total_samples=total_target_samples,
            completed_samples=len(scoped_existing_rows),
            existing_records=len(scoped_existing_rows),
            new_records=new_count,
            checkpoint=checkpoint,
            elapsed_seconds=elapsed_seconds,
            run_metadata=run_metadata,
            inspect_log_path=None,
        )

        if not existing_ids:
            if checkpoint is None:
                sample_id_batches: list[list[str] | None] = [target_problem_ids if limit is not None else None]
            else:
                sample_id_batches = _chunk_sample_ids_with_limits(
                    target_problem_ids,
                    max_items=chunk_sample_count,
                )
            for sample_id_batch in sample_id_batches:
                batch_added = 0
                new_rows, inspect_log_path, run_elapsed_seconds = _run_inspect_command_once(
                    config=config,
                    model_path=model_path,
                    inspect_model=inspect_model,
                    sample_ids=sample_id_batch,
                    total_samples=(len(sample_id_batch) if sample_id_batch is not None else len(target_problem_ids)) * EPOCHS,
                    epochs=EPOCHS,
                    rollout_offset=0,
                    max_connections=max_connections,
                    sampling_params=sampling_params,
                    run_metadata=run_metadata,
                    model_base_url=model_base_url,
                    extra_env=extra_env,
                    show_progress=False,
                )
                for row in new_rows:
                    if row.inference_id in existing_ids:
                        continue
                    append_jsonl(output_path, row)
                    existing_ids.add(row.inference_id)
                    new_count += 1
                    batch_added += 1
                elapsed_seconds += run_elapsed_seconds
                _update_checkpoint_and_progress(batch_added)
        else:
            for rollout_id in range(1, EPOCHS + 1):
                missing_problem_ids = [
                    problem_id
                    for problem_id in target_problem_ids
                    if _eci_inference_id(
                        benchmark_name=config.benchmark_id,
                        model=model_path,
                        problem_id=problem_id,
                        rollout_id=rollout_id,
                    )
                    not in existing_ids
                ]
                if not missing_problem_ids:
                    _log(
                        f"[eci] skip benchmark={config.benchmark_id} model={model_path} rollout={rollout_id}",
                    )
                    continue

                _log(
                    f"[eci] pending benchmark={config.benchmark_id} model={model_path} "
                    f"rollout={rollout_id} pending={len(missing_problem_ids)}",
                )
                if checkpoint is None:
                    sample_id_batches = _chunk_sample_ids(missing_problem_ids)
                else:
                    sample_id_batches = _chunk_sample_ids_with_limits(
                        missing_problem_ids,
                        max_items=max(1, checkpoint),
                    )
                for sample_id_batch in sample_id_batches:
                    batch_added = 0
                    new_rows, inspect_log_path, run_elapsed_seconds = _run_inspect_command_once(
                        config=config,
                        model_path=model_path,
                        inspect_model=inspect_model,
                        sample_ids=sample_id_batch,
                        total_samples=len(sample_id_batch),
                        epochs=1,
                        rollout_offset=rollout_id - 1,
                        max_connections=max_connections,
                        sampling_params=sampling_params,
                        run_metadata=run_metadata,
                        model_base_url=model_base_url,
                        extra_env=extra_env,
                        show_progress=False,
                    )
                    for row in new_rows:
                        if row.inference_id in existing_ids:
                            continue
                        append_jsonl(output_path, row)
                        existing_ids.add(row.inference_id)
                        new_count += 1
                        batch_added += 1
                    elapsed_seconds += run_elapsed_seconds
                    _update_checkpoint_and_progress(batch_added)
    finally:
        if benchmark_progress is not None:
            benchmark_progress.close()

    final_rows = _dedupe_rows(_read_existing_rows(output_path))
    scoped_rows = [
        row
        for row in final_rows
        if row.problem_id in target_problem_id_set and 1 <= row.rollout_id <= EPOCHS
    ]
    if not scoped_rows:
        raise ValueError(
            f"No scored rows found for benchmark={config.benchmark_id} model={model_path} after running."
        )
    source_metric, accuracy, rollout_count = _summarize_rollout_rows(
        rows=scoped_rows,
        source_metric_names=config.source_metric_names,
    )

    return BenchmarkRunSummary(
        benchmark_id=config.benchmark_id,
        output_path=str(output_path),
        inspect_log_path=str(inspect_log_path),
        source_metric=source_metric,
        accuracy=accuracy,
        rollout_count=rollout_count,
        existing_count=len(scoped_existing_rows),
        new_count=new_count,
        elapsed_seconds=elapsed_seconds,
    )


def run_eci_benchmarks(
    *,
    benchmark_names: list[str],
    model_path: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    sampling_params: dict[str, bool | float | int],
    run_metadata: dict[str, Any],
    limit: int | None,
    max_connections: int,
    checkpoint: int | None,
    gpu_memory_utilization: float,
    dtype: str,
    backend: str,
) -> dict[str, Any]:
    summaries: list[BenchmarkRunSummary] = []

    def _run_all(model_base_url: str | None, extra_env: dict[str, str] | None) -> None:
        for benchmark_name in benchmark_names:
            summary = run_eci_benchmark(
                config=BENCHMARK_CONFIGS[benchmark_name],
                model_path=model_path,
                backend=backend,
                limit=limit,
                max_connections=max_connections,
                sampling_params=sampling_params,
                run_metadata=run_metadata,
                checkpoint=checkpoint,
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
            metrics_url = f"http://localhost:{server.port}/metrics"
            metrics_stop = threading.Event()
            metrics_thread = threading.Thread(
                target=_metrics_poller,
                args=(metrics_url, 5.0, metrics_stop),
                daemon=True,
            )
            metrics_thread.start()
            try:
                _run_all(
                    model_base_url=base_url,
                    extra_env={
                        "VLLM_BASE_URL": base_url,
                        "VLLM_API_KEY": "local",
                    },
                )
            finally:
                metrics_stop.set()
                metrics_thread.join(timeout=1)
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
