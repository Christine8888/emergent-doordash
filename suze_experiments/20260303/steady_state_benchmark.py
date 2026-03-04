"""Standalone steady-state throughput benchmark for inspect and direct vLLM paths.

One config per run:
    python suze_experiments/20260303/steady_state_benchmark.py \
      --backend inspect \
      --config suze_experiments/20260303/configs/inspect_v2.yaml \
      --warmup-seconds 180 \
      --measure-seconds 60


    python suze_experiments/20260303/steady_state_benchmark.py \
      --backend direct \
      --config suze_experiments/20260303/configs/batch_v2.yaml \
      --warmup-seconds 180 \
      --measure-seconds 60
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from typing import Any
import urllib.request
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver

from utils.vllm_server import vLLMServer
from experiments.runner import setup_vllm_env
from environments.aime.aime import aime_scorer

logger = logging.getLogger(__name__)

AIME_DATASET_PATH = "di-zhang-fdu/AIME_1983_2024"
INSTRUCTIONS_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your '
    'response should be of the form "ANSWER: $ANSWER" (without quotes) '
    'where $ANSWER is the answer to the problem.'
)

INSPECT_REQUIRED_KEYS = [
    "model_path", "tensor_parallel_size",
    "max_model_len", "gpu_memory_utilization",
    "enable_prefix_caching", "enable_chunked_prefill", "dtype",
    "max_num_batched_tokens", "max_num_seqs",
    "num_samples", "seed", "shuffle",
    "hint_fraction", "hint_type", "solver_type", "mode",
    "mask_token", "stop_string", "hint_prefix",
    "max_tokens", "timeout", "temperature", "top_p", "top_k", "presence_penalty",
    "max_connections", "epochs", "max_retries", "retry_on_error", "fail_on_error", "display",
    "log_dir",
]

DIRECT_REQUIRED_KEYS = [
    "model_path",
    "max_model_len", "gpu_memory_utilization", "dtype",
    "max_num_batched_tokens", "max_num_seqs",
    "num_samples", "seed", "shuffle",
    "hint_fraction", "hint_type", "solver_type", "mode",
    "mask_token", "stop_string", "hint_prefix",
    "max_tokens", "temperature", "top_p", "top_k", "presence_penalty",
    "log_dir",
]


class ThroughputTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._start_ts = time.time()
        self._total_out_tokens = 0
        self._total_requests = 0
        self._trace: list[tuple[float, int, int]] = []

    def reset(self):
        with self._lock:
            self._start_ts = time.time()
            self._total_out_tokens = 0
            self._total_requests = 0
            self._trace = []

    def record(self, out_tokens: int | None):
        now = time.time()
        with self._lock:
            # Count only successful generations (where output token usage is known).
            if isinstance(out_tokens, int):
                self._total_requests += 1
                self._total_out_tokens += out_tokens
                self._trace.append(
                    (
                        max(now - self._start_ts, 0.0),
                        self._total_out_tokens,
                        self._total_requests,
                    )
                )

    def snapshot(self) -> tuple[list[tuple[float, int, int]], int, int]:
        with self._lock:
            return list(self._trace), self._total_out_tokens, self._total_requests


def _extract_output_tokens(output) -> int | None:
    if output is None:
        return None
    usage = getattr(output, "usage", None)
    if usage is not None:
        for attr in ("output_tokens", "completion_tokens"):
            val = getattr(usage, attr, None)
            if isinstance(val, int):
                return val
        if isinstance(usage, dict):
            for key in ("output_tokens", "completion_tokens"):
                val = usage.get(key)
                if isinstance(val, int):
                    return val
    return None


def _extract_finished_time(output) -> float | None:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    for attr in ("finished_time", "finished_ts", "completion_time"):
        val = getattr(metrics, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _interpolate_value_at_time(
    trace: list[tuple[float, int, int]],
    target_time: float,
    field: str,
) -> float:
    if target_time <= 0:
        return 0.0

    prev_t = 0.0
    prev_v = 0.0
    for t, tok, req in trace:
        value = float(tok if field == "tokens" else req)
        if t >= target_time:
            if t <= prev_t:
                return value
            ratio = (target_time - prev_t) / (t - prev_t)
            return prev_v + ratio * (value - prev_v)
        prev_t = t
        prev_v = value
    return prev_v


def _compute_window_throughput(
    trace: list[tuple[float, int, int]],
    warmup_seconds: float,
    measure_seconds: float,
) -> dict[str, float]:
    start = warmup_seconds
    end = warmup_seconds + measure_seconds
    if end <= start:
        raise ValueError(f"Invalid window (warmup={warmup_seconds}, measure={measure_seconds})")

    start_tokens = _interpolate_value_at_time(trace, start, "tokens")
    end_tokens = _interpolate_value_at_time(trace, end, "tokens")
    start_requests = _interpolate_value_at_time(trace, start, "requests")
    end_requests = _interpolate_value_at_time(trace, end, "requests")

    window_tokens = max(end_tokens - start_tokens, 0.0)
    window_requests = max(end_requests - start_requests, 0.0)

    return {
        "output_tokens": window_tokens,
        "requests": window_requests,
        "output_tokens_per_sec": window_tokens / measure_seconds,
        "samples_per_sec": window_requests / measure_seconds,
    }


def _trace_end_seconds(trace: list[tuple[float, int, int]]) -> float:
    if not trace:
        return 0.0
    return float(trace[-1][0])


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S")

def _parse_prometheus_value(text: str, metric_names: list[str], is_counter: bool = True) -> float | None:
    """Sum all matching lines for a Prometheus counter, or return last gauge value.
    Tries each name in metric_names until one matches."""
    for metric_name in metric_names:
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if not line.startswith(metric_name):
                continue
            rest = line[len(metric_name):]
            if rest and rest[0] not in ("{", " "):
                continue
            val_str = line.rsplit("}", 1)[-1].strip() if "{" in line else line.split()[-1]
            try:
                v = float(val_str)
                if is_counter:
                    total += v
                    found = True
                else:
                    return v
            except ValueError:
                pass
        if found:
            return total
    return None


def _metrics_poller(port: int, interval: float, stop_event: threading.Event):
    """Background thread: poll vLLM /metrics and print throughput."""
    url = f"http://localhost:{port}/metrics"
    prev_gen_tokens: float | None = None
    prev_prompt_tokens: float | None = None
    prev_time: float | None = None
    first_fetch = True
    start_time = time.time()

    # Try both colon and underscore variants
    GEN_NAMES = ["vllm:generation_tokens_total", "vllm_generation_tokens_total"]
    PROMPT_NAMES = ["vllm:prompt_tokens_total", "vllm_prompt_tokens_total"]
    RUNNING_NAMES = ["vllm:num_requests_running", "vllm_num_requests_running"]
    WAITING_NAMES = ["vllm:num_requests_waiting", "vllm_num_requests_waiting"]
    KV_NAMES = ["vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc",
                "vllm:gpu_cache_usage_percent", "vllm_gpu_cache_usage_percent"]

    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                text = resp.read().decode()
        except Exception as exc:
            print(f"[metrics-poller] fetch failed: {exc}", file=sys.stderr, flush=True)
            continue

        if first_fetch:
            # Dump available metric names (non-comment, non-empty) for debugging
            available = sorted({
                line.split("{")[0].split(" ")[0]
                for line in text.splitlines()
                if line and not line.startswith("#")
            })
            print(
                f"[metrics-poller] available metrics ({len(available)}): "
                + ", ".join(available[:20])
                + ("..." if len(available) > 20 else ""),
                file=sys.stderr, flush=True,
            )
            first_fetch = False

        now = time.time()
        gen_tokens = _parse_prometheus_value(text, GEN_NAMES)
        prompt_tokens = _parse_prometheus_value(text, PROMPT_NAMES)
        running = _parse_prometheus_value(text, RUNNING_NAMES, is_counter=False)
        waiting = _parse_prometheus_value(text, WAITING_NAMES, is_counter=False)
        kv_usage = _parse_prometheus_value(text, KV_NAMES, is_counter=False)

        if gen_tokens is not None and prev_gen_tokens is not None and prev_time is not None:
            dt = now - prev_time
            if dt > 0:
                gen_rate = (gen_tokens - prev_gen_tokens) / dt
                prompt_rate = ((prompt_tokens or 0) - (prev_prompt_tokens or 0)) / dt
                elapsed_seconds = max(int(now - start_time), 0)
                elapsed_h = elapsed_seconds // 3600
                elapsed_m = (elapsed_seconds % 3600) // 60
                elapsed_s = elapsed_seconds % 60
                parts = [
                    f"elapsed={elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}",
                    f"gen_tok/s={gen_rate:.1f}",
                    f"prompt_tok/s={prompt_rate:.1f}",
                ]
                if running is not None:
                    parts.append(f"running={int(running)}")
                if waiting is not None:
                    parts.append(f"waiting={int(waiting)}")
                if kv_usage is not None:
                    parts.append(f"kv_cache={kv_usage:.1%}")
                print(f"[metrics-poller] {' '.join(parts)}", file=sys.stderr, flush=True)
        elif gen_tokens is None:
            print("[metrics-poller] WARNING: no generation token counter found", file=sys.stderr, flush=True)

        prev_gen_tokens = gen_tokens
        prev_prompt_tokens = prompt_tokens
        prev_time = now

def load_config(path: str, required_keys: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping, got {type(cfg)}")
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config file {path} missing required keys: {missing}")
    return cfg


def _truncate_at_stop_string(text: str, stop_string: str) -> str:
    if stop_string not in text:
        return text
    return text[:text.index(stop_string)].strip()


def _split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    tokens = re.split(r"(\s+)", text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    return tokens, word_indices


def get_prefill_fraction(text: str, fraction: float, stop_string: str) -> str:
    text = _truncate_at_stop_string(text, stop_string)
    tokens, word_indices = _split_preserving_whitespace(text)
    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text
    last_word_idx = word_indices[num_words - 1]
    return "".join(tokens[: last_word_idx + 1]).strip()


def get_masked_text(
    text: str,
    fraction: float,
    mask_token: str,
    stop_string: str,
    seed: int | str | None = None,
) -> str:
    text = _truncate_at_stop_string(text, stop_string)
    tokens, word_indices = _split_preserving_whitespace(text)
    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask == 0:
        return text
    rng = random.Random(seed) if seed is not None else random
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join(mask_token if i in mask_indices else t for i, t in enumerate(tokens)).strip()


def load_hints(path: str, mode: str, fraction: float, stop_string: str) -> dict[str, dict[int, str]]:
    if fraction == 0.0:
        return {}
    hint_data: dict[str, dict[int, str]] = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            hint = record.get("hint", "")
            if not isinstance(hint, str) or not hint.strip():
                continue
            sid = record["id"]
            sample_idx = record.get("sample_idx", 0)
            if mode == "sequential":
                hint = get_prefill_fraction(hint, fraction=fraction, stop_string=stop_string)
            if sid not in hint_data:
                hint_data[sid] = {}
            hint_data[sid][sample_idx] = hint
    return hint_data


def _record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["ID"],
        input=record["Question"],
        target=str(record["Answer"]),
        metadata={"year": record["Year"], "problem_number": record["Problem Number"]},
    )


def load_aime_dataset_for_inspect(sample_ids: set[str] | None, shuffle: bool):
    dataset = hf_dataset(
        path=AIME_DATASET_PATH,
        split="train",
        sample_fields=_record_to_sample,
        shuffle=shuffle,
    )
    if sample_ids is not None:
        dataset = dataset.filter(
            name=dataset.name,
            predicate=lambda sample: sample.id in sample_ids,
        )
    return dataset


def subsample_ids(hint_jsonl: Path, num_samples: int, seed: int) -> set[str]:
    ids: set[str] = set()
    with open(hint_jsonl) as f:
        for line in f:
            ids.add(json.loads(line)["id"])
    all_ids = sorted(ids)
    if num_samples >= len(all_ids):
        return set(all_ids)
    rng = random.Random(seed)
    return set(rng.sample(all_ids, num_samples))


@solver
def instructions_solver(template: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = template + "\n\n" + state.user_prompt.text
        return state
    return solve


@solver
def intext_solver(hint_data: dict[str, dict[int, str]], cfg: dict) -> Solver:
    fraction = cfg["hint_fraction"]
    mode = cfg["mode"]
    mask_token = cfg["mask_token"]
    stop_string = cfg["stop_string"]
    prefix = cfg["hint_prefix"]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if fraction > 0.0:
            sid = str(state.sample_id)
            if sid not in hint_data:
                raise KeyError(f"Sample '{sid}' not found in hint data")
            samples = hint_data[sid]
            rng = random.Random(f"{state.epoch}_{state.sample_id}")
            chosen_idx = rng.choice(sorted(samples.keys()))
            hint_text = samples[chosen_idx]
            if mode == "masked":
                mask_seed = f"{state.epoch}_{state.sample_id}_{chosen_idx}"
                hint_text = get_masked_text(
                    hint_text,
                    fraction=fraction,
                    mask_token=mask_token,
                    stop_string=stop_string,
                    seed=mask_seed,
                )
            state.user_prompt.text = state.user_prompt.text + "\n\n" + prefix + hint_text
        return state

    return solve


@solver
def prefill_solver(hint_data: dict[str, dict[int, str]], cfg: dict) -> Solver:
    fraction = cfg["hint_fraction"]
    mode = cfg["mode"]
    mask_token = cfg["mask_token"]
    stop_string = cfg["stop_string"]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if fraction > 0.0:
            sid = str(state.sample_id)
            if sid not in hint_data:
                raise KeyError(f"Sample '{sid}' not found in hint data")
            samples = hint_data[sid]
            rng = random.Random(f"{state.epoch}_{state.sample_id}")
            chosen_idx = rng.choice(sorted(samples.keys()))
            prefill_text = samples[chosen_idx]
            if mode == "masked":
                prefill_text = get_masked_text(
                    prefill_text,
                    fraction=fraction,
                    mask_token=mask_token,
                    stop_string=stop_string,
                )
            state.messages.append(ChatMessageAssistant(content=prefill_text))
        return state

    return solve


@solver
def generate_solver(cfg: dict, tracker: ThroughputTracker) -> Solver:
    max_tokens = cfg["max_tokens"]

    async def solve(state: TaskState, gen: Generate) -> TaskState:
        continue_message = (
            len(state.messages) > 0 and isinstance(state.messages[-1], ChatMessageAssistant)
        )
        state = await gen(
            state,
            max_tokens=max_tokens,
            timeout=cfg["timeout"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            presence_penalty=cfg["presence_penalty"],
            continue_final_message=continue_message,
        )
        if state.output:
            tracker.record(_extract_output_tokens(state.output))
        return state

    return solve


def build_inspect_task(cfg: dict, sample_ids: set[str], hint_data: dict[str, dict[int, str]], tracker: ThroughputTracker) -> Task:
    hint_solver = intext_solver(hint_data, cfg) if cfg["solver_type"] == "intext" else prefill_solver(hint_data, cfg)
    solvers = [
        instructions_solver(INSTRUCTIONS_TEMPLATE),
        hint_solver,
        generate_solver(cfg, tracker),
    ]
    dataset = load_aime_dataset_for_inspect(sample_ids, shuffle=cfg["shuffle"])
    return Task(dataset=dataset, solver=solvers, scorer=aime_scorer())


def load_aime_raw(num_samples: int, seed: int, shuffle: bool, hint_jsonl: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(AIME_DATASET_PATH, split="train")
    ids_with_hints: set[str] = set()
    with open(hint_jsonl) as f:
        for line in f:
            ids_with_hints.add(json.loads(line)["id"])
    all_ids = sorted(ids_with_hints)

    rng = random.Random(seed)
    if num_samples < len(all_ids):
        selected = set(rng.sample(all_ids, num_samples))
    else:
        selected = set(all_ids)

    records = []
    for row in ds:
        if row["ID"] in selected:
            records.append(
                {
                    "id": row["ID"],
                    "question": row["Question"],
                    "answer": str(row["Answer"]),
                    "year": row["Year"],
                    "problem_number": row["Problem Number"],
                }
            )
    if shuffle:
        rng.shuffle(records)
    return records


def build_prompts(records: list[dict], cfg: dict, hint_data: dict[str, dict[int, str]]) -> list[dict]:
    prompts = []
    for rec in records:
        sid = rec["id"]
        question = rec["question"]
        user_text = INSTRUCTIONS_TEMPLATE + "\n\n" + question

        if cfg["hint_fraction"] > 0 and sid in hint_data:
            samples = hint_data[sid]
            rng = random.Random(f"1_{sid}")
            chosen_idx = rng.choice(sorted(samples.keys()))
            hint_text = samples[chosen_idx]

            if cfg["mode"] == "masked":
                hint_text = get_masked_text(
                    hint_text,
                    fraction=cfg["hint_fraction"],
                    mask_token=cfg["mask_token"],
                    stop_string=cfg["stop_string"],
                )

            if cfg["solver_type"] == "intext":
                user_text += "\n\n" + cfg["hint_prefix"] + hint_text
                messages = [{"role": "user", "content": user_text}]
            else:
                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": hint_text},
                ]
        else:
            messages = [{"role": "user", "content": user_text}]

        prompts.append({"id": sid, "answer": rec["answer"], "messages": messages})
    return prompts


def run_inspect(cfg: dict, warmup_seconds: float, measure_seconds: float, run_name: str) -> dict[str, Any]:
    model_name = os.path.basename(cfg["model_path"])
    data_path = REPO_ROOT / "christine_experiments" / "data" / cfg["hint_type"] / "aime.jsonl"
    sample_ids = subsample_ids(data_path, cfg["num_samples"], cfg["seed"])
    hint_data = load_hints(
        path=str(data_path),
        mode=cfg["mode"],
        fraction=cfg["hint_fraction"],
        stop_string=cfg["stop_string"],
    )
    tracker = ThroughputTracker()
    task = build_inspect_task(cfg, sample_ids, hint_data, tracker)
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    server_kwargs: dict[str, Any] = {}
    if cfg["max_num_seqs"] is not None:
        server_kwargs["max_num_seqs"] = cfg["max_num_seqs"]

    with vLLMServer(
        model_path=cfg["model_path"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=cfg["enable_prefix_caching"],
        enable_chunked_prefill=cfg["enable_chunked_prefill"],
        dtype=cfg["dtype"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        **server_kwargs,
    ) as server:
        setup_vllm_env(server.port, model_name)
        os.environ["VLLM_MAX_MODEL_LEN"] = str(cfg["max_model_len"])
        os.environ["INSPECT_EVAL_MODEL"] = f"vllm/{model_name}"

        # Start polling vLLM /metrics in background
        poll_interval = cfg.get("poll_interval", 10.0)
        stop_event = threading.Event()
        poller = threading.Thread(
            target=_metrics_poller,
            args=(server.port, poll_interval, stop_event),
            daemon=True,
        )
        poller.start()
        logger.info(
            "Polling vLLM /metrics every %.0fs. Ctrl+C to stop early.", poll_interval
        )

        tracker.reset()
        t0 = time.time()
        try:
            logging.getLogger("inspect_ai").setLevel(logging.DEBUG) # TEMP
            logging.getLogger("httpx").setLevel(logging.DEBUG) # TEMP

            eval_log = inspect_eval(
                task,
                model=f"vllm/{model_name}",
                log_dir=str(log_dir),
                epochs=1,
                limit=None,
                max_connections=cfg["max_connections"],
                max_retries=cfg["max_retries"],
                display=cfg["display"],
                fail_on_error=cfg["fail_on_error"],
                retry_on_error=cfg["retry_on_error"],
            )
            log = eval_log[0]
            if log.status != "success":
                error_msg = f"Inspect round failed: status={log.status}"
                if log.error:
                    error_msg += f", error={log.error.message}"
                raise RuntimeError(error_msg)
        except KeyboardInterrupt:
            logger.info("Interrupted by user — stopping early.")
        finally:
            stop_event.set()
            poller.join(timeout=5)
        elapsed = time.time() - t0

    trace_snapshot, total_output_tokens, total_requests = tracker.snapshot()
    measured = _compute_window_throughput(trace_snapshot, warmup_seconds, measure_seconds)
    trace_end = _trace_end_seconds(trace_snapshot)
    target_end = warmup_seconds + measure_seconds
    if trace_end < target_end:
        logger.warning(
            "[inspect] trace ends before measurement window end (trace_end=%.1fs, window_end=%.1fs). "
            "Increase load or measurement duration for a more stable estimate.",
            trace_end,
            target_end,
        )

    result = {
        "kind": "inspect",
        "run_name": run_name,
        "elapsed_seconds": round(elapsed, 1),
        "rounds_completed": 1,
        "measurement_window": {
            "warmup_seconds": warmup_seconds,
            "measure_seconds": measure_seconds,
            "output_tokens": round(measured["output_tokens"], 1),
            "requests": round(measured["requests"], 1),
            "trace_end_seconds": round(trace_end, 1),
        },
        "throughput": {
            "output_tokens_per_sec": round(measured["output_tokens_per_sec"], 1),
            "samples_per_sec": round(measured["samples_per_sec"], 4),
        },
        "totals": {
            "output_tokens": total_output_tokens,
            "requests": total_requests,
        },
    }
    return result


def run_direct(cfg: dict, warmup_seconds: float, measure_seconds: float, run_name: str) -> dict[str, Any]:
    hint_jsonl = REPO_ROOT / "christine_experiments" / "data" / cfg["hint_type"] / "aime.jsonl"
    records = load_aime_raw(cfg["num_samples"], cfg["seed"], cfg["shuffle"], hint_jsonl)
    hint_data = load_hints(
        path=str(hint_jsonl),
        mode=cfg["mode"],
        fraction=cfg["hint_fraction"],
        stop_string=cfg["stop_string"],
    )
    prompts = build_prompts(records, cfg, hint_data)
    conversations = [p["messages"] for p in prompts]

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {}
    if cfg["max_num_batched_tokens"] is not None:
        llm_kwargs["max_num_batched_tokens"] = cfg["max_num_batched_tokens"]
    if cfg["max_num_seqs"] is not None:
        llm_kwargs["max_num_seqs"] = cfg["max_num_seqs"]

    llm = LLM(
        model=cfg["model_path"],
        dtype=cfg["dtype"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        seed=cfg["seed"],
        enable_prefix_caching=cfg.get("enable_prefix_caching", False),
        enable_chunked_prefill=cfg.get("enable_chunked_prefill", False),
        **llm_kwargs,
    )
    sampling_params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        presence_penalty=cfg["presence_penalty"],
        max_tokens=cfg["max_tokens"],
    )

    finish_records: list[tuple[float, int]] = []
    total_output_tokens = 0
    total_requests = 0
    max_tokens_hits = 0

    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    round_end = time.time()
    for out in outputs:
        out_tokens = len(out.outputs[0].token_ids)
        total_output_tokens += out_tokens
        total_requests += 1
        if out_tokens >= cfg["max_tokens"]:
            max_tokens_hits += 1
        finished_time = _extract_finished_time(out)
        rel_time = max((finished_time if finished_time is not None else round_end) - t0, 0.0)
        finish_records.append((rel_time, out_tokens))
    elapsed = time.time() - t0

    finish_records.sort(key=lambda x: x[0])
    cumulative_tokens = 0
    cumulative_requests = 0
    trace: list[tuple[float, int, int]] = []
    for rel_time, out_tokens in finish_records:
        cumulative_requests += 1
        cumulative_tokens += out_tokens
        trace.append((rel_time, cumulative_tokens, cumulative_requests))

    measured = _compute_window_throughput(trace, warmup_seconds, measure_seconds)
    trace_end = _trace_end_seconds(trace)
    target_end = warmup_seconds + measure_seconds
    if trace_end < target_end:
        logger.warning(
            "[direct] trace ends before measurement window end (trace_end=%.1fs, window_end=%.1fs). "
            "Increase load or measurement duration for a more stable estimate.",
            trace_end,
            target_end,
        )

    result = {
        "kind": "direct",
        "run_name": run_name,
        "elapsed_seconds": round(elapsed, 1),
        "rounds_completed": 1,
        "measurement_window": {
            "warmup_seconds": warmup_seconds,
            "measure_seconds": measure_seconds,
            "output_tokens": round(measured["output_tokens"], 1),
            "requests": round(measured["requests"], 1),
            "trace_end_seconds": round(trace_end, 1),
        },
        "throughput": {
            "output_tokens_per_sec": round(measured["output_tokens_per_sec"], 1),
            "samples_per_sec": round(measured["samples_per_sec"], 4),
        },
        "totals": {
            "output_tokens": total_output_tokens,
            "requests": total_requests,
            "max_tokens_hits": max_tokens_hits,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Standalone fixed-window throughput benchmark.")
    parser.add_argument("--backend", choices=["inspect", "direct"], required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to one YAML config file.")
    parser.add_argument("--warmup-seconds", type=float, required=True)
    parser.add_argument("--measure-seconds", type=float, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path. Defaults under config log_dir.",
    )
    args = parser.parse_args()

    if args.warmup_seconds < 0:
        raise ValueError("--warmup-seconds must be >= 0")
    if args.measure_seconds <= 0:
        raise ValueError("--measure-seconds must be > 0")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_name = f"steady_{args.backend}_{Path(args.config).stem}_{_timestamp()}"

    if args.backend == "inspect":
        cfg = load_config(args.config, INSPECT_REQUIRED_KEYS)
        result = run_inspect(cfg, args.warmup_seconds, args.measure_seconds, run_name)
    else:
        cfg = load_config(args.config, DIRECT_REQUIRED_KEYS)
        result = run_direct(cfg, args.warmup_seconds, args.measure_seconds, run_name)

    result["config_path"] = args.config
    result["config"] = cfg

    if args.out is None:
        out_dir = Path(cfg["log_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_name}.json"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to {out_path}")
    print(
        f"Throughput: output_tok/s={result['throughput']['output_tokens_per_sec']} "
        f"samples/s={result['throughput']['samples_per_sec']}"
    )


if __name__ == "__main__":
    main()
