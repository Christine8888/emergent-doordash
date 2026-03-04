"""Self-contained debug inference script for AIME evaluation.

Loads a YAML config, starts a local vLLM server, runs AIME eval via
inspect_ai.eval(), and prints results. No SLURM, no submitit, no checkpointing.

All solver/task-building logic is local to this file -- the only production
imports are vLLMServer (infrastructure), setup_vllm_env (env vars), and
aime_scorer (complex sympy grading).

Usage:
    python suze_experiments/20260301/debug_inference.py \
        --config suze_experiments/20260301/configs/original.yaml

    python suze_experiments/20260301/debug_inference.py \
        --config suze_experiments/20260301/configs/lower_model_len.yaml

    python suze_experiments/20260301/debug_inference.py \
        --config suze_experiments/20260301/configs/lower_model_len_more_connections.yaml

    python suze_experiments/20260301/debug_inference.py \
        --config suze_experiments/20260301/configs/higher_max_tok.yaml

    python suze_experiments/20260301/debug_inference.py \
        --config suze_experiments/20260301/configs/high_max_model_len.yaml

"""

from pathlib import Path
import sys
import os
import json
import random
import re
import argparse
import logging
import threading
import time
from typing import Any

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

REQUIRED_KEYS = [
    "model_path", "tensor_parallel_size",
    "max_model_len", "gpu_memory_utilization",
    "enable_prefix_caching", "enable_chunked_prefill", "dtype",
    "max_num_batched_tokens",
    "max_num_seqs",
    "num_samples", "seed", "shuffle",
    "hint_fraction", "hint_type", "solver_type", "mode",
    "mask_token", "stop_string", "hint_prefix",
    "max_tokens", "timeout", "temperature", "top_p", "top_k", "presence_penalty",
    "max_connections", "epochs", "max_retries", "retry_on_error", "fail_on_error", "display",
    "log_dir",
]


# ---------------------------------------------------------------------------
# Live throughput tracking (prints every 30s during eval)
# ---------------------------------------------------------------------------

_THROUGHPUT_LOCK = threading.Lock()
_THROUGHPUT_START_TS = 0.0
_THROUGHPUT_LAST_PRINT_TS = 0.0
_THROUGHPUT_TOTAL_OUT_TOKENS = 0
_THROUGHPUT_TOTAL_REQUESTS = 0
_THROUGHPUT_LAST_PRINT_TOKENS = 0
_THROUGHPUT_LAST_PRINT_REQUESTS = 0
_THROUGHPUT_PRINT_INTERVAL = 30.0
_LENGTH_WARN_COUNT = 0
_THROUGHPUT_TRACE: list[tuple[float, int, int]] = []
_STEADY_LOWER_FRACTION = 0.2
_STEADY_UPPER_FRACTION = 0.8


def _reset_throughput_tracker():
    global _THROUGHPUT_START_TS, _THROUGHPUT_LAST_PRINT_TS
    global _THROUGHPUT_TOTAL_OUT_TOKENS, _THROUGHPUT_TOTAL_REQUESTS
    global _THROUGHPUT_LAST_PRINT_TOKENS, _THROUGHPUT_LAST_PRINT_REQUESTS
    global _LENGTH_WARN_COUNT
    global _THROUGHPUT_TRACE
    now = time.time()
    _THROUGHPUT_START_TS = now
    _THROUGHPUT_LAST_PRINT_TS = now
    _THROUGHPUT_TOTAL_OUT_TOKENS = 0
    _THROUGHPUT_TOTAL_REQUESTS = 0
    _THROUGHPUT_LAST_PRINT_TOKENS = 0
    _THROUGHPUT_LAST_PRINT_REQUESTS = 0
    _LENGTH_WARN_COUNT = 0
    _THROUGHPUT_TRACE = []


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


def _record_throughput(output) -> None:
    out_tokens = _extract_output_tokens(output)
    now = time.time()

    global _THROUGHPUT_LAST_PRINT_TS, _THROUGHPUT_TOTAL_OUT_TOKENS
    global _THROUGHPUT_TOTAL_REQUESTS, _THROUGHPUT_LAST_PRINT_TOKENS
    global _THROUGHPUT_LAST_PRINT_REQUESTS
    global _THROUGHPUT_TRACE

    with _THROUGHPUT_LOCK:
        _THROUGHPUT_TOTAL_REQUESTS += 1
        if isinstance(out_tokens, int):
            _THROUGHPUT_TOTAL_OUT_TOKENS += out_tokens
        _THROUGHPUT_TRACE.append(
            (
                max(now - _THROUGHPUT_START_TS, 0.0),
                _THROUGHPUT_TOTAL_OUT_TOKENS,
                _THROUGHPUT_TOTAL_REQUESTS,
            )
        )

        since_last = now - _THROUGHPUT_LAST_PRINT_TS
        if since_last < _THROUGHPUT_PRINT_INTERVAL:
            return

        elapsed = max(now - _THROUGHPUT_START_TS, 1e-6)
        delta_tokens = _THROUGHPUT_TOTAL_OUT_TOKENS - _THROUGHPUT_LAST_PRINT_TOKENS
        delta_requests = _THROUGHPUT_TOTAL_REQUESTS - _THROUGHPUT_LAST_PRINT_REQUESTS
        avg_tps = _THROUGHPUT_TOTAL_OUT_TOKENS / elapsed
        win_tps = delta_tokens / max(since_last, 1e-6)

        ts = time.strftime("%m/%d %H:%M:%S")
        elapsed_m, elapsed_s = divmod(int(elapsed), 60)
        print(
            f"[{ts}] [{elapsed_m}m{elapsed_s:02d}s] throughput: "
            f"output_tok/s avg={avg_tps:.1f} window={win_tps:.1f} "
            f"total_out={_THROUGHPUT_TOTAL_OUT_TOKENS:,} "
            f"requests={_THROUGHPUT_TOTAL_REQUESTS} "
            f"window_requests={delta_requests}",
            flush=True,
        )

        _THROUGHPUT_LAST_PRINT_TS = now
        _THROUGHPUT_LAST_PRINT_TOKENS = _THROUGHPUT_TOTAL_OUT_TOKENS
        _THROUGHPUT_LAST_PRINT_REQUESTS = _THROUGHPUT_TOTAL_REQUESTS


def _interpolate_time_at_value(points: list[tuple[float, float]], target: float) -> float | None:
    """Linearly interpolate time where cumulative value reaches target."""
    if not points:
        return None

    prev_t = 0.0
    prev_v = 0.0
    for t, v in points:
        if v >= target:
            if v <= prev_v:
                return t
            ratio = (target - prev_v) / (v - prev_v)
            return prev_t + ratio * (t - prev_t)
        prev_t = t
        prev_v = v
    return points[-1][0]


def _compute_steady_state_throughput(
    trace: list[tuple[float, int, int]],
    total_output_tokens: int,
    total_requests: int,
    lower_fraction: float = _STEADY_LOWER_FRACTION,
    upper_fraction: float = _STEADY_UPPER_FRACTION,
) -> dict[str, float | None]:
    """Estimate steady-state throughput from the middle token-progress band."""
    if not trace or upper_fraction <= lower_fraction:
        return {"output_tokens_per_sec": None, "samples_per_sec": None}

    token_points = [(t, float(tok)) for t, tok, _ in trace]
    req_points = [(t, float(req)) for t, _, req in trace]

    steady_output_tps: float | None = None
    if total_output_tokens > 0:
        lo_tok = lower_fraction * total_output_tokens
        hi_tok = upper_fraction * total_output_tokens
        t_lo = _interpolate_time_at_value(token_points, lo_tok)
        t_hi = _interpolate_time_at_value(token_points, hi_tok)
        if t_lo is not None and t_hi is not None and t_hi > t_lo:
            steady_output_tps = (hi_tok - lo_tok) / (t_hi - t_lo)

    steady_samples_ps: float | None = None
    if total_requests > 0:
        lo_req = lower_fraction * total_requests
        hi_req = upper_fraction * total_requests
        t_lo = _interpolate_time_at_value(req_points, lo_req)
        t_hi = _interpolate_time_at_value(req_points, hi_req)
        if t_lo is not None and t_hi is not None and t_hi > t_lo:
            steady_samples_ps = (hi_req - lo_req) / (t_hi - t_lo)

    return {
        "output_tokens_per_sec": steady_output_tps,
        "samples_per_sec": steady_samples_ps,
    }


def _is_length_stop(output) -> bool:
    if output is None:
        return False
    for attr in ("finish_reason", "stop_reason", "reason"):
        val = getattr(output, attr, None)
        if isinstance(val, str) and val.lower() in ("length", "max_tokens"):
            return True
    choices = getattr(output, "choices", None)
    if isinstance(choices, list) and choices:
        fr = getattr(choices[0], "finish_reason", None)
        if isinstance(fr, str) and fr.lower() == "length":
            return True
    return False


def _warn_length_stop(state: TaskState, max_tokens: int | None):
    global _LENGTH_WARN_COUNT
    _LENGTH_WARN_COUNT += 1
    ts = time.strftime("%m/%d %H:%M:%S")
    sid = getattr(state, "sample_id", None)
    epoch = getattr(state, "epoch", None)
    print(
        f"[{ts}] WARNING: generation hit max_tokens (max_tokens={max_tokens}) "
        f"sample_id={sid!r} epoch={epoch!r} (count={_LENGTH_WARN_COUNT})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config. All keys must be specified in the file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping, got {type(cfg)}")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config file {path} is missing required keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Hint loading and masking (replicated from evals/prefill.py)
# ---------------------------------------------------------------------------

def _truncate_at_stop_string(text: str, stop_string: str) -> str:
    if stop_string not in text:
        return text
    return text[:text.index(stop_string)].strip()


def _split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    return tokens, word_indices


def get_prefill_fraction(text: str, fraction: float, stop_string: str) -> str:
    text = _truncate_at_stop_string(text, stop_string)
    tokens, word_indices = _split_preserving_whitespace(text)
    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text
    last_word_idx = word_indices[num_words - 1]
    return "".join(tokens[:last_word_idx + 1]).strip()


def get_masked_text(text: str, fraction: float, mask_token: str,
                    stop_string: str, seed: int | str | None = None) -> str:
    text = _truncate_at_stop_string(text, stop_string)
    tokens, word_indices = _split_preserving_whitespace(text)
    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask == 0:
        return text
    rng = random.Random(seed) if seed is not None else random
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join(mask_token if i in mask_indices else t for i, t in enumerate(tokens)).strip()


def load_hints(path: str, mode: str, fraction: float,
               stop_string: str) -> dict[str, dict[int, str]]:
    """Load hint data from JSONL. Returns {sample_id: {sample_idx: hint_text}}."""
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
    logger.info(f"Loaded hints for {len(hint_data)} questions from {path} (mode={mode}, fraction={fraction})")
    return hint_data


# ---------------------------------------------------------------------------
# Dataset loading (replicated from environments/aime/aime.py)
# ---------------------------------------------------------------------------

def _record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["ID"],
        input=record["Question"],
        target=str(record["Answer"]),
        metadata={"year": record["Year"], "problem_number": record["Problem Number"]},
    )


def load_aime_dataset(sample_ids: set[str] | None, shuffle: bool):
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


# ---------------------------------------------------------------------------
# Local solvers
# ---------------------------------------------------------------------------

@solver
def _instructions_solver(template: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = template + "\n\n" + state.user_prompt.text
        return state
    return solve


@solver
def _intext_solver(hint_data: dict[str, dict[int, str]], cfg: dict) -> Solver:
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
                    hint_text, fraction=fraction,
                    mask_token=mask_token, stop_string=stop_string, seed=mask_seed,
                )
            state.user_prompt.text = state.user_prompt.text + "\n\n" + prefix + hint_text
        return state
    return solve


@solver
def _prefill_solver(hint_data: dict[str, dict[int, str]], cfg: dict) -> Solver:
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
                    prefill_text, fraction=fraction,
                    mask_token=mask_token, stop_string=stop_string,
                )
            state.messages.append(ChatMessageAssistant(content=prefill_text))
        return state
    return solve


@solver
def _generate_solver(cfg: dict) -> Solver:
    max_tokens = cfg["max_tokens"]

    async def solve(state: TaskState, gen: Generate) -> TaskState:
        continue_message = (
            len(state.messages) > 0
            and isinstance(state.messages[-1], ChatMessageAssistant)
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
            _record_throughput(state.output)
            if _is_length_stop(state.output):
                _warn_length_stop(state, max_tokens)
        return state
    return solve


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------

def build_task(cfg: dict, sample_ids: set[str], hint_data: dict[str, dict[int, str]]) -> Task:
    if cfg["solver_type"] == "intext":
        hint_solver = _intext_solver(hint_data, cfg)
    else:
        hint_solver = _prefill_solver(hint_data, cfg)

    solvers = [
        _instructions_solver(INSTRUCTIONS_TEMPLATE),
        hint_solver,
        _generate_solver(cfg),
    ]

    dataset = load_aime_dataset(sample_ids, shuffle=cfg["shuffle"])
    return Task(dataset=dataset, solver=solvers, scorer=aime_scorer())


# ---------------------------------------------------------------------------
# Subsample and main
# ---------------------------------------------------------------------------

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


def main(cfg: dict, run_name: str = "run"):
    model_name = os.path.basename(cfg["model_path"])
    data_path = REPO_ROOT / "christine_experiments" / "data" / cfg["hint_type"] / "aime.jsonl"

    logger.info("=== Debug Inference Config ===")
    for k, v in sorted(cfg.items()):
        logger.info(f"  {k}: {v}")

    sample_ids = subsample_ids(data_path, cfg["num_samples"], cfg["seed"])
    logger.info(f"Subsampled {len(sample_ids)} AIME problems")

    hint_data = load_hints(
        path=str(data_path),
        mode=cfg["mode"],
        fraction=cfg["hint_fraction"],
        stop_string=cfg["stop_string"],
    )

    task = build_task(cfg, sample_ids, hint_data)

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

        logger.info(f"vLLM server ready on port {server.port}")

        _reset_throughput_tracker()
        t0 = time.time()
        eval_log = inspect_eval(
            task,
            model=f"vllm/{model_name}",
            log_dir=str(log_dir),
            epochs=cfg["epochs"],
            limit=None,
            max_connections=cfg["max_connections"],
            max_retries=cfg["max_retries"],
            display=cfg["display"],
            fail_on_error=cfg["fail_on_error"],
            retry_on_error=cfg["retry_on_error"],
        )
        elapsed = time.time() - t0

    log = eval_log[0]
    if log.status != "success":
        error_msg = f"Eval did not succeed: status={log.status}"
        if log.error:
            error_msg += f", error={log.error.message}"
        logger.error(error_msg)
        sys.exit(1)

    # --- token usage from inspect_ai stats ---
    usage_summary = {}
    if log.stats and log.stats.model_usage:
        for model_key, usage in log.stats.model_usage.items():
            usage_summary[model_key] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage.input_tokens_cache_read is not None:
                usage_summary[model_key]["input_tokens_cache_read"] = usage.input_tokens_cache_read
            if usage.input_tokens_cache_write is not None:
                usage_summary[model_key]["input_tokens_cache_write"] = usage.input_tokens_cache_write

    total_output_tokens = sum(u["output_tokens"] for u in usage_summary.values())
    total_input_tokens = sum(u["input_tokens"] for u in usage_summary.values())
    total_tokens = sum(u["total_tokens"] for u in usage_summary.values())
    n_samples = log.results.completed_samples
    output_tokens_per_sec = total_output_tokens / elapsed if elapsed > 0 else 0
    samples_per_sec = n_samples / elapsed if elapsed > 0 else 0
    with _THROUGHPUT_LOCK:
        trace_snapshot = list(_THROUGHPUT_TRACE)
        total_requests = _THROUGHPUT_TOTAL_REQUESTS
    steady = _compute_steady_state_throughput(
        trace=trace_snapshot,
        total_output_tokens=total_output_tokens,
        total_requests=total_requests,
    )
    steady_output_tps = steady["output_tokens_per_sec"]
    steady_samples_ps = steady["samples_per_sec"]

    logger.info("=== Results ===")
    logger.info(f"  Model: {log.eval.model}")
    logger.info(f"  Total samples: {log.results.total_samples}")
    logger.info(f"  Completed samples: {n_samples}")
    logger.info(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"  Tokens — input: {total_input_tokens:,}  output: {total_output_tokens:,}  total: {total_tokens:,}")
    logger.info(f"  Throughput — overall={output_tokens_per_sec:.1f} output tok/s, {samples_per_sec:.3f} samples/s")
    if steady_output_tps is not None and steady_samples_ps is not None:
        logger.info(
            f"  Throughput (steady {int(_STEADY_LOWER_FRACTION*100)}-{int(_STEADY_UPPER_FRACTION*100)}%) "
            f"— {steady_output_tps:.1f} output tok/s, {steady_samples_ps:.3f} samples/s"
        )
    else:
        logger.info("  Throughput (steady 20-80%) — unavailable (insufficient trace data)")
    for score in log.results.scores:
        logger.info(f"  {score.name}:")
        for metric_name, metric_value in score.metrics.items():
            logger.info(f"    {metric_name}: {metric_value.value}")

    results = {
        "model": log.eval.model,
        "total_samples": log.results.total_samples,
        "completed_samples": n_samples,
        "elapsed_seconds": round(elapsed, 1),
        "throughput": {
            "output_tokens_per_sec": round(output_tokens_per_sec, 1),
            "samples_per_sec": round(samples_per_sec, 4),
            "steady_output_tokens_per_sec": round(steady_output_tps, 1) if steady_output_tps is not None else None,
            "steady_samples_per_sec": round(steady_samples_ps, 4) if steady_samples_ps is not None else None,
            "steady_window_fraction": [_STEADY_LOWER_FRACTION, _STEADY_UPPER_FRACTION],
        },
        "token_usage": usage_summary,
        "token_totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "config": cfg,
    }
    for score in log.results.scores:
        results[score.name] = {
            m: v.value for m, v in score.metrics.items()
        }

    out_path = log_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {out_path}")


class _Tee:
    """Write to both the original stream and a file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug AIME inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    config_stem = Path(args.config).stem
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{config_stem}_{timestamp}"
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / f"{run_name}.log"

    with open(run_log_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            force=True, handlers=[logging.StreamHandler(sys.stdout)])
        try:
            main(cfg, run_name=run_name)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    print(f"Full run log saved to {run_log_path}")
