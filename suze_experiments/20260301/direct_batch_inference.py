"""Direct vLLM batch inference for AIME evaluation.

Uses vllm.LLM offline batching (no HTTP server) to measure throughput
upper-bound. Comparable to debug_inference.py but bypasses inspect_ai
and the OpenAI-compatible server entirely.

Usage:
    python suze_experiments/20260301/direct_batch_inference.py \
        --config suze_experiments/20260301/configs/batch_default.yaml

    python suze_experiments/20260301/direct_batch_inference.py \
        --config suze_experiments/20260301/configs/batch_high_max_model_len.yaml

"""

from pathlib import Path
import sys
import os
import json
import random
import re
import argparse
import asyncio
import logging
import threading
import time
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environments.math.utils import extract_answer, grade_math_answer

logger = logging.getLogger(__name__)

AIME_DATASET_PATH = "di-zhang-fdu/AIME_1983_2024"

INSTRUCTIONS_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your '
    'response should be of the form "ANSWER: $ANSWER" (without quotes) '
    'where $ANSWER is the answer to the problem.'
)

REQUIRED_KEYS = [
    "model_path",
    "max_model_len", "gpu_memory_utilization", "dtype",
    "max_num_seqs",
    "num_samples", "seed", "shuffle",
    "hint_fraction", "hint_type", "solver_type", "mode",
    "mask_token", "stop_string", "hint_prefix",
    "max_tokens", "temperature", "top_p", "top_k", "presence_penalty",
    "log_dir",
]

_STEADY_LOWER_FRACTION = 0.2
_STEADY_UPPER_FRACTION = 0.8


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


def _extract_finished_time(output) -> float | None:
    """Best-effort extraction of vLLM per-request finished timestamp."""
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    for attr in ("finished_time", "finished_ts", "completion_time"):
        val = getattr(metrics, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    return None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping, got {type(cfg)}")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config file {path} is missing required keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Hint loading and masking (same as debug_inference.py)
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
# Dataset loading (raw HuggingFace, no inspect_ai)
# ---------------------------------------------------------------------------

def load_aime_raw(num_samples: int, seed: int, shuffle: bool,
                  hint_jsonl: Path) -> list[dict]:
    """Load AIME records and subsample using IDs that have hints."""
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
            records.append({
                "id": row["ID"],
                "question": row["Question"],
                "answer": str(row["Answer"]),
                "year": row["Year"],
                "problem_number": row["Problem Number"],
            })

    if shuffle:
        rng.shuffle(records)

    logger.info(f"Loaded {len(records)} AIME problems (selected from {len(all_ids)} with hints)")
    return records


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompts(records: list[dict], cfg: dict,
                  hint_data: dict[str, dict[int, str]]) -> list[dict]:
    """Build chat-format prompts, returning list of {id, answer, messages}."""
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

        prompts.append({
            "id": sid,
            "answer": rec["answer"],
            "messages": messages,
        })
    return prompts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(prompts: list[dict], completions: list[str]) -> dict:
    """Grade each completion against the target answer."""
    correct = 0
    total = len(prompts)
    per_sample = []

    for prompt_info, completion in zip(prompts, completions):
        extracted = extract_answer(completion)
        is_correct = asyncio.run(
            grade_math_answer(extracted, prompt_info["answer"],
                              exact_match=True, use_sympy=True)
        )
        per_sample.append({
            "id": prompt_info["id"],
            "target": prompt_info["answer"],
            "extracted": extracted,
            "correct": is_correct,
        })
        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict, run_name: str = "run"):
    logger.info("=== Direct Batch Inference Config ===")
    for k, v in sorted(cfg.items()):
        logger.info(f"  {k}: {v}")

    hint_jsonl = REPO_ROOT / "christine_experiments" / "data" / cfg["hint_type"] / "aime.jsonl"
    records = load_aime_raw(cfg["num_samples"], cfg["seed"], cfg["shuffle"], hint_jsonl)

    hint_data = load_hints(
        path=str(hint_jsonl),
        mode=cfg["mode"],
        fraction=cfg["hint_fraction"],
        stop_string=cfg["stop_string"],
    )

    prompts = build_prompts(records, cfg, hint_data)
    logger.info(f"Built {len(prompts)} prompts")

    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- vLLM offline inference ---
    from vllm import LLM, SamplingParams

    logger.info("Loading vLLM model (offline mode)...")
    t_load = time.time()
    llm_kwargs: dict[str, Any] = {}
    if cfg["max_num_seqs"] is not None:
        llm_kwargs["max_num_seqs"] = cfg["max_num_seqs"]

    llm = LLM(
        model=cfg["model_path"],
        dtype=cfg["dtype"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        seed=cfg["seed"],
        **llm_kwargs,
    )
    logger.info(f"Model loaded in {time.time() - t_load:.1f}s")

    sampling_params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        presence_penalty=cfg["presence_penalty"],
        max_tokens=cfg["max_tokens"],
    )

    conversations = [p["messages"] for p in prompts]

    logger.info(f"Starting batch inference on {len(conversations)} prompts...")
    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    elapsed = time.time() - t0

    completions = [out.outputs[0].text for out in outputs]
    total_input_tokens = sum(len(out.prompt_token_ids) for out in outputs)
    total_output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    total_tokens = total_input_tokens + total_output_tokens
    output_tps = total_output_tokens / elapsed if elapsed > 0 else 0
    samples_per_sec = len(prompts) / elapsed if elapsed > 0 else 0
    steady_output_tps: float | None = None
    steady_samples_per_sec: float | None = None

    # Build a completion-time trace to estimate steady-state throughput (20%-80%).
    finish_records: list[tuple[float, int]] = []
    for out in outputs:
        finished_time = _extract_finished_time(out)
        out_tokens = len(out.outputs[0].token_ids)
        if finished_time is not None:
            finish_records.append((finished_time, out_tokens))

    if len(finish_records) == len(outputs) and finish_records:
        finish_records.sort(key=lambda x: x[0])
        t0_finish = finish_records[0][0]
        cumulative_tokens = 0
        cumulative_requests = 0
        trace: list[tuple[float, int, int]] = []
        for finished_time, out_tokens in finish_records:
            cumulative_requests += 1
            cumulative_tokens += out_tokens
            trace.append((finished_time - t0_finish, cumulative_tokens, cumulative_requests))
        steady = _compute_steady_state_throughput(
            trace=trace,
            total_output_tokens=total_output_tokens,
            total_requests=len(outputs),
        )
        steady_output_tps = steady["output_tokens_per_sec"]
        steady_samples_per_sec = steady["samples_per_sec"]

    max_tokens_hits = sum(
        1 for out in outputs if len(out.outputs[0].token_ids) >= cfg["max_tokens"]
    )

    logger.info("=== Inference Complete ===")
    logger.info(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"  Tokens — input: {total_input_tokens:,}  output: {total_output_tokens:,}  total: {total_tokens:,}")
    logger.info(f"  Throughput — overall={output_tps:.1f} output tok/s, {samples_per_sec:.3f} samples/s")
    if steady_output_tps is not None and steady_samples_per_sec is not None:
        logger.info(
            f"  Throughput (steady {int(_STEADY_LOWER_FRACTION*100)}-{int(_STEADY_UPPER_FRACTION*100)}%) "
            f"— {steady_output_tps:.1f} output tok/s, {steady_samples_per_sec:.3f} samples/s"
        )
    else:
        logger.info("  Throughput (steady 20-80%) — unavailable (missing per-request finish metrics)")
    logger.info(f"  Max tokens hits: {max_tokens_hits}/{len(prompts)}")

    # --- Scoring ---
    logger.info("Scoring...")
    scores = score_results(prompts, completions)
    logger.info(f"  Accuracy: {scores['accuracy']:.3f} ({scores['correct']}/{scores['total']})")

    # --- Save results ---
    results = {
        "model": cfg["model_path"],
        "num_samples": len(prompts),
        "elapsed_seconds": round(elapsed, 1),
        "throughput": {
            "output_tokens_per_sec": round(output_tps, 1),
            "samples_per_sec": round(samples_per_sec, 4),
            "steady_output_tokens_per_sec": round(steady_output_tps, 1) if steady_output_tps is not None else None,
            "steady_samples_per_sec": round(steady_samples_per_sec, 4) if steady_samples_per_sec is not None else None,
            "steady_window_fraction": [_STEADY_LOWER_FRACTION, _STEADY_UPPER_FRACTION],
        },
        "token_totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "max_tokens_hits": max_tokens_hits,
        "accuracy": scores["accuracy"],
        "correct": scores["correct"],
        "total": scores["total"],
        "per_sample": scores["per_sample"],
        "config": cfg,
    }

    out_path = log_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {out_path}")


# ---------------------------------------------------------------------------
# Tee for run logging
# ---------------------------------------------------------------------------

class _Tee:
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
    parser = argparse.ArgumentParser(description="Direct vLLM batch AIME inference")
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
