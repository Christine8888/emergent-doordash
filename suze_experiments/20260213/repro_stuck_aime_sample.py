#!/usr/bin/env python3
"""Reproduce and debug AIME sample timeouts with detailed per-epoch telemetry.

Example:
python suze_experiments/20260213/repro_stuck_aime_sample.py \
  --model Qwen/Qwen3-4B \
  --sample_id 2008-II-13 \
  --hint_fraction 0.0 \
  --epochs 10 \
  --attempts 4 \
  --max_connections 48 \
  --timeout 3600 \
  --inspect_http_debug \
  --openai_retry_debug
"""
# ^ job submitit_logs/14806371_0_log.out 
# also internal server error in submitit_logs/14813845_0_log.out

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_IMPORT_ERROR: Exception | None = None
try:
    from inspect_ai import eval as inspect_eval

    from environments.aime.aime import DEFAULT_INSTRUCTIONS, aime
    from evals.prefill import PrefillConfig
    from evals.solvers import generate, instructions, intext, prefill
    from experiments.runner import setup_vllm_env
    from utils.model_config import GEMMA_MODELS, LLAMA_MODELS, QWEN25_MODELS, QWEN3_MODELS
    from utils.setup import setup_inspect_logging, setup_logging, setup_openai_retry_debug_logging
    from utils.vllm_server import vLLMServer
except Exception as exc:
    _IMPORT_ERROR = exc
    inspect_eval = None
    DEFAULT_INSTRUCTIONS = None
    aime = None
    PrefillConfig = None
    generate = instructions = intext = prefill = None
    setup_vllm_env = None
    GEMMA_MODELS = LLAMA_MODELS = QWEN25_MODELS = QWEN3_MODELS = []
    vLLMServer = None

    def setup_inspect_logging(*_args, **_kwargs):  # type: ignore[no-redef]
        return None

    def setup_openai_retry_debug_logging(*_args, **_kwargs):  # type: ignore[no-redef]
        return None

    import logging

    def setup_logging(level=logging.INFO):  # type: ignore[no-redef]
        logging.basicConfig(level=level, format="%(message)s")
        return logging.getLogger(__name__)

logger = setup_logging()

ALL_MODELS = QWEN3_MODELS + QWEN25_MODELS + GEMMA_MODELS + LLAMA_MODELS


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_CONSOLE_TEE_FILE: io.TextIOWrapper | None = None


class _TeeTextIO(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def writable(self) -> bool:
        return True


def _install_console_tee(log_path: Path) -> None:
    global _CONSOLE_TEE_FILE
    if _CONSOLE_TEE_FILE is not None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _CONSOLE_TEE_FILE = open(log_path, "a", encoding="utf-8", buffering=1)
    _CONSOLE_TEE_FILE.write(f"\n=== repro run started {_now()} ===\n")
    _CONSOLE_TEE_FILE.flush()
    sys.stdout = _TeeTextIO(sys.stdout, _CONSOLE_TEE_FILE)
    sys.stderr = _TeeTextIO(sys.stderr, _CONSOLE_TEE_FILE)

    def _close_console_tee() -> None:
        global _CONSOLE_TEE_FILE
        if _CONSOLE_TEE_FILE is not None:
            _CONSOLE_TEE_FILE.flush()
            _CONSOLE_TEE_FILE.close()
            _CONSOLE_TEE_FILE = None

    atexit.register(_close_console_tee)


def _openai_timeout_defaults() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from openai._constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT

        out["openai_default_max_retries"] = DEFAULT_MAX_RETRIES
        out["openai_default_timeout_repr"] = repr(DEFAULT_TIMEOUT)
        out["openai_default_timeout_connect_sec"] = getattr(DEFAULT_TIMEOUT, "connect", None)
        out["openai_default_timeout_read_sec"] = getattr(DEFAULT_TIMEOUT, "read", None)
        out["openai_default_timeout_write_sec"] = getattr(DEFAULT_TIMEOUT, "write", None)
        out["openai_default_timeout_pool_sec"] = getattr(DEFAULT_TIMEOUT, "pool", None)
    except Exception as exc:
        out["openai_default_timeout_error"] = str(exc)
    return out


def _resolve_model_path(model: str) -> str:
    matches = [m.path for m in ALL_MODELS if m.path == model or os.path.basename(m.path) == model]
    if not matches:
        return model
    if len(matches) > 1:
        raise ValueError(f"Ambiguous model {model!r}; matches: {sorted(set(matches))}")
    return matches[0]


def _resolve_tp(model_path: str, tp_override: int | None) -> int:
    if tp_override is not None:
        return tp_override
    matches = [m for m in ALL_MODELS if m.path == model_path or os.path.basename(m.path) == os.path.basename(model_path)]
    if not matches:
        return 1
    return matches[0].tp


def _build_task(
    *,
    sample_id: str,
    hint_type: str,
    solver_type: str,
    mode: str,
    hint_fraction: float,
    timeout: int,
) -> Any:
    data_path = str(REPO_ROOT / "christine_experiments" / "data" / hint_type / "aime.jsonl")
    cfg = PrefillConfig(path=data_path, fraction=hint_fraction, mode=mode)
    hint_solver = intext(cfg, prefix="Here is part of a hint that may be helpful to your solution:\n") if solver_type == "intext" else prefill(cfg)
    solver = [
        instructions(DEFAULT_INSTRUCTIONS),
        hint_solver,
        generate(timeout=timeout),
    ]
    return aime(sample_ids={sample_id}, solver=solver)


def _to_dict_maybe(x: Any) -> dict[str, Any] | None:
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            out = x.model_dump()
            if isinstance(out, dict):
                return out
        except Exception:
            return None
    return None


def _extract_usage(sample: Any, served_model_name: str) -> dict[str, Any]:
    out_tokens = None
    in_tokens = None

    output = getattr(sample, "output", None)
    output_dict = _to_dict_maybe(output)
    usage = None
    if output_dict is not None:
        usage = output_dict.get("usage")
    else:
        usage = getattr(output, "usage", None)
    usage_dict = usage if isinstance(usage, dict) else _to_dict_maybe(usage)
    if usage_dict is not None:
        out_tokens = usage_dict.get("output_tokens")
        in_tokens = usage_dict.get("input_tokens")

    model_usage = getattr(sample, "model_usage", None)
    if isinstance(model_usage, dict):
        preferred_keys = [f"vllm/{served_model_name}", served_model_name]
        for key in preferred_keys + list(model_usage.keys()):
            mu = model_usage.get(key)
            if isinstance(mu, dict):
                if out_tokens is None:
                    out_tokens = mu.get("output_tokens")
                if in_tokens is None:
                    in_tokens = mu.get("input_tokens")
                if out_tokens is not None and in_tokens is not None:
                    break

    return {"output_tokens": out_tokens, "input_tokens": in_tokens}


def _extract_finish_reason(sample: Any) -> str | None:
    output = getattr(sample, "output", None)
    output_dict = _to_dict_maybe(output)
    if output_dict is None:
        return None
    choices = output_dict.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return first.get("stop_reason") or first.get("finish_reason")
    return output_dict.get("stop_reason") or output_dict.get("finish_reason")


def _extract_error_text(sample: Any) -> str | None:
    err = getattr(sample, "error", None)
    if err is None:
        return None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err)


def _summarize_attempt(eval_log0: Any, sample_id: str, epochs: int, served_model_name: str) -> dict[str, Any]:
    rows = []
    for sample in (getattr(eval_log0, "samples", None) or []):
        sid = str(getattr(sample, "id", ""))
        if sid != sample_id:
            continue
        usage = _extract_usage(sample, served_model_name=served_model_name)
        row = {
            "sample_id": sid,
            "epoch": getattr(sample, "epoch", None),
            "error": _extract_error_text(sample),
            "total_time": getattr(sample, "total_time", None),
            "working_time": getattr(sample, "working_time", None),
            "retries": getattr(sample, "error_retries", None),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "finish_reason": _extract_finish_reason(sample),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"]))
    ok_epochs = sum(1 for r in rows if not r["error"])
    failed_epochs = sum(1 for r in rows if r["error"])
    unique_epochs = sorted({r["epoch"] for r in rows if isinstance(r["epoch"], int)})

    eval_path = None
    for attr in ("location", "log_file", "path"):
        val = getattr(eval_log0, attr, None)
        if isinstance(val, str):
            eval_path = val
            break

    return {
        "ok_epochs": ok_epochs,
        "failed_epochs": failed_epochs,
        "expected_epochs": epochs,
        "observed_epoch_ids": unique_epochs,
        "complete": ok_epochs == epochs and failed_epochs == 0,
        "eval_path": eval_path,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce/debug stuck AIME samples with verbose telemetry.")
    parser.add_argument("--model", type=str, required=True, help="HF model path or basename (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size override (default: inferred).")
    parser.add_argument("--sample_id", type=str, default="2014-I-15")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=4, help="How many full requeue-style attempts to run.")
    parser.add_argument("--hint_type", choices=["solution"], default="solution")
    parser.add_argument("--solver_type", choices=["intext", "prefill"], default="intext")
    parser.add_argument("--mode", choices=["masked", "sequential"], default="masked")
    parser.add_argument("--hint_fraction", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=3600, help="Per-request generation timeout (seconds).")
    parser.add_argument("--max_connections", type=int, default=48)
    parser.add_argument("--max_retries", type=int, default=1, help="HTTP-level retries passed to inspect eval.")
    parser.add_argument("--retry_on_error", type=int, default=1, help="Sample-level retries passed to inspect eval.")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--server_port", type=int, default=None, help="Use an already-running vLLM server on this port.")
    parser.add_argument("--n_gpus", type=int, default=None, help="When launching vLLM, total GPUs for DP calc.")
    parser.add_argument("--inspect_http_debug", action="store_true")
    parser.add_argument("--openai_retry_debug", action="store_true")
    parser.add_argument("--stop_on_success", action="store_true", help="Stop early if one attempt fully succeeds.")
    parser.add_argument("--log_dir", type=str, default="debug_logs/stuck_aime_repro")
    parser.add_argument("--console_log_file", type=str, default=None, help="Optional path for full stdout/stderr tee log.")
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependencies for this script. "
            "Activate the experiment env first (e.g. source scripts/setup_env_suze.sh). "
            f"Original import error: {_IMPORT_ERROR}"
        )

    model_path = _resolve_model_path(args.model)
    served_model_name = os.path.basename(model_path)
    tp = _resolve_tp(model_path, args.tp)

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_json = out_dir / f"repro_{served_model_name}_{args.sample_id}_{run_stamp}.json"
    console_log = (
        Path(args.console_log_file)
        if args.console_log_file
        else out_dir / f"repro_{served_model_name}_{args.sample_id}_{run_stamp}.console.log"
    )
    _install_console_tee(console_log)

    if args.inspect_http_debug:
        setup_inspect_logging(level="http")
    if args.openai_retry_debug:
        setup_openai_retry_debug_logging(enabled=True)
    logger.info("Console tee log: %s", console_log)

    base_meta = {
        "started_at": _now(),
        "repo_root": str(REPO_ROOT),
        "model_path": model_path,
        "served_model_name": served_model_name,
        "tp": tp,
        "sample_id": args.sample_id,
        "epochs": args.epochs,
        "attempts": args.attempts,
        "hint_type": args.hint_type,
        "solver_type": args.solver_type,
        "mode": args.mode,
        "hint_fraction": args.hint_fraction,
        "timeout": args.timeout,
        "max_connections": args.max_connections,
        "max_retries": args.max_retries,
        "retry_on_error": args.retry_on_error,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "server_port": args.server_port,
        "n_gpus": args.n_gpus,
        "env_OPENAI_TIMEOUT": os.environ.get("OPENAI_TIMEOUT"),
        "console_log_file": str(console_log),
    }
    base_meta.update(_openai_timeout_defaults())
    logger.info("Debug run config:\n%s", json.dumps(base_meta, indent=2, sort_keys=True))

    all_attempts: list[dict[str, Any]] = []

    def _run_attempts(port: int) -> None:
        setup_vllm_env(
            port,
            served_model_name,
            openai_client_timeout=args.timeout,
        )
        os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
        logger.info(
            "Runtime env: VLLM_BASE_URL=%s OPENAI_TIMEOUT=%s VLLM_MAX_MODEL_LEN=%s",
            os.environ.get("VLLM_BASE_URL"),
            os.environ.get("OPENAI_TIMEOUT"),
            os.environ.get("VLLM_MAX_MODEL_LEN"),
        )

        for attempt_idx in range(1, args.attempts + 1):
            logger.info("[%s] Attempt %d/%d starting", _now(), attempt_idx, args.attempts)
            task = _build_task(
                sample_id=args.sample_id,
                hint_type=args.hint_type,
                solver_type=args.solver_type,
                mode=args.mode,
                hint_fraction=args.hint_fraction,
                timeout=args.timeout,
            )
            attempt_started = time.time()
            eval_log = inspect_eval(
                task,
                model=f"vllm/{served_model_name}",
                log_dir=str(out_dir),
                epochs=args.epochs,
                limit=None,
                max_connections=args.max_connections,
                max_retries=args.max_retries,
                display="plain",
                fail_on_error=False,
                retry_on_error=args.retry_on_error,
                metadata={
                    "debug_repro": True,
                    "sample_id": args.sample_id,
                    "attempt": attempt_idx,
                    "timeout": args.timeout,
                    "max_connections": args.max_connections,
                },
            )
            elapsed = time.time() - attempt_started
            summary = _summarize_attempt(
                eval_log[0],
                sample_id=args.sample_id,
                epochs=args.epochs,
                served_model_name=served_model_name,
            )
            summary["attempt"] = attempt_idx
            summary["attempt_wall_time_sec"] = round(elapsed, 3)
            all_attempts.append(summary)

            logger.info(
                "[%s] Attempt %d done: ok=%d failed=%d complete=%s wall=%.1fs eval_path=%s",
                _now(),
                attempt_idx,
                summary["ok_epochs"],
                summary["failed_epochs"],
                summary["complete"],
                elapsed,
                summary["eval_path"],
            )
            for row in summary["rows"]:
                logger.info(
                    "  epoch=%s err=%s out_tok=%s in_tok=%s total_time=%s working=%s retries=%s finish=%s",
                    row["epoch"],
                    "Y" if row["error"] else "N",
                    row["output_tokens"],
                    row["input_tokens"],
                    row["total_time"],
                    row["working_time"],
                    row["retries"],
                    row["finish_reason"],
                )
                if row["error"]:
                    logger.info("    error=%s", row["error"])

            if args.stop_on_success and summary["complete"]:
                logger.info("Stopping early due to --stop_on_success")
                break

    if args.server_port is not None:
        _run_attempts(args.server_port)
    else:
        with vLLMServer(
            model_path=model_path,
            tensor_parallel_size=tp,
            n_gpus=args.n_gpus,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
        ) as server:
            _run_attempts(server.port)

    output = {
        "config": base_meta,
        "finished_at": _now(),
        "attempts": all_attempts,
    }
    with open(run_json, "w") as f:
        json.dump(output, f, indent=2, sort_keys=False)
    logger.info("Wrote debug summary: %s", run_json)


if __name__ == "__main__":
    main()
