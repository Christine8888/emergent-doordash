"""Generic eval runner - core infrastructure for running any Inspect eval."""

import os
import json
import logging
import sys
import time
import contextlib
import re
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from inspect_ai import eval as inspect_eval
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs, compute_pass_at_k

logger = logging.getLogger(__name__)


class _TimestampStepsStream:
    """Wrap a text stream and prefix Inspect progress 'Steps:' lines with a timestamp + ETA."""

    _steps_re = re.compile(r"^Steps:\s*(\d+)\s*/\s*(\d+)\b")
    _samples_segment_re = re.compile(r"\s*\|\s*Samples:\s*\d+\s*/\s*\d+\s*")

    def __init__(self, stream, *, line_prefix: str = "Steps:", label: str = ""):
        self._stream = stream
        self._line_prefix = line_prefix
        self._label = label
        self._buf = ""
        self._t_first = None
        self._history = deque(maxlen=40)  # (t, steps)

    def write(self, s: str):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.startswith(self._line_prefix):
                line = self._format_steps_line(line)
            self._stream.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            line = self._buf
            self._buf = ""
            if line.startswith(self._line_prefix):
                line = self._format_steps_line(line)
            self._stream.write(line)
        self._stream.flush()

    def _format_steps_line(self, line: str) -> str:
        """Prefix timestamp and append ETA based on a rolling-window rate."""
        ts = time.strftime("%m/%d %H:%M:%S")

        # "Samples: x/y" is redundant with Steps for our evals; strip it.
        line = self._samples_segment_re.sub("", line)

        m = self._steps_re.match(line)
        if not m:
            suffix = f" | {self._label}" if self._label else ""
            return f"[{ts}] {line}{suffix}"

        try:
            steps = int(m.group(1))
            total = int(m.group(2))
            now = time.time()

            if self._t_first is None:
                self._t_first = now
            self._history.append((now, steps))

            remaining = max(total - steps, 0)

            # Build a stable rate from a longer window.
            # Prefer a window that's at least 120s old; fall back to oldest point.
            t_old, s_old = None, None
            for t_i, s_i in self._history:
                if now - t_i >= 120:
                    t_old, s_old = t_i, s_i
                    break
            if t_old is None:
                t_old, s_old = self._history[0]

            dt = max(now - t_old, 1e-6)
            dsteps = max(steps - s_old, 0)
            rate = (dsteps / dt) if dsteps > 0 else None  # steps/sec

            elapsed_seconds = int(now - (self._t_first or now))
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

            # Don't show ETA until we have enough progress to make it meaningful.
            if steps < 50 or rate is None or rate <= 0:
                suffix = f" | {self._label}" if self._label else ""
                return f"[{ts}] {line} | elapsed: {elapsed_str} | ETA: ?{suffix}"

            eta_seconds = int(remaining / rate) if remaining > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            suffix = f" | {self._label}" if self._label else ""
            return f"[{ts}] {line} | elapsed: {elapsed_str} | ETA: {eta_str}{suffix}"
        except Exception:
            suffix = f" | {self._label}" if self._label else ""
            return f"[{ts}] {line}{suffix}"

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _timestamp_steps_stdout(enabled: bool = True, label: str = ""):
    """Prefix Inspect progress lines (Steps: ...) with wallclock timestamps."""
    if not enabled:
        yield
        return
    old_stdout = sys.stdout
    try:
        sys.stdout = _TimestampStepsStream(old_stdout, label=label)
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = old_stdout


def setup_vllm_env(port: int, model_name: str = None):
    """Set vLLM environment variables."""
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["VLLM_API_KEY"] = "local"
    if model_name:
        os.environ["INSPECT_EVAL_MODEL"] = f"vllm/{model_name}"


def run_eval(
    task,
    model_name: str,
    output_file: str,
    epochs: int = 1,
    limit: Optional[int] = None,
    max_connections: int = 32,
    metadata: Optional[dict] = None,
    *,
    max_tokens: int,
    label: str = "",
) -> dict:
    """Run an Inspect eval and save results.

    Args:
        task: Inspect Task object
        model_name: Name of model (used for vllm/{model_name})
        output_file: Path to save results JSON
        epochs: Number of epochs
        limit: Optional sample limit
        max_connections: Max concurrent connections
        metadata: Optional metadata to include
        max_tokens: Max tokens to generate per sample (generation cap)

    Returns:
        Dictionary with results and status
    """
    output_path = Path(output_file)

    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        total = existing.get("total_samples")
        completed = existing.get("completed_samples")
        if isinstance(total, int) and isinstance(completed, int) and total > 0 and completed == total:
            logger.info(f"Output already complete: {output_file}")
            return {"filename": output_file, "status": "skipped"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running eval: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    if limit:
        logger.info(f"  Limit: {limit}")

    with _timestamp_steps_stdout(enabled=True, label=label or model_name):
        eval_log = inspect_eval(
            task,
            model=f"vllm/{model_name}",
            log_dir=str(output_path.parent),
            epochs=epochs,
            limit=limit,
            max_connections=max_connections,
            max_retries=10,  # HTTP-level retries (prevents infinite retry loops)
            display="plain",
            fail_on_error=False,
            retry_on_error=10,  # sample-level retries
            metadata=metadata or {},
            max_tokens=max_tokens,
        )

    results = extract_scores_from_log(eval_log[0])

    if epochs > 1:
        # Compute bootstrap metrics using the first scorer in the results dict.
        scorer_candidates = [
            k for k in results.keys()
            if k not in ("model", "total_samples", "completed_samples", "metadata")
            and k.endswith("_scorer")
        ]
        if scorer_candidates:
            bootstrap_metric = {"scorer": scorer_candidates[0], "metric": "accuracy"}
            results["manual_bootstrap"] = compute_bootstrap_over_epochs(eval_log[0], bootstrap_metric)
            results["pass_at_k"] = compute_pass_at_k(eval_log[0], bootstrap_metric)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    return {"filename": output_file, "status": "completed", "results": results}


def run_eval_with_vllm(
    task_fn: Callable,
    model_path: str,
    tensor_parallel_size: int,
    output_file: str,
    config,
    epochs: int = 1,
    limit: Optional[int] = None,
    task_kwargs: Optional[dict] = None,
    *,
    max_tokens: int,
) -> dict:
    """Run eval with vLLM server management.

    This is the main entry point for running evals in SLURM jobs.

    Args:
        task_fn: Function that returns an Inspect Task
        model_path: HuggingFace model path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        output_file: Path to save results JSON
        config: SubmitConfig with vLLM settings
        epochs: Number of epochs
        limit: Optional sample limit
        task_kwargs: Optional kwargs to pass to task_fn

    Returns:
        Dictionary with results and status
    """
    from utils.setup import setup_logging
    from utils.vllm_server import vLLMServer
    from utils.submitit_utils import GPUMonitor

    setup_logging()

    model_name = os.path.basename(model_path)

    if Path(output_file).exists():
        with open(output_file, "r") as f:
            existing = json.load(f)
        total = existing.get("total_samples")
        completed = existing.get("completed_samples")
        if isinstance(total, int) and isinstance(completed, int) and total > 0 and completed == total:
            logger.info(f"Output already complete: {output_file}")
            return {"filename": output_file, "status": "skipped"}

    n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', tensor_parallel_size))
    logger.info(f"Allocated {n_gpus} GPUs by SLURM")

    with GPUMonitor(), vLLMServer(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        n_gpus=n_gpus,
    ) as server:
        setup_vllm_env(server.port, model_name)
        os.environ["VLLM_MAX_MODEL_LEN"] = str(config.max_model_len)

        # Create task after env is set (some evals like niah call get_model() at creation)
        task_kwargs = task_kwargs or {}
        task = task_fn(**task_kwargs)

        return run_eval(
            task=task,
            model_name=model_name,
            output_file=output_file,
            epochs=epochs,
            limit=limit,
            max_connections=config.max_connections,
            max_tokens=max_tokens,
        )
