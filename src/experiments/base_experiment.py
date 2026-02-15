"""Base experiment class for all experiments."""

import os
import logging
import sys
import time
import contextlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from inspect_ai import eval
from inspect_ai.dataset import Sample
from utils.eval_utils import get_valid_problem_ids
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs, compute_pass_at_k
from utils.setup import setup_logging, setup_inspect_logging
from experiments.runner import setup_vllm_env
import json

logger = setup_logging()


class _TimestampStepsStream:
    """Wrap a text stream and prefix Inspect progress 'Steps:' lines with a timestamp + ETA."""

    _steps_re = re.compile(r"^Steps:\s*(\d+)\s*/\s*(\d+)\b")

    def __init__(self, stream, *, line_prefix: str = "Steps:"):
        self._stream = stream
        self._line_prefix = line_prefix
        self._buf = ""
        self._t0 = time.time()
        self._last_t = None
        self._last_steps = None
        self._ema_rate = None  # steps/sec

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
        """Prefix timestamp and append ETA based on observed step rate."""
        ts = time.strftime("%m/%d %H:%M:%S")

        m = self._steps_re.match(line)
        if not m:
            return f"[{ts}] {line}"

        try:
            steps = int(m.group(1))
            total = int(m.group(2))
            now = time.time()

            if self._last_t is not None and self._last_steps is not None and steps >= self._last_steps:
                dt = max(now - self._last_t, 1e-6)
                dsteps = steps - self._last_steps
                inst_rate = dsteps / dt if dsteps > 0 else 0.0
                # Exponential moving average for stability
                alpha = 0.2
                if self._ema_rate is None:
                    self._ema_rate = inst_rate
                else:
                    self._ema_rate = (alpha * inst_rate) + ((1 - alpha) * self._ema_rate)

            self._last_t = now
            self._last_steps = steps

            remaining = max(total - steps, 0)
            rate = self._ema_rate if self._ema_rate and self._ema_rate > 0 else None

            if rate is None:
                return f"[{ts}] {line} | ETA: ?"

            eta_seconds = int(remaining / rate) if remaining > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            elapsed_seconds = int(now - self._t0)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

            return f"[{ts}] {line} | elapsed: {elapsed_str} | ETA: {eta_str}"
        except Exception:
            return f"[{ts}] {line}"

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _timestamp_steps_stdout(enabled: bool = True):
    """Prefix Inspect progress lines (Steps: ...) with wallclock timestamps."""
    if not enabled:
        yield
        return
    old_stdout = sys.stdout
    try:
        sys.stdout = _TimestampStepsStream(old_stdout)
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = old_stdout


def init_inspect_debug(debug: bool = False, log_file: str | None = None):
    """Initialize Inspect debug logging if enabled.

    Args:
        debug: If True, sets log level to "http" for detailed request logging
        log_file: Optional path to write logs to file
    """
    if debug:
        setup_inspect_logging(level="http", log_file=log_file)
        logger.info("Inspect debug logging enabled (level=http)")


class Experiment(ABC):
    """Base class for experiments.

    Subclasses must define:
    - name: Experiment name (e.g., "cot_intext")
    - eval_name: Eval dataset name (e.g., "gpqa")
    - data_path: Path to hint data JSONL file
    - build_task(): Method to construct Inspect task

    Example:
        class MyExperiment(Experiment):
            name = "my_exp"
            eval_name = "gpqa"
            data_path = "data/hints.jsonl"

            def build_task(self, hint_fraction, sample_ids):
                # Build and return Inspect task
                return my_task(sample_ids=sample_ids, solver=my_solver)
    """

    # Subclasses must define these
    name: str = NotImplemented
    eval_name: str = NotImplemented
    data_path: str = NotImplemented

    def __init__(
        self,
        model_name: str,
        vllm_port: int,
        timeout: int = 600,
        max_connections: int = 32,
    ):
        """Initialize experiment.

        Args:
            model_name: Name of model being evaluated
            vllm_port: Port where vLLM server is running
            timeout: Timeout for eval tasks
            max_connections: Max concurrent connections
        """
        self.model_name = model_name
        self.vllm_port = vllm_port
        self.timeout = timeout
        self.max_connections = max_connections

        setup_vllm_env(vllm_port)

    @abstractmethod
    def build_task(self, hint_fraction: float, sample_ids: set[str]):
        """Build the Inspect task for this experiment.

        Args:
            hint_fraction: Fraction of hint to provide
            sample_ids: Set of sample IDs to evaluate on

        Returns:
            Inspect Task object
        """
        pass

    @classmethod
    def get_output_filename(
        cls,
        results_dir: str,
        model_name: str,
        fewshot: int,
        hint_fraction: float,
    ) -> str:
        """Get output filename for this configuration.

        Args:
            results_dir: Results directory
            model_name: Model name
            fewshot: Number of fewshot examples
            hint_fraction: Hint fraction

        Returns:
            Full path to output file
        """
        output_dir = Path(results_dir) / cls.eval_name / cls.name / f"{fewshot}shot" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{cls.eval_name}_{cls.name}_{fewshot}shot_{hint_fraction}.json"
        return str(output_dir / filename)

    def run(
        self,
        hint_fraction: float,
        fewshot: int,
        epochs: int,
        results_dir: str,
        limit: Optional[int] = None,
    ) -> dict:
        """Run the experiment.

        Args:
            hint_fraction: Fraction of hint to provide
            fewshot: Number of fewshot examples
            epochs: Number of epochs
            results_dir: Directory to save results
            limit: Optional limit on number of samples

        Returns:
            Dictionary with results and metadata
        """
        # Get output filename
        output_file = self.get_output_filename(
            results_dir=results_dir,
            model_name=self.model_name,
            fewshot=fewshot,
            hint_fraction=hint_fraction,
        )

        # Check if output already exists
        if os.path.exists(output_file):
            logger.info(f"Output already exists: {output_file}")
            return {"filename": output_file, "status": "skipped"}

        valid_samples = get_valid_problem_ids([self.data_path])
        if valid_samples is None:
            raise ValueError(f"Failed to load sample IDs from {self.data_path}")

        sample_ids = set(valid_samples.keys())
        logger.info(f"Running {self.name} on {len(sample_ids)} samples")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Fewshot: {fewshot}")
        logger.info(f"  Hint fraction: {hint_fraction}")
        logger.info(f"  Epochs: {epochs}")

        # Build task
        task = self.build_task(
            hint_fraction=hint_fraction,
            sample_ids=sample_ids if not limit else list(sample_ids)[:limit]
        )

        # Run evaluation - Inspect logs go in same dir as results JSON
        output_dir = Path(output_file).parent

        with _timestamp_steps_stdout(enabled=True):
            eval_log = eval(
                task,
                model=f"vllm/{self.model_name}",
                log_dir=str(output_dir),
                epochs=epochs,
                limit=limit,
                max_connections=self.max_connections,
                max_retries=10,  # HTTP-level retries (prevents infinite retry loops)
                display="plain",
                fail_on_error=False,
                retry_on_error=10,  # sample-level retries
                metadata={
                    "timeout": self.timeout,
                    "hint_fraction": hint_fraction,
                    "fewshot": fewshot,
                    "data_path": self.data_path,
                    "solver_name": self.name,
                },
            )

        # Extract results
        results = extract_scores_from_log(eval_log[0])

        # Compute bootstrap metrics if multiple epochs
        if epochs > 1:
            scorer_name = f"{self.eval_name}_scorer"
            bootstrap_metric = {'scorer': scorer_name, 'metric': 'accuracy'}
            results["manual_bootstrap"] = compute_bootstrap_over_epochs(eval_log[0], bootstrap_metric)
            results["pass_at_k"] = compute_pass_at_k(eval_log[0], bootstrap_metric)

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return {
            "filename": output_file,
            "status": "completed",
            "results": results
        }
