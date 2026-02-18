"""Base experiment class for all experiments."""

import os
import logging
import math
import sys
import time
import contextlib
import re
from collections import deque
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict

from inspect_ai import eval
from inspect_ai.dataset import Sample
from utils.eval_utils import get_valid_problem_ids
from utils.inspect_utils import (
    extract_scores_from_log,
    compute_bootstrap_over_epochs,
    compute_pass_at_k,
    compute_accuracy_stderr_from_correctness,
    compute_bootstrap_over_epochs_from_correctness,
    compute_pass_at_k_from_correctness,
)
from utils.setup import setup_logging, setup_inspect_logging
from experiments.runner import setup_vllm_env
import json

logger = setup_logging()

_CHECKPOINT_VERSION = 1


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to `path` (write temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read JSON from {path}: {e}")
        return None


def _ckpt_path_for_output(output_path: Path) -> Path:
    # output.json -> output.ckpt.json
    if output_path.suffix != ".json":
        return output_path.with_name(output_path.name + ".ckpt.json")
    return output_path.with_suffix(".ckpt.json")


def _new_checkpoint_state(*, meta: dict[str, Any], total_instances: int) -> dict[str, Any]:
    return {
        "version": _CHECKPOINT_VERSION,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "meta": meta,
        # total/completed in *instances* (samples * epochs), mirroring Inspect behavior.
        "total_samples": int(total_instances),
        "completed_samples": 0,
        "per_sample_epoch_correct": {},  # sample_id -> [0/1,...] length=epochs
    }


def _validate_checkpoint_state(state: Any, *, expected_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Return validated checkpoint dict or None if incompatible/corrupt."""
    if not isinstance(state, dict):
        return None
    if state.get("version") != _CHECKPOINT_VERSION:
        return None
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return None

    # Strict resume: only resume if key meta fields match exactly.
    # This avoids accidentally reusing a checkpoint from a different run configuration.
    keys = ("eval_name", "experiment_name", "model_name", "fewshot", "hint_fraction", "epochs", "data_path")
    for k in keys:
        if meta.get(k) != expected_meta.get(k):
            logger.warning(f"Checkpoint meta mismatch on {k}: ckpt={meta.get(k)!r} expected={expected_meta.get(k)!r}")
            return None

    per = state.get("per_sample_epoch_correct")
    if not isinstance(per, dict):
        return None
    # Ensure values are list[int]-ish; if malformed, treat as incompatible.
    for sid, arr in per.items():
        if not isinstance(sid, str) or not isinstance(arr, list):
            return None
    return state


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
        output_path = Path(output_file)
        ckpt_path = _ckpt_path_for_output(output_path)

        # Check if output already exists and is complete.
        # (Some older runs may have written partial JSON; treat those as resumable.)
        if output_path.exists():
            existing = _load_json(output_path)
            if isinstance(existing, dict):
                total = existing.get("total_samples")
                completed = existing.get("completed_samples")
                if isinstance(total, int) and isinstance(completed, int) and total > 0 and completed == total:
                    logger.info(f"Output already complete: {output_file}")
                    return {"filename": output_file, "status": "skipped"}
            logger.warning(f"Output exists but incomplete; will resume and overwrite: {output_file}")

        valid_samples = get_valid_problem_ids([self.data_path])
        if valid_samples is None:
            raise ValueError(f"Failed to load sample IDs from {self.data_path}")

        all_sample_ids = sorted(valid_samples.keys())
        if limit:
            all_sample_ids = all_sample_ids[:limit]
        sample_ids_set = set(all_sample_ids)

        logger.info(f"Running {self.name} on {len(all_sample_ids)} samples")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Fewshot: {fewshot}")
        logger.info(f"  Hint fraction: {hint_fraction}")
        logger.info(f"  Epochs: {epochs}")

        output_dir = output_path.parent
        scorer_name = f"{self.eval_name}_scorer"
        metadata = {
            "timeout": self.timeout,
            "hint_fraction": hint_fraction,
            "fewshot": fewshot,
            "data_path": self.data_path,
            "solver_name": self.name,
        }

        # Optional escape hatch to preserve previous single-shot behavior.
        if os.environ.get("EXPERIMENT_DISABLE_CHECKPOINT") == "1":
            task = self.build_task(hint_fraction=hint_fraction, sample_ids=sample_ids_set)
            _label = f"{self.model_name} | hint={hint_fraction:.2f}"
            with _timestamp_steps_stdout(enabled=True, label=_label):
                eval_log = eval(
                    task,
                    model=f"vllm/{self.model_name}",
                    log_dir=str(output_dir),
                    epochs=epochs,
                    limit=None,
                    max_connections=self.max_connections,
                    max_retries=10,  # HTTP-level retries (prevents infinite retry loops)
                    display="plain",
                    fail_on_error=False,
                    retry_on_error=10,  # sample-level retries
                    metadata=metadata,
                )

            results = extract_scores_from_log(eval_log[0])
            if epochs > 1:
                bootstrap_metric = {"scorer": scorer_name, "metric": "accuracy"}
                results["manual_bootstrap"] = compute_bootstrap_over_epochs(eval_log[0], bootstrap_metric)
                results["pass_at_k"] = compute_pass_at_k(eval_log[0], bootstrap_metric)

            _atomic_write_json(output_path, results)
            logger.info(f"Results saved to {output_file}")
            return {"filename": output_file, "status": "completed", "results": results}

        # ---- Checkpointed, resumable run ----
        # EXPERIMENT_CHECKPOINT_CHUNK_INSTANCES controls how many (sample × epoch)
        # instances to process per chunk.  chunk_size (in samples) is derived as
        # ceil(chunk_instances / epochs) so checkpoint frequency stays consistent
        # regardless of epoch count.  Default: 100 instances per chunk.
        _DEFAULT_CHUNK_INSTANCES = 100
        chunk_instances_env = os.environ.get("EXPERIMENT_CHECKPOINT_CHUNK_INSTANCES")
        try:
            chunk_instances = max(int(chunk_instances_env), 1) if chunk_instances_env is not None else _DEFAULT_CHUNK_INSTANCES
        except Exception:
            logger.warning(f"Invalid EXPERIMENT_CHECKPOINT_CHUNK_INSTANCES={chunk_instances_env!r}; using {_DEFAULT_CHUNK_INSTANCES}")
            chunk_instances = _DEFAULT_CHUNK_INSTANCES

        expected_meta = {
            "eval_name": self.eval_name,
            "experiment_name": self.name,
            "model_name": self.model_name,
            "fewshot": fewshot,
            "hint_fraction": hint_fraction,
            "epochs": epochs,
            "data_path": self.data_path,
        }
        total_instances = len(all_sample_ids) * int(epochs)

        # Convert instance budget to a sample count, capped at total samples.
        chunk_size = max(1, math.ceil(chunk_instances / int(epochs)))
        chunk_size = min(chunk_size, len(all_sample_ids))

        state = _load_json(ckpt_path)
        state = _validate_checkpoint_state(state, expected_meta=expected_meta) if state is not None else None
        if state is None:
            state = _new_checkpoint_state(meta=expected_meta, total_instances=total_instances)

        state["total_samples"] = int(total_instances)
        per: dict[str, list[int]] = dict(state.get("per_sample_epoch_correct", {}) or {})

        # Sanitize checkpoint entries (drop malformed / wrong-length / no-longer-needed).
        sanitized: dict[str, list[int]] = {}
        for sid, arr in per.items():
            if sid not in sample_ids_set:
                continue
            if not isinstance(arr, list) or len(arr) != epochs:
                continue
            try:
                sanitized[sid] = [1 if int(v) else 0 for v in arr]
            except Exception:
                continue
        per = sanitized

        completed_ids = set(per.keys())
        remaining_ids = [sid for sid in all_sample_ids if sid not in completed_ids]
        logger.info(f"Checkpoint progress: {len(completed_ids)}/{len(all_sample_ids)} samples complete; remaining={len(remaining_ids)}")
        logger.info(f"Checkpoint file: {ckpt_path}")

        def _extract_chunk_correctness(eval_log0, *, chunk_id_set: set[str]) -> dict[str, list[int]]:
            per_chunk: dict[str, list[int]] = defaultdict(list)
            for sample in getattr(eval_log0, "samples", []) or []:
                sid = getattr(sample, "id", None)
                if not isinstance(sid, str) or sid not in chunk_id_set:
                    continue
                scores = getattr(sample, "scores", None)
                if not isinstance(scores, dict) or not scores:
                    continue
                score_obj = scores.get(scorer_name) or next(iter(scores.values()))
                val = getattr(score_obj, "value", None)
                correct = 1 if (val == "C" or val is True or val == 1) else 0
                per_chunk[sid].append(correct)

            out: dict[str, list[int]] = {}
            for sid, arr in per_chunk.items():
                if len(arr) != epochs:
                    raise RuntimeError(f"Unexpected epoch count for sample {sid}: got {len(arr)} expected {epochs}")
                out[sid] = list(arr)

            missing = chunk_id_set - set(out.keys())
            if missing:
                raise RuntimeError(f"Missing scores for {len(missing)} samples in chunk (e.g. {sorted(list(missing))[:5]})")
            return out

        # Run remaining samples in chunks, checkpointing after each chunk.
        while remaining_ids:
            chunk = remaining_ids[:chunk_size]
            remaining_ids = remaining_ids[chunk_size:]
            chunk_set = set(chunk)

            logger.info(f"Running chunk: samples={len(chunk)} instances={len(chunk) * int(epochs)} remaining_after={len(remaining_ids)}")
            task = self.build_task(hint_fraction=hint_fraction, sample_ids=chunk_set)

            _label = f"{self.model_name} | hint={hint_fraction:.2f}"
            with _timestamp_steps_stdout(enabled=True, label=_label):
                eval_log = eval(
                    task,
                    model=f"vllm/{self.model_name}",
                    log_dir=str(output_dir),
                    epochs=epochs,
                    limit=None,
                    max_connections=self.max_connections,
                    max_retries=10,  # HTTP-level retries (prevents infinite retry loops)
                    display="plain",
                    fail_on_error=False,
                    retry_on_error=10,  # sample-level retries
                    metadata=metadata,
                )

            chunk_correct = _extract_chunk_correctness(eval_log[0], chunk_id_set=chunk_set)
            per.update(chunk_correct)

            state["updated_at"] = _now_iso()
            state["per_sample_epoch_correct"] = per
            state["completed_sample_ids"] = sorted(per.keys())
            state["completed_samples"] = int(len(per) * epochs)
            _atomic_write_json(ckpt_path, state)

            msg = (
                f"[{_now_iso()}] Checkpoint saved: "
                f"completed={len(per)}/{len(all_sample_ids)} samples "
                f"-> {ckpt_path}"
            )
            logger.info(msg)
            print(msg, flush=True)

        # Compute final results from checkpointed correctness data.
        per_ordered = {sid: per[sid] for sid in all_sample_ids if sid in per}
        if len(per_ordered) != len(all_sample_ids):
            missing = [sid for sid in all_sample_ids if sid not in per_ordered]
            raise RuntimeError(f"Checkpoint missing {len(missing)} samples at end (e.g. {missing[:5]})")

        acc_stderr = compute_accuracy_stderr_from_correctness(per_ordered)
        results: dict[str, Any] = {
            "model": f"vllm/{self.model_name}",
            "total_samples": int(total_instances),
            "completed_samples": int(total_instances),
            "metadata": metadata,
            scorer_name: {
                "accuracy": acc_stderr["accuracy"],
                "stderr": acc_stderr["stderr"],
                "scorer": scorer_name,
            },
        }

        if epochs > 1:
            results["manual_bootstrap"] = compute_bootstrap_over_epochs_from_correctness(per_ordered)
            results["pass_at_k"] = compute_pass_at_k_from_correctness(per_ordered)

        _atomic_write_json(output_path, results)
        logger.info(f"Results saved to {output_file}")
        return {"filename": output_file, "status": "completed", "results": results}
