"""Generic eval runner - core infrastructure for running any Inspect eval."""

import os
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from inspect_ai import eval as inspect_eval
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs, compute_pass_at_k

logger = logging.getLogger(__name__)


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

    Returns:
        Dictionary with results and status
    """
    output_path = Path(output_file)

    if output_path.exists():
        logger.info(f"Output already exists: {output_file}")
        return {"filename": output_file, "status": "skipped"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running eval: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    if limit:
        logger.info(f"  Limit: {limit}")

    eval_log = inspect_eval(
        task,
        model=f"vllm/{model_name}",
        log_dir=str(output_path.parent),
        epochs=epochs,
        limit=limit,
        max_connections=max_connections,
        display="plain",
        fail_on_error=False,
        retry_on_error=10,
        metadata=metadata or {},
    )

    results = extract_scores_from_log(eval_log[0])

    if epochs > 1:
        # Try to compute bootstrap metrics - need to find the scorer name
        try:
            scorer_names = list(results.get("scores", {}).keys())
            if scorer_names:
                bootstrap_metric = {'scorer': scorer_names[0], 'metric': 'accuracy'}
                results["manual_bootstrap"] = compute_bootstrap_over_epochs(eval_log[0], bootstrap_metric)
                results["pass_at_k"] = compute_pass_at_k(eval_log[0], bootstrap_metric)
        except Exception as e:
            logger.warning(f"Could not compute bootstrap metrics: {e}")

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
        logger.info(f"Output already exists: {output_file}")
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
        )
