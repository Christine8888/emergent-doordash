"""Submitit utilities for launching experiments."""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Any
import submitit

from utils.submitit_defaults import SubmitConfig, DEFAULT_CONFIG
from utils.vllm_server import vLLMServer

logger = logging.getLogger(__name__)


def run_single_experiment(
    experiment_class,
    model_path: str,
    tensor_parallel_size: int,
    hint_fraction: float,
    fewshot: int,
    epochs: int,
    results_dir: str,
    config: SubmitConfig,
) -> dict:
    """Run a single experiment configuration.

    This function runs inside a submitit job. It:
    1. Starts vLLM server
    2. Runs experiment
    3. Shuts down vLLM server
    4. Returns results

    Args:
        experiment_class: Experiment class (not instance)
        model_path: Path to model
        tensor_parallel_size: Tensor parallelism size
        hint_fraction: Hint fraction
        fewshot: Number of fewshot examples
        epochs: Number of epochs
        results_dir: Directory to save results
        config: Submit configuration

    Returns:
        Dictionary with experiment results and metadata
    """
    from utils.setup import setup_logging
    setup_logging()

    model_name = os.path.basename(model_path)
    logger.info(f"Starting experiment: {model_name}, fewshot={fewshot}, hint_fraction={hint_fraction}")

    # Get number of GPUs from SLURM
    n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', tensor_parallel_size))
    logger.info(f"Allocated {n_gpus} GPUs by SLURM")

    # Start vLLM server
    with vLLMServer(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        n_gpus=n_gpus,
    ) as server:
        # Create experiment instance
        experiment = experiment_class(
            model_name=model_name,
            vllm_port=server.port,
            timeout=config.timeout,
            max_connections=config.max_connections,
        )

        # Run experiment
        result = experiment.run(
            hint_fraction=hint_fraction,
            fewshot=fewshot,
            epochs=epochs,
            results_dir=results_dir,
        )

        logger.info(f"Experiment complete: {result.get('filename')}")
        return result


def submit_jobs(
    experiment_class,
    models: List[Tuple[str, int]],
    hint_fractions: List[float],
    fewshots: List[int],
    epochs: int,
    results_dir: str,
    config: SubmitConfig,
    skip_existing: bool = True,
) -> List[submitit.Job]:
    """Submit jobs for experiment grid. Internal function."""
    executor = submitit.AutoExecutor(folder=config.submitit_folder)

    jobs = []
    skipped = []

    for model_path, tensor_parallel_size in models:
        model_name = os.path.basename(model_path)
        job_config = config.with_gpus(tensor_parallel_size)

        executor.update_parameters(
            name=f"{config.job_name_prefix}_{model_name}",
            slurm_partition=job_config.partition,
            slurm_account=job_config.account,
            slurm_gpus_per_node=job_config.gpus_per_job,
            slurm_cpus_per_task=job_config.cpus_per_task,
            slurm_mem=f"{job_config.mem_gb}GB",
            slurm_time=job_config.time_hours * 60,
            timeout_min=job_config.time_hours * 60,
            slurm_setup=job_config.setup_commands,
            slurm_srun_args=["--cpu-bind=none"],
        )

        for fewshot in fewshots:
            for hint_fraction in hint_fractions:
                if skip_existing:
                    output_file = experiment_class.get_output_filename(
                        results_dir=results_dir,
                        model_name=model_name,
                        fewshot=fewshot,
                        hint_fraction=hint_fraction,
                    )
                    if os.path.exists(output_file):
                        skipped.append(output_file)
                        continue

                job = executor.submit(
                    run_single_experiment,
                    experiment_class=experiment_class,
                    model_path=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    hint_fraction=hint_fraction,
                    fewshot=fewshot,
                    epochs=epochs,
                    results_dir=results_dir,
                    config=job_config,
                )
                jobs.append(job)
                logger.info(f"Submitted job {job.job_id}: {model_name}, fewshot={fewshot}, hint={hint_fraction}")

    logger.info(f"\nSubmitted {len(jobs)} jobs")
    if skipped:
        logger.info(f"Skipped {len(skipped)} existing outputs")

    return jobs


def launch_experiment(
    experiment_class,
    models: List[Tuple[str, int]],
    hint_fractions: List[float],
    fewshots: List[int] = [0],
    epochs: int = 1,
    results_dir: str = "./results",
    config: Optional[SubmitConfig] = None,
    skip_existing: bool = True,
    wait: bool = True,
    poll_interval: int = 300,
    max_retries: int = 3,
):
    """Launch experiment grid, wait for completion, and resubmit failures.

    Creates one job per (model, fewshot, hint_fraction) combination.
    By default, waits for all jobs and continuously resubmits failures.

    Args:
        experiment_class: Experiment class to run
        models: List of (model_path, tensor_parallel_size) tuples
        hint_fractions: List of hint fractions to sweep
        fewshots: List of fewshot counts to sweep
        epochs: Number of epochs per experiment
        results_dir: Directory to save results
        config: Submit configuration (uses DEFAULT_CONFIG if None)
        skip_existing: Skip jobs where output file already exists
        wait: If True (default), wait for jobs and resubmit failures
        poll_interval: Seconds between status checks (default 300 = 5 min)
        max_retries: Max times to resubmit a failed job (default 3)

    Logs are saved to: {config.submitit_folder}/ (default: ./submitit_logs/)
        - {job_id}_submission.sh  - SLURM submission script
        - {job_id}_0_log.out      - stdout
        - {job_id}_0_log.err      - stderr
    """
    import time

    config = config or DEFAULT_CONFIG

    # Validate hint fractions
    for frac in hint_fractions:
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"hint_fraction must be in [0.0, 1.0], got {frac}")

    logger.info(f"Submitit logs: {config.submitit_folder}/")
    logger.info(f"Results dir: {results_dir}/")

    # Initial submission
    jobs = submit_jobs(
        experiment_class=experiment_class,
        models=models,
        hint_fractions=hint_fractions,
        fewshots=fewshots,
        epochs=epochs,
        results_dir=results_dir,
        config=config,
        skip_existing=skip_existing,
    )

    if not jobs:
        logger.info("No jobs to run (all outputs exist)")
        return []

    if not wait:
        return jobs

    # Track all jobs and retry counts
    all_jobs = list(jobs)
    retry_counts = {job.job_id: 0 for job in jobs}

    logger.info(f"\nWaiting for {len(all_jobs)} jobs (polling every {poll_interval}s)...")

    while True:
        time.sleep(poll_interval)

        status_map = check_job_status(all_jobs)

        done = len(status_map.get('DONE', []))
        failed_jobs = status_map.get('FAILED', []) + status_map.get('TIMEOUT', [])
        active = len(status_map.get('PENDING', [])) + len(status_map.get('RUNNING', []))

        logger.info(f"Status: {done} done, {len(failed_jobs)} failed, {active} active")

        # Resubmit failed jobs (if under retry limit)
        for job in failed_jobs:
            original_id = job.job_id
            if retry_counts.get(original_id, 0) < max_retries:
                try:
                    new_job = job.executor.submit(job.fn, *job.args, **job.kwargs)
                    retry_counts[new_job.job_id] = retry_counts.get(original_id, 0) + 1
                    all_jobs.append(new_job)
                    all_jobs.remove(job)
                    logger.info(f"Resubmitted {original_id} as {new_job.job_id} (retry {retry_counts[new_job.job_id]}/{max_retries})")
                except Exception as e:
                    logger.error(f"Failed to resubmit {original_id}: {e}")
            else:
                logger.warning(f"Job {original_id} exceeded max retries ({max_retries})")

        # Check if done
        status_map = check_job_status(all_jobs)
        active = len(status_map.get('PENDING', [])) + len(status_map.get('RUNNING', []))

        if active == 0:
            done = len(status_map.get('DONE', []))
            failed = len(status_map.get('FAILED', [])) + len(status_map.get('TIMEOUT', []))
            logger.info(f"\n✓ All jobs complete! Done: {done}, Failed: {failed}")
            break

    return all_jobs


def check_job_status(jobs: List[submitit.Job]) -> dict:
    """Check status of jobs.

    Args:
        jobs: List of submitit jobs

    Returns:
        Dictionary with status counts and lists of jobs by status
    """
    status_map = {
        'PENDING': [],
        'RUNNING': [],
        'DONE': [],
        'FAILED': [],
        'TIMEOUT': [],
        'CANCELLED': [],
    }

    for job in jobs:
        try:
            # Force refresh by invalidating cache
            job._state = None
            state = job.state
            status_map.get(state, []).append(job)
        except Exception as e:
            logger.warning(f"Could not get status for job {job.job_id}: {e}")
            status_map.setdefault('UNKNOWN', []).append(job)

    # Print summary
    logger.info("\nJob Status Summary:")
    for status, job_list in status_map.items():
        if job_list:
            logger.info(f"  {status}: {len(job_list)}")

    return status_map


def resubmit_failed_jobs(
    jobs: List[submitit.Job],
    max_retries: Optional[int] = None,
    config: Optional[SubmitConfig] = None,
) -> List[submitit.Job]:
    """Resubmit failed and timed-out jobs.

    Args:
        jobs: List of submitit jobs to check
        max_retries: Maximum retries (uses config.max_retries if None)
        config: Submit configuration

    Returns:
        List of newly submitted jobs
    """
    config = config or DEFAULT_CONFIG
    max_retries = max_retries if max_retries is not None else config.max_retries

    status_map = check_job_status(jobs)
    failed_jobs = status_map.get('FAILED', []) + status_map.get('TIMEOUT', [])

    if not failed_jobs:
        logger.info("No failed jobs to resubmit")
        return []

    logger.info(f"\nFound {len(failed_jobs)} failed jobs")

    new_jobs = []
    for job in failed_jobs:
        # Check retry count (stored in job metadata if we track it)
        # For now, just resubmit once
        try:
            # Get original function and args
            func = job.fn
            args = job.args
            kwargs = job.kwargs

            # Resubmit
            new_job = job.executor.submit(func, *args, **kwargs)
            new_jobs.append(new_job)
            logger.info(f"Resubmitted job {job.job_id} as {new_job.job_id}")

        except Exception as e:
            logger.error(f"Failed to resubmit job {job.job_id}: {e}")

    logger.info(f"Resubmitted {len(new_jobs)} jobs")
    return new_jobs


def wait_for_jobs(jobs: List[submitit.Job], check_interval: int = 60):
    """Wait for all jobs to complete, with periodic status updates.

    Args:
        jobs: List of submitit jobs
        check_interval: Seconds between status checks
    """
    import time

    logger.info(f"Waiting for {len(jobs)} jobs to complete...")
    logger.info(f"Checking status every {check_interval}s")

    while True:
        status_map = check_job_status(jobs)

        done = len(status_map.get('DONE', []))
        failed = len(status_map.get('FAILED', [])) + len(status_map.get('TIMEOUT', []))
        active = len(status_map.get('PENDING', [])) + len(status_map.get('RUNNING', []))

        if active == 0:
            logger.info(f"\nAll jobs complete! Done: {done}, Failed: {failed}")
            break

        time.sleep(check_interval)


# --- Baseline eval support ---

def run_baseline_eval(
    eval_name: str,
    model_path: str,
    tensor_parallel_size: int,
    results_dir: str,
    config: SubmitConfig,
    epochs: int = 1,
    limit: int | None = None,
) -> dict:
    """Run a single baseline eval (no hints).

    This function runs inside a submitit job.
    """
    from experiments.runner import run_eval_with_vllm
    from experiments.registry import get_eval

    model_name = os.path.basename(model_path)
    output_dir = Path(results_dir) / eval_name / model_name
    output_file = str(output_dir / f"{eval_name}.json")

    return run_eval_with_vllm(
        task_fn=get_eval(eval_name),
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        output_file=output_file,
        config=config,
        epochs=epochs,
        limit=limit,
    )


def get_baseline_output_file(results_dir: str, eval_name: str, model_name: str) -> str:
    """Get output filename for baseline eval."""
    return str(Path(results_dir) / eval_name / model_name / f"{eval_name}.json")


def submit_baseline_jobs(
    eval_names: List[str],
    models: List[Tuple[str, int]],
    results_dir: str,
    config: SubmitConfig,
    epochs: int = 1,
    limit: int | None = None,
    skip_existing: bool = True,
) -> List[submitit.Job]:
    """Submit baseline eval jobs."""
    executor = submitit.AutoExecutor(folder=config.submitit_folder)

    jobs = []
    skipped = []

    for eval_name in eval_names:
        for model_path, tensor_parallel_size in models:
            model_name = os.path.basename(model_path)

            if skip_existing:
                output_file = get_baseline_output_file(results_dir, eval_name, model_name)
                if os.path.exists(output_file):
                    skipped.append(output_file)
                    continue

            job_config = config.with_gpus(tensor_parallel_size)

            executor.update_parameters(
                name=f"baseline_{eval_name}_{model_name}",
                slurm_partition=job_config.partition,
                slurm_account=job_config.account,
                slurm_gpus_per_node=job_config.gpus_per_job,
                slurm_cpus_per_task=job_config.cpus_per_task,
                slurm_mem=f"{job_config.mem_gb}GB",
                slurm_time=job_config.time_hours * 60,
                timeout_min=job_config.time_hours * 60,
                slurm_setup=job_config.setup_commands,
                slurm_srun_args=["--cpu-bind=none"],
            )

            job = executor.submit(
                run_baseline_eval,
                eval_name=eval_name,
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                results_dir=results_dir,
                config=job_config,
                epochs=epochs,
                limit=limit,
            )
            jobs.append(job)
            logger.info(f"Submitted job {job.job_id}: {eval_name} / {model_name}")

    logger.info(f"\nSubmitted {len(jobs)} jobs")
    if skipped:
        logger.info(f"Skipped {len(skipped)} existing outputs")

    return jobs


def launch_baseline(
    eval_names: List[str],
    models: List[Tuple[str, int]],
    results_dir: str = "./baseline_results",
    config: Optional[SubmitConfig] = None,
    epochs: int = 1,
    limit: int | None = None,
    skip_existing: bool = True,
    wait: bool = True,
    poll_interval: int = 300,
    max_retries: int = 3,
):
    """Launch baseline evals with retry logic.

    Args:
        eval_names: List of eval names to run
        models: List of (model_path, tensor_parallel_size) tuples
        results_dir: Directory to save results
        config: Submit configuration
        epochs: Number of epochs
        limit: Optional sample limit
        skip_existing: Skip if output exists
        wait: Wait for completion with retries
        poll_interval: Seconds between status checks
        max_retries: Max retries per job
    """
    import time

    config = config or DEFAULT_CONFIG

    logger.info(f"Launching baseline evals: {eval_names}")
    logger.info(f"Results dir: {results_dir}/")

    jobs = submit_baseline_jobs(
        eval_names=eval_names,
        models=models,
        results_dir=results_dir,
        config=config,
        epochs=epochs,
        limit=limit,
        skip_existing=skip_existing,
    )

    if not jobs:
        logger.info("No jobs to run (all outputs exist)")
        return []

    if not wait:
        return jobs

    # Wait with retry logic
    all_jobs = list(jobs)
    retry_counts = {job.job_id: 0 for job in jobs}

    logger.info(f"\nWaiting for {len(all_jobs)} jobs (polling every {poll_interval}s)...")

    while True:
        time.sleep(poll_interval)

        status_map = check_job_status(all_jobs)
        done = len(status_map.get('DONE', []))
        failed_jobs = status_map.get('FAILED', []) + status_map.get('TIMEOUT', [])
        active = len(status_map.get('PENDING', [])) + len(status_map.get('RUNNING', []))

        logger.info(f"Status: {done} done, {len(failed_jobs)} failed, {active} active")

        for job in failed_jobs:
            original_id = job.job_id
            if retry_counts.get(original_id, 0) < max_retries:
                try:
                    new_job = job.executor.submit(job.fn, *job.args, **job.kwargs)
                    retry_counts[new_job.job_id] = retry_counts.get(original_id, 0) + 1
                    all_jobs.append(new_job)
                    all_jobs.remove(job)
                    logger.info(f"Resubmitted {original_id} as {new_job.job_id}")
                except Exception as e:
                    logger.error(f"Failed to resubmit {original_id}: {e}")

        status_map = check_job_status(all_jobs)
        active = len(status_map.get('PENDING', [])) + len(status_map.get('RUNNING', []))

        if active == 0:
            done = len(status_map.get('DONE', []))
            failed = len(status_map.get('FAILED', [])) + len(status_map.get('TIMEOUT', []))
            logger.info(f"\n✓ All jobs complete! Done: {done}, Failed: {failed}")
            break

    return all_jobs
