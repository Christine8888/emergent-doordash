"""Submitit utilities for launching experiments."""

import os
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable
import submitit

from utils.submitit_defaults import SubmitConfig, DEFAULT_CONFIG
from utils.vllm_server import vLLMServer

logger = logging.getLogger(__name__)

# sacct state -> internal state
STATE_MAP = {
    'COMPLETED': 'DONE', 'PENDING': 'PENDING', 'RUNNING': 'RUNNING',
    'FAILED': 'FAILED', 'TIMEOUT': 'TIMEOUT', 'CANCELLED': 'CANCELLED',
    'OUT_OF_MEMORY': 'FAILED', 'NODE_FAIL': 'FAILED',
}


def check_job_status(jobs: list[submitit.Job]) -> dict[str, list]:
    """Check job status via sacct."""
    status_map = {s: [] for s in ['PENDING', 'RUNNING', 'DONE', 'FAILED', 'TIMEOUT', 'CANCELLED']}
    if not jobs:
        return status_map

    job_lookup = {str(j.job_id): j for j in jobs}
    result = subprocess.run(
        ["sacct", "-j", ",".join(job_lookup.keys()), "-n", "-o", "JobID,State", "--parsable2"],
        capture_output=True, text=True, timeout=30
    )
    seen = set()
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jid, state = parts[0].split(".")[0], parts[1].split()[0]
        if jid in job_lookup and jid not in seen:
            seen.add(jid)
            status_map[STATE_MAP.get(state, 'FAILED')].append(job_lookup[jid])

    for jid, job in job_lookup.items():
        if jid not in seen:
            status_map['PENDING'].append(job)

    logger.info("Job Status: " + ", ".join(f"{k}={len(v)}" for k, v in status_map.items() if v))
    return status_map


def _configure_executor(executor: submitit.AutoExecutor, config: SubmitConfig, name: str):
    """Configure executor with slurm parameters."""
    params = dict(
        name=name,
        slurm_partition=config.partition,
        slurm_account=config.account,
        slurm_gpus_per_node=config.gpus_per_job,
        slurm_cpus_per_task=config.cpus_per_task,
        slurm_mem=f"{config.mem_gb}GB",
        slurm_time=config.time_hours * 60,
        timeout_min=config.time_hours * 60,
        slurm_setup=config.setup_commands,
        slurm_srun_args=["--cpu-bind=none"],
        slurm_exclude=config.exclude_nodes,
    )
    if config.qos:
        params["slurm_qos"] = config.qos
    executor.update_parameters(**params)


def _wait_with_retries(jobs: list[submitit.Job], poll_interval: int, max_retries: int):
    """Wait for jobs, resubmitting failures up to max_retries."""
    retry_counts = {job.job_id: 0 for job in jobs}
    all_jobs = list(jobs)

    while True:
        time.sleep(poll_interval)
        status_map = check_job_status(all_jobs)

        for job in status_map.get('FAILED', []) + status_map.get('TIMEOUT', []):
            if retry_counts.get(job.job_id, 0) < max_retries:
                new_job = job.executor.submit(job.fn, *job.args, **job.kwargs)
                retry_counts[new_job.job_id] = retry_counts.get(job.job_id, 0) + 1
                all_jobs.append(new_job)
                all_jobs.remove(job)
                logger.info(f"Resubmitted {job.job_id} as {new_job.job_id}")

        active = len(status_map.get('PENDING', [])) + len(status_map.get('RUNNING', []))
        if active == 0:
            done = len(status_map.get('DONE', []))
            failed = len(status_map.get('FAILED', [])) + len(status_map.get('TIMEOUT', []))
            logger.info(f"Complete: {done} done, {failed} failed")
            return all_jobs


def run_single_experiment(
    experiment_class, model_path: str, tensor_parallel_size: int,
    hint_fraction: float, fewshot: int, epochs: int, results_dir: str, config: SubmitConfig,
) -> dict:
    """Run single experiment inside submitit job."""
    from utils.setup import setup_logging
    setup_logging()

    model_name = os.path.basename(model_path)
    n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', tensor_parallel_size))

    with vLLMServer(
        model_path=model_path, tensor_parallel_size=tensor_parallel_size,
        max_model_len=config.max_model_len, gpu_memory_utilization=config.gpu_memory_utilization, n_gpus=n_gpus,
    ) as server:
        experiment = experiment_class(
            model_name=model_name, vllm_port=server.port,
            timeout=config.timeout, max_connections=config.max_connections,
        )
        return experiment.run(hint_fraction=hint_fraction, fewshot=fewshot, epochs=epochs, results_dir=results_dir)


def run_baseline_eval(
    eval_name: str, model_path: str, tensor_parallel_size: int,
    results_dir: str, config: SubmitConfig, epochs: int = 1, limit: int | None = None,
) -> dict:
    """Run single baseline eval inside submitit job."""
    from experiments.runner import run_eval_with_vllm
    from experiments.registry import get_eval

    model_name = os.path.basename(model_path)
    output_file = str(Path(results_dir) / eval_name / model_name / f"{eval_name}.json")
    return run_eval_with_vllm(
        task_fn=get_eval(eval_name), model_path=model_path, tensor_parallel_size=tensor_parallel_size,
        output_file=output_file, config=config, epochs=epochs, limit=limit,
    )


def launch_experiment(
    experiment_class, models: list[tuple[str, int]], hint_fractions: list[float],
    fewshots: list[int] = [0], epochs: int = 1, results_dir: str = "./results",
    config: SubmitConfig | None = None, skip_existing: bool = True,
    wait: bool = True, poll_interval: int = 300, max_retries: int = 3,
):
    """Launch experiment grid with retry logic."""
    config = config or DEFAULT_CONFIG
    executor = submitit.AutoExecutor(folder=config.submitit_folder)
    jobs = []

    for model_path, tp in models:
        model_name = os.path.basename(model_path)
        job_config = config.with_gpus(tp)
        _configure_executor(executor, job_config, f"{config.job_name_prefix}_{model_name}")

        for fewshot in fewshots:
            for hint_fraction in hint_fractions:
                if skip_existing:
                    output = experiment_class.get_output_filename(
                        results_dir=results_dir, model_name=model_name, fewshot=fewshot, hint_fraction=hint_fraction)
                    if os.path.exists(output):
                        continue

                job = executor.submit(
                    run_single_experiment, experiment_class=experiment_class, model_path=model_path,
                    tensor_parallel_size=tp, hint_fraction=hint_fraction, fewshot=fewshot,
                    epochs=epochs, results_dir=results_dir, config=job_config,
                )
                jobs.append(job)
                logger.info(f"Submitted {job.job_id}: {model_name}, fewshot={fewshot}, hint={hint_fraction}")

    logger.info(f"Submitted {len(jobs)} jobs")
    if not jobs or not wait:
        return jobs
    return _wait_with_retries(jobs, poll_interval, max_retries)


def launch_baseline(
    eval_names: list[str], models: list[tuple[str, int]], results_dir: str = "./baseline_results",
    config: SubmitConfig | None = None, epochs: int = 1, limit: int | None = None,
    skip_existing: bool = True, wait: bool = True, poll_interval: int = 300, max_retries: int = 3,
):
    """Launch baseline evals with retry logic."""
    config = config or DEFAULT_CONFIG
    executor = submitit.AutoExecutor(folder=config.submitit_folder)
    jobs = []

    for eval_name in eval_names:
        for model_path, tp in models:
            model_name = os.path.basename(model_path)

            if skip_existing:
                output = str(Path(results_dir) / eval_name / model_name / f"{eval_name}.json")
                if os.path.exists(output):
                    continue

            job_config = config.with_gpus(tp)
            _configure_executor(executor, job_config, f"baseline_{eval_name}_{model_name}")

            job = executor.submit(
                run_baseline_eval, eval_name=eval_name, model_path=model_path, tensor_parallel_size=tp,
                results_dir=results_dir, config=job_config, epochs=epochs, limit=limit,
            )
            jobs.append(job)
            logger.info(f"Submitted {job.job_id}: {eval_name} / {model_name}")

    logger.info(f"Submitted {len(jobs)} jobs")
    if not jobs or not wait:
        return jobs
    return _wait_with_retries(jobs, poll_interval, max_retries)
