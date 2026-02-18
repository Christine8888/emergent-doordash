"""Submitit utilities for launching experiments."""

import os
import json
import logging
import subprocess
import time
import threading
from pathlib import Path
from typing import Callable
import submitit

from utils.submitit_defaults import SubmitConfig, DEFAULT_CONFIG
from utils.model_config import ModelSpec
from utils.vllm_server import vLLMServer

logger = logging.getLogger(__name__)

_SMOKE_DEFAULT_PROMPT = "Reply with exactly: OK"
_SMOKE_DEFAULT_MAX_TOKENS = 8


class GPUMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        self.samples.append(float(line.strip()))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join(timeout=1)
        if self.samples:
            avg = sum(self.samples) / len(self.samples)
            print(f"gpu utilization: avg={avg:.1f}%, samples={len(self.samples)}", flush=True)

DONE_STATES = {'COMPLETED'}
FAILED_STATES = {'FAILED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
TIMEOUT_STATES = {'TIMEOUT'}
CANCELLED_STATES = {'CANCELLED', 'REVOKED'}
RUNNING_STATES = {'RUNNING', 'COMPLETING', 'RESIZING', 'SUSPENDED'}
PENDING_STATES = {'PENDING', 'CONFIGURING', 'REQUEUED'}


def check_job_status(jobs: list[submitit.Job], job_meta: dict) -> dict[str, list]:
    status_map = {s: [] for s in ['pending', 'running', 'done', 'failed', 'timeout', 'cancelled']}
    if not jobs:
        return status_map

    job_lookup = {str(j.job_id): j for j in jobs}
    result = subprocess.run(
        ["sacct", "-j", ",".join(job_lookup.keys()), "-n", "-o", "JobID,State,ExitCode,Elapsed,MaxRSS", "--parsable2"],
        capture_output=True, text=True, timeout=30
    )

    seen = set()
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        raw_jid = parts[0]
        if "." in raw_jid:
            continue
        if raw_jid not in job_lookup or raw_jid in seen:
            continue
        seen.add(raw_jid)

        state = parts[1].split()[0]
        if state in DONE_STATES:
            mapped = 'done'
        elif state in FAILED_STATES:
            mapped = 'failed'
        elif state in TIMEOUT_STATES:
            mapped = 'timeout'
        elif state in CANCELLED_STATES:
            mapped = 'cancelled'
        elif state in RUNNING_STATES:
            mapped = 'running'
        elif state in PENDING_STATES:
            mapped = 'pending'
        else:
            logger.warning(f"unknown slurm state '{state}' for job {raw_jid}, treating as running")
            mapped = 'running'

        status_map[mapped].append(job_lookup[raw_jid])

        if mapped in ('failed', 'timeout'):
            exit_code = parts[2] if len(parts) > 2 else "?"
            elapsed = parts[3] if len(parts) > 3 else "?"
            name = job_meta.get(raw_jid, {}).get('name', raw_jid)
            logger.info(f"  {name} ({raw_jid}): {state.lower()}, exit={exit_code}, elapsed={elapsed}")

    for jid, job in job_lookup.items():
        if jid not in seen:
            status_map['pending'].append(job)

    logger.info("job status: " + ", ".join(f"{k}={len(v)}" for k, v in status_map.items() if v))
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
    if config.nodelist:
        params["slurm_nodelist"] = config.nodelist
    if config.constraint:
        params["slurm_constraint"] = config.constraint
    executor.update_parameters(**params)


def _wait_with_retries(
    jobs: list[submitit.Job],
    job_meta: dict[str, dict],  # job_id -> {config, name, fn, kwargs}
    executor: submitit.AutoExecutor,
    poll_interval: int,
    max_retries: int,
):
    """Wait for jobs, resubmitting failures up to max_retries."""
    retry_counts = {str(job.job_id): 0 for job in jobs}
    all_jobs = list(jobs)

    while True:
        time.sleep(poll_interval)
        status_map = check_job_status(all_jobs, job_meta)

        for job in status_map.get('failed', []) + status_map.get('timeout', []):
            jid = str(job.job_id)
            if retry_counts.get(jid, 0) < max_retries:
                meta = job_meta[jid]
                _configure_executor(executor, meta['config'], meta['name'])
                new_job = executor.submit(meta['fn'], **meta['kwargs'])
                job_meta[str(new_job.job_id)] = meta
                retry_counts[str(new_job.job_id)] = retry_counts.get(jid, 0) + 1
                all_jobs.append(new_job)
                all_jobs.remove(job)
                logger.info(f"resubmitted {jid} as {new_job.job_id}")

        active = len(status_map.get('pending', [])) + len(status_map.get('running', []))
        if active == 0:
            done = len(status_map.get('done', []))
            failed = len(status_map.get('failed', [])) + len(status_map.get('timeout', []))
            logger.info(f"complete: {done} done, {failed} failed")
            return all_jobs


def run_single_experiment(
    experiment_class, model_path: str, tensor_parallel_size: int,
    hint_fraction: float, fewshot: int, epochs: int, results_dir: str, config: SubmitConfig,
    debug: bool = False,
) -> dict:
    """Run single experiment inside submitit job."""
    from utils.setup import setup_logging, setup_inspect_logging, setup_openai_retry_debug_logging
    setup_logging()
    if debug:
        setup_inspect_logging(level="http")
        setup_openai_retry_debug_logging(enabled=True)

    model_name = os.path.basename(model_path)
    n_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', tensor_parallel_size))

    with GPUMonitor(), vLLMServer(
        model_path=model_path, tensor_parallel_size=tensor_parallel_size,
        max_model_len=config.max_model_len, gpu_memory_utilization=config.gpu_memory_utilization, n_gpus=n_gpus,
    ) as server:
        os.environ["VLLM_MAX_MODEL_LEN"] = str(config.max_model_len)
        experiment = experiment_class(
            model_name=model_name, vllm_port=server.port,
            timeout=config.timeout, max_connections=config.max_connections,
        )
        return experiment.run(hint_fraction=hint_fraction, fewshot=fewshot, epochs=epochs, results_dir=results_dir)


def run_baseline_eval(
    eval_name: str, model_path: str, tensor_parallel_size: int,
    results_dir: str, config: SubmitConfig, epochs: int = 1, limit: int | None = None,
    *,
    max_tokens: int,
    debug: bool = False,
) -> dict:
    """Run single baseline eval inside submitit job."""
    from utils.setup import setup_inspect_logging
    from experiments.runner import run_eval_with_vllm
    from experiments.registry import get_eval

    if debug:
        setup_inspect_logging(level="http")

    model_name = os.path.basename(model_path)
    output_file = str(Path(results_dir) / eval_name / model_name / f"{eval_name}.json")
    return run_eval_with_vllm(
        task_fn=get_eval(eval_name), model_path=model_path, tensor_parallel_size=tensor_parallel_size,
        output_file=output_file, config=config, epochs=epochs, limit=limit, max_tokens=max_tokens,
    )


def _output_json_is_complete(path: str) -> bool:
    """Return True iff output JSON exists and indicates completion."""
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        data = json.load(f)
    total = data.get("total_samples")
    completed = data.get("completed_samples")
    return isinstance(total, int) and isinstance(completed, int) and total > 0 and completed == total


def _get_running_job_configs(submitit_folder: str) -> set[tuple]:
    """Get set of (model_name, fewshot, hint_fraction) for currently running/pending jobs.
    
    Checks SLURM queue to find actually running/pending jobs, then gets their configs.
    """
    import glob
    import subprocess
    from pathlib import Path
    
    running_configs = set()
    
    # Get list of actually running/pending job IDs from SLURM
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            active_job_ids = set(result.stdout.strip().split())
        else:
            logger.warning("Failed to get SLURM queue, will not skip running jobs")
            return running_configs
    except Exception as e:
        logger.warning(f"Failed to check SLURM queue: {e}, will not skip running jobs")
        return running_configs
    
    # For each active job, try to get its configuration
    for job_id in active_job_ids:
        submitted_file = os.path.join(submitit_folder, f"{job_id}_submitted.pkl")
        if os.path.exists(submitted_file):
            try:
                import pickle
                with open(submitted_file, "rb") as f:
                    job_info = pickle.load(f)
                    # Extract configuration from job kwargs
                    if hasattr(job_info, 'kwargs'):
                        kwargs = job_info.kwargs
                        model_path = kwargs.get('model_path', '')
                        model_name = os.path.basename(model_path)
                        fewshot = kwargs.get('fewshot')
                        hint_fraction = kwargs.get('hint_fraction')
                        if model_name and fewshot is not None and hint_fraction is not None:
                            running_configs.add((model_name, fewshot, hint_fraction))
            except Exception:
                # If we can't read the pickle, skip it
                pass
    
    return running_configs


def launch_experiment(
    experiment_class, models: list[ModelSpec], hint_fractions: list[float],
    fewshots: list[int] = [0], epochs: int = 1, results_dir: str = "./results",
    config: SubmitConfig | None = None, skip_existing: bool = True,
    wait: bool = True, poll_interval: int = 300, max_retries: int = 3,
    debug: bool = False, max_jobs: int | None = None,
):
    """Launch experiment grid with retry logic.
    
    Args:
        max_jobs: Maximum number of jobs to submit (default: None = no limit)
        skip_existing: Skip jobs with existing output files or currently running (default: True)
    """
    config = config or DEFAULT_CONFIG
    executor = submitit.AutoExecutor(folder=config.submitit_folder)
    jobs = []
    job_meta = {}
    
    # Get currently running/pending jobs to avoid duplicates
    running_configs = _get_running_job_configs(config.submitit_folder) if skip_existing else set()
    if running_configs:
        logger.info(f"Found {len(running_configs)} jobs already running/pending, will skip those")

    for model in models:
        model_name = os.path.basename(model.path)
        overrides = dict(gpus_per_job=model.tp, partition=model.partitions, nodelist=model.nodelist)
        if model.account:
            overrides["account"] = model.account
        if model.constraint:
            overrides["constraint"] = model.constraint
        job_config = config.override(**overrides)
        job_name = f"{config.job_name_prefix}_{model_name}"
        _configure_executor(executor, job_config, job_name)

        for fewshot in fewshots:
            for hint_fraction in hint_fractions:
                if skip_existing:
                    # Check if output file already exists
                    output = experiment_class.get_output_filename(
                        results_dir=results_dir, model_name=model_name, fewshot=fewshot, hint_fraction=hint_fraction)
                    if os.path.exists(output):
                        continue
                    
                    # Check if job is already running/pending
                    config_tuple = (model_name, fewshot, hint_fraction)
                    if config_tuple in running_configs:
                        logger.info(f"Skipping {model_name}, fewshot={fewshot}, hint={hint_fraction} (already running)")
                        continue

                # Check if we've reached the max_jobs limit
                if max_jobs is not None and len(jobs) >= max_jobs:
                    logger.warning(f"Reached max_jobs limit of {max_jobs}, stopping submission")
                    break

                kwargs = dict(
                    experiment_class=experiment_class, model_path=model.path,
                    tensor_parallel_size=model.tp, hint_fraction=hint_fraction, fewshot=fewshot,
                    epochs=epochs, results_dir=results_dir, config=job_config, debug=debug,
                )
                job = executor.submit(run_single_experiment, **kwargs)
                jobs.append(job)
                job_meta[str(job.job_id)] = {'config': job_config, 'name': job_name, 'fn': run_single_experiment, 'kwargs': kwargs}
                logger.info(f"submitted {job.job_id}: {model_name}, fewshot={fewshot}, hint={hint_fraction}")
            
            # Break from fewshot loop if max_jobs reached
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
        
        # Break from model loop if max_jobs reached
        if max_jobs is not None and len(jobs) >= max_jobs:
            break

    logger.info(f"submitted {len(jobs)} jobs")
    if not jobs or not wait:
        return jobs
    return _wait_with_retries(jobs, job_meta, executor, poll_interval, max_retries)


def launch_baseline(
    eval_names: list[str], models: list[ModelSpec], results_dir: str = "./baseline",
    config: SubmitConfig | None = None, epochs: int = 1, limit: int | None = None,
    *,
    max_tokens: int,
    skip_existing: bool = True, wait: bool = True, poll_interval: int = 300, max_retries: int = 3,
    debug: bool = False,
):
    """Launch baseline evals with retry logic."""
    config = config or DEFAULT_CONFIG
    executor = submitit.AutoExecutor(folder=config.submitit_folder)
    jobs = []
    job_meta = {}

    for eval_name in eval_names:
        for model in models:
            model_name = os.path.basename(model.path)

            if skip_existing:
                output = str(Path(results_dir) / eval_name / model_name / f"{eval_name}.json")
                if _output_json_is_complete(output):
                    continue

            job_config = config.override(
                gpus_per_job=model.tp,
                partition=model.partitions,
                nodelist=model.nodelist,
                **({"constraint": model.constraint} if model.constraint else {}),
            )
            job_name = f"baseline_{eval_name}_{model_name}"
            _configure_executor(executor, job_config, job_name)

            kwargs = dict(
                eval_name=eval_name, model_path=model.path, tensor_parallel_size=model.tp,
                results_dir=results_dir, config=job_config, epochs=epochs, limit=limit,
                max_tokens=max_tokens, debug=debug,
            )
            job = executor.submit(run_baseline_eval, **kwargs)
            jobs.append(job)
            job_meta[str(job.job_id)] = {'config': job_config, 'name': job_name, 'fn': run_baseline_eval, 'kwargs': kwargs}
            logger.info(f"submitted {job.job_id}: {eval_name} / {model_name}")

    logger.info(f"submitted {len(jobs)} jobs")
    if not jobs or not wait:
        return jobs
    return _wait_with_retries(jobs, job_meta, executor, poll_interval, max_retries)


def run_smoke_inference(
    model_path: str,
    tensor_parallel_size: int,
    config: SubmitConfig,
    prompt: str = _SMOKE_DEFAULT_PROMPT,
    max_tokens: int = _SMOKE_DEFAULT_MAX_TOKENS,
) -> dict:
    """Start vLLM and run one tiny OpenAI-style request."""
    from utils.setup import setup_logging

    setup_logging()

    n_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", tensor_parallel_size))
    logger.info(f"Allocated {n_gpus} GPUs by SLURM")

    with GPUMonitor(), vLLMServer(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        n_gpus=n_gpus,
    ) as server:
        import openai

        base_url = f"http://localhost:{server.port}/v1"
        logger.info(f"Sending smoke request to {base_url}")
        client = openai.OpenAI(base_url=base_url, api_key="local")

        response = client.chat.completions.create(
            model=server.served_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        logger.info(f"Smoke response: {text!r}")
        return {
            "status": "ok",
            "model_path": model_path,
            "served_model_name": server.served_model_name,
            "base_url": base_url,
            "prompt": prompt,
            "response": text,
        }


def launch_smoke_inference(
    model: ModelSpec,
    config: SubmitConfig | None = None,
    prompt: str = _SMOKE_DEFAULT_PROMPT,
    max_tokens: int = _SMOKE_DEFAULT_MAX_TOKENS,
    wait: bool = True,
) -> submitit.Job:
    """Submit a single 1-job smoke test."""
    config = config or DEFAULT_CONFIG
    executor = submitit.AutoExecutor(folder=config.submitit_folder)
    _configure_executor(executor, config, name=f"smoke_{os.path.basename(model.path)}")

    kwargs = dict(
        model_path=model.path,
        tensor_parallel_size=model.tp,
        config=config,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    job = executor.submit(run_smoke_inference, **kwargs)
    logger.info(f"submitted {job.job_id}: smoke / {os.path.basename(model.path)}")
    if wait:
        job.result()
    return job
