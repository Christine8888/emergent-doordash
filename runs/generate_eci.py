from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

from src.eci_progress import compute_eci_benchmark_progress
from src.eci_runner import (
    BENCHMARK_CONFIGS,
    DEFAULT_CHECKPOINT_EVERY,
    EPOCHS,
    MAX_RETRIES,
    MAX_TOKENS,
    is_eci_model_complete,
    run_eci_benchmarks,
)
from src.model_config import ALL_MODEL_PATHS, ModelSpec, get_model_spec
from src.storage import build_eci_score_path

BENCHMARKS = [
    "mmlu_5_shot__language_en_us__cot_false",
    "bbh__prompt_type_answer_only",
    "arc_challenge",
    "math__levels_5__fewshot_0",
    "hellaswag__split_validation",
    "piqa",
    "winogrande__dataset_name_winogrande_xl__fewshot_5",
]
MODELS_TO_RUN = list(ALL_MODEL_PATHS)
DEFAULT_SLURM_CPUS_PER_TASK = 16
DEFAULT_SLURM_MEM_GB = 64
EIGHT_GPU_SLURM_CPUS_PER_TASK = 120
EIGHT_GPU_SLURM_MEM_GB = 1000
SLURM_TIME_HOURS_OVERRIDE: int | None = None
NLP_SLURM_ACCOUNT = "nlp"
NLP_SLURM_PARTITION = "sphinx,jag-standard"
SPHINX_SLURM_ACCOUNT = "nlp"
SPHINX_SLURM_PARTITION = "sphinx"
MISO_SLURM_ACCOUNT = "miso"
MISO_SLURM_PARTITION = "miso"


def _timestamp() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _selected_models(model: str) -> list[ModelSpec]:
    if model == "all":
        return [get_model_spec(model_path) for model_path in MODELS_TO_RUN]
    if model not in MODELS_TO_RUN:
        raise ValueError(f"Model {model!r} is not in MODELS_TO_RUN")
    return [get_model_spec(model)]


def _resolve_parallelism(spec: ModelSpec, num_gpus: int | None) -> tuple[int, int, int]:
    tp = spec.tp
    if num_gpus is None:
        return tp, 1, tp
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if num_gpus < tp:
        raise ValueError(f"num_gpus={num_gpus} is smaller than model tp={tp} for model={spec.path}")
    if num_gpus % tp != 0:
        raise ValueError(
            f"num_gpus={num_gpus} must be divisible by model tp={tp} for model={spec.path}"
        )
    dp = num_gpus // tp
    return tp, dp, num_gpus


def _resolve_slurm_account(*, cluster: str) -> tuple[str, str, str]:
    if cluster == "miso":
        return "miso", MISO_SLURM_ACCOUNT, MISO_SLURM_PARTITION
    if cluster == "sphinx":
        return "sphinx", SPHINX_SLURM_ACCOUNT, SPHINX_SLURM_PARTITION
    if cluster == "nlp":
        return "nlp", NLP_SLURM_ACCOUNT, NLP_SLURM_PARTITION
    raise ValueError(f"Unsupported cluster: {cluster!r}")


def _resolve_slurm_time_hours(*, slurm_account: str) -> int:
    if SLURM_TIME_HOURS_OVERRIDE is not None:
        return SLURM_TIME_HOURS_OVERRIDE
    if slurm_account == "miso":
        return 6
    return 60


def _resolve_slurm_resources(*, requested_gpus: int) -> tuple[int, int]:
    if requested_gpus == 8:
        return EIGHT_GPU_SLURM_CPUS_PER_TASK, EIGHT_GPU_SLURM_MEM_GB
    return DEFAULT_SLURM_CPUS_PER_TASK, DEFAULT_SLURM_MEM_GB


def _load_sampling_params(models: list[ModelSpec]) -> dict[str, dict[str, Any]]:
    return {spec.path: dict(spec.sampling_params) for spec in models}


def _get_active_eci_jobs_by_model() -> dict[str, list[str]]:
    active_jobs_by_model: dict[str, list[str]] = {}
    submitit_dir = Path("data/submitit_logs/eci_scores")
    try:
        user = os.environ.get("USER", "")
        cmd = ["squeue", "-h", "-o", "%i"]
        if user:
            cmd[1:1] = ["-u", user]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0:
            fallback = subprocess.run(["squeue", "-h", "-o", "%i"], capture_output=True, text=True, timeout=5, check=False)
            if fallback.returncode != 0:
                _log("[generate_eci] warning: failed to query SLURM queue; submitting without active-job filter")
                return active_jobs_by_model
            result = fallback
        active_job_ids = {job_id.strip() for job_id in result.stdout.splitlines() if job_id.strip()}
    except Exception as exc:
        _log(f"[generate_eci] warning: failed to query SLURM queue ({exc}); submitting without active-job filter")
        return active_jobs_by_model

    for job_id in sorted(active_job_ids):
        submitted_file = submitit_dir / f"{job_id}_submitted.pkl"
        if not submitted_file.exists():
            continue
        try:
            with open(submitted_file, "rb") as f:
                job_info = pickle.load(f)
            kwargs = getattr(job_info, "kwargs", None)
            if not isinstance(kwargs, dict):
                continue
            model_path = kwargs.get("model_path")
            if isinstance(model_path, str) and model_path:
                active_jobs_by_model.setdefault(model_path, []).append(job_id)
        except Exception:
            continue

    return active_jobs_by_model


def _selected_benchmarks(benchmark: str) -> list[str]:
    if benchmark == "all":
        return list(BENCHMARKS)
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Benchmark {benchmark!r} is not in BENCHMARKS")
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(f"Benchmark {benchmark!r} is missing from src.eci_runner.BENCHMARK_CONFIGS")
    return [benchmark]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simplified Inspect-backed ECI scoring.")
    parser.add_argument("--benchmark", type=str, choices=["all"] + BENCHMARKS, default="all")
    parser.add_argument("--model", type=str, choices=["all"] + MODELS_TO_RUN, default="all")
    parser.add_argument("--backend", choices=["local-vllm", "together-serverless"], default="local-vllm")
    parser.add_argument(
        "--cluster",
        choices=["nlp", "sphinx", "miso"],
        default="nlp",
        help="Submit target cluster/account routing.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--executor", choices=["local", "submitit"], default="local")
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.91)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Requested GPUs per model job. Must be divisible by the model tp from src/model_config.py.",
    )
    parser.add_argument("--dry-run", type=_parse_bool, default=False)
    return parser


def _build_run_metadata(
    *,
    args: argparse.Namespace,
    benchmark_names: list[str],
    spec: ModelSpec,
    tp: int,
    dp: int,
    requested_gpus: int,
    sampling_params: dict[str, bool | float | int],
    resolved_cluster: str | None = None,
    slurm_account: str | None = None,
    slurm_partition: str | None = None,
    slurm_time_hours: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem_gb: int | None = None,
) -> dict[str, Any]:
    return {
        "launcher": "runs.generate_eci",
        "cli_args": dict(vars(args)),
        "job": {
            "executor": args.executor,
            "backend": args.backend,
            "benchmarks": benchmark_names,
            "limit": args.limit,
            "epochs": EPOCHS,
        },
        "model_spec": {
            "path": spec.path,
            "name": spec.name,
            "tp": spec.tp,
            "constraint": spec.constraint,
            "sampling_params": sampling_params,
        },
        "parallelism": {
            "num_gpus_arg": args.num_gpus,
            "requested_gpus": requested_gpus,
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
        },
        "inspect": {
            "max_connections": args.max_connections,
            "checkpoint_every": args.checkpoint_every,
            "max_tokens": MAX_TOKENS,
            "max_retries": MAX_RETRIES,
        },
        "vllm_server": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
        },
        "slurm": {
            "cluster_arg": args.cluster,
            "resolved_cluster": resolved_cluster,
            "account": slurm_account,
            "partition": slurm_partition,
            "time_hours": slurm_time_hours,
            "cpus_per_task": slurm_cpus_per_task,
            "mem_gb": slurm_mem_gb,
        },
    }


def _print_plan(args: argparse.Namespace, benchmark_names: list[str], models: list[ModelSpec]) -> None:
    _log("[generate_eci] plan")
    print(
        json.dumps(
            {
                "executor": args.executor,
                "backend": args.backend,
                "benchmarks": benchmark_names,
                "models": [m.path for m in models],
                "limit": args.limit,
                "epochs": EPOCHS,
                "max_retries": MAX_RETRIES,
            },
            indent=2,
        ),
        flush=True,
    )
    for spec in models:
        for benchmark_name in benchmark_names:
            path = build_eci_score_path(
                benchmark_name=benchmark_name,
                model=spec.path,
                data_root="data",
            )
            _log(f"[generate_eci] output -> {path}")


def _run_local(
    args: argparse.Namespace,
    benchmark_names: list[str],
    models: list[ModelSpec],
    sampling_params_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in models:
        tp, dp, requested_gpus = _resolve_parallelism(spec, args.num_gpus)
        sampling_params = sampling_params_by_model.get(spec.path, {})
        run_metadata = _build_run_metadata(
            args=args,
            benchmark_names=benchmark_names,
            spec=spec,
            tp=tp,
            dp=dp,
            requested_gpus=requested_gpus,
            sampling_params=sampling_params,
        )
        _log(f"[generate_eci] local model={spec.path}")
        results.append(
            run_eci_benchmarks(
                benchmark_names=benchmark_names,
                model_path=spec.path,
                tensor_parallel_size=tp,
                data_parallel_size=dp,
                sampling_params=sampling_params,
                run_metadata=run_metadata,
                limit=args.limit,
                max_connections=args.max_connections,
                checkpoint_every=args.checkpoint_every,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                backend=args.backend,
            )
        )
    return results


def _run_submitit(
    args: argparse.Namespace,
    benchmark_names: list[str],
    models: list[ModelSpec],
    sampling_params_by_model: dict[str, dict[str, Any]],
) -> list[Any]:
    import submitit

    submitit_dir = Path("data/submitit_logs/eci_scores")
    submitit_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(submitit_dir))
    active_jobs_by_model = _get_active_eci_jobs_by_model()

    jobs = []
    for spec in models:
        is_complete = False
        if args.limit is None:
            progress_rows = [
                compute_eci_benchmark_progress(
                    benchmark_name=benchmark_name,
                    model=spec.path,
                    data_root="data",
                )
                for benchmark_name in benchmark_names
            ]
            is_complete = bool(progress_rows) and all(row.status == "complete" for row in progress_rows)
        else:
            is_complete = is_eci_model_complete(
                benchmark_names=benchmark_names,
                model_path=spec.path,
                limit=args.limit,
                data_root="data",
            )
        if is_complete:
            _log(f"[generate_eci] skip complete model={spec.path}")
            continue
        active_job_ids = active_jobs_by_model.get(spec.path, [])
        if active_job_ids:
            _log(
                f"[generate_eci] skip already queued/running model={spec.path} "
                f"job_ids={','.join(active_job_ids)}"
            )
            continue
        tp, dp, requested_gpus = _resolve_parallelism(spec, args.num_gpus)
        model_name = spec.path.split("/")[-1]
        resolved_cluster, account, partition = _resolve_slurm_account(cluster=args.cluster)
        time_hours = _resolve_slurm_time_hours(slurm_account=account)
        cpus_per_task, mem_gb = _resolve_slurm_resources(requested_gpus=requested_gpus)
        sampling_params = sampling_params_by_model.get(spec.path, {})
        run_metadata = _build_run_metadata(
            args=args,
            benchmark_names=benchmark_names,
            spec=spec,
            tp=tp,
            dp=dp,
            requested_gpus=requested_gpus,
            sampling_params=sampling_params,
            resolved_cluster=resolved_cluster,
            slurm_account=account,
            slurm_partition=partition,
            slurm_time_hours=time_hours,
            slurm_cpus_per_task=cpus_per_task,
            slurm_mem_gb=mem_gb,
        )
        executor.update_parameters(
            name=f"eci_{model_name}",
            slurm_account=account,
            slurm_partition=partition,
            slurm_gpus_per_node=requested_gpus,
            slurm_cpus_per_task=cpus_per_task,
            slurm_mem=f"{mem_gb}GB",
            slurm_time=time_hours * 60,
            timeout_min=time_hours * 60,
        )
        job = executor.submit(
            run_eci_benchmarks,
            benchmark_names=benchmark_names,
            model_path=spec.path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            sampling_params=sampling_params,
            run_metadata=run_metadata,
            limit=args.limit,
            max_connections=args.max_connections,
            checkpoint_every=args.checkpoint_every,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            backend=args.backend,
        )
        jobs.append(job)
        _log(f"[generate_eci] submitted job_id={job.job_id} model={spec.path}")
    return jobs


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    benchmark_names = _selected_benchmarks(args.benchmark)
    models = _selected_models(args.model)
    sampling_params_by_model = _load_sampling_params(models)
    _print_plan(args, benchmark_names, models)

    if args.dry_run:
        return

    if args.executor == "local":
        results = _run_local(args, benchmark_names, models, sampling_params_by_model)
        print(json.dumps(results, indent=2), flush=True)
        return

    jobs = _run_submitit(args, benchmark_names, models, sampling_params_by_model)
    print(
        json.dumps(
            {
                "submitted_jobs": [job.job_id for job in jobs],
                "models": [spec.path for spec in models],
                "benchmarks": benchmark_names,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()



"""
MISO
python -m runs.generate_eci \
    --backend local-vllm \
    --executor submitit \
    --cluster miso \
    --num-gpus 8 \
    --max-connections 360

NLP
python -m runs.generate_eci \
      --backend local-vllm \
      --executor submitit \
      --cluster sphinx \
      --num-gpus 1 \
      --max-connections 48
"""
