from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.hint_types import HintType
from src.hinted_inference import run_hinted_inference
from src.model_config import ALL_MODEL_PATHS, ModelSpec, get_model_spec
from src.storage import build_hinted_inference_path
from src.vllm_server import VLLMServer, VLLMServerConfig

MODELS_TO_RUN = list(ALL_MODEL_PATHS)
HINT_FRACTIONS = [i / 10 for i in range(11)]
REQUEST_TIMEOUT_SECONDS = 3600
MAX_NUM_BATCHED_TOKENS = 32768
MAX_TOKENS = 32000
MAX_MODEL_LEN = 40000
MAX_RETRIES = 2
# Leave as None for auto policy: 60h default, 6h on miso.
SLURM_TIME_HOURS_OVERRIDE: int | None = None
DEFAULT_SLURM_CPUS_PER_TASK = 16
DEFAULT_SLURM_MEM_GB = 64
EIGHT_GPU_SLURM_CPUS_PER_TASK = 120
EIGHT_GPU_SLURM_MEM_GB = 1000
NLP_SLURM_ACCOUNT = "nlp"
NLP_SLURM_PARTITION = "sphinx,jag-standard"
SPHINX_SLURM_ACCOUNT = "nlp"
SPHINX_SLURM_PARTITION = "sphinx"
MISO_SLURM_ACCOUNT = "miso"
MISO_SLURM_PARTITION = "miso"


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _setup_vllm_env(*, port: int, served_model_name: str) -> None:
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["VLLM_API_KEY"] = "local"
    os.environ["INSPECT_EVAL_MODEL"] = f"vllm/{served_model_name}"
    os.environ["OPENAI_TIMEOUT"] = str(REQUEST_TIMEOUT_SECONDS)


def _run_single_model_job(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    hint_fractions: list[float],
    model_path: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    sampling_params: dict[str, bool | float | int],
    run_metadata: dict[str, Any],
    max_connections: int,
    checkpoint_every: int,
    gpu_memory_utilization: float,
    dtype: str,
    build_only: bool,
) -> dict[str, Any]:
    model_name = model_path.split("/")[-1]

    if build_only:
        summaries = run_hinted_inference(
            benchmark_name=benchmark,
            hint_type=hint_type,
            model=model_name,
            inspect_model_id=f"vllm/{model_name}",
            fractioner=fractioner,
            hint_fractions=hint_fractions,
            do_sample=sampling_params.get("do_sample"),
            temperature=sampling_params.get("temperature"),
            top_p=sampling_params.get("top_p"),
            top_k=sampling_params.get("top_k"),
            repetition_penalty=sampling_params.get("repetition_penalty"),
            max_tokens=MAX_TOKENS,
            max_connections=max_connections,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            checkpoint_every=checkpoint_every,
            vllm_metrics_url=None,
            build_only=True,
            run_metadata=run_metadata,
        )
    else:
        server_config = VLLMServerConfig(
            model_path=model_path,
            served_model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            dtype=dtype,
        )

        with VLLMServer(server_config) as server:
            _setup_vllm_env(
                port=server.port,
                served_model_name=model_name,
            )
            summaries = run_hinted_inference(
                benchmark_name=benchmark,
                hint_type=hint_type,
                model=model_name,
                inspect_model_id=f"vllm/{model_name}",
                fractioner=fractioner,
                hint_fractions=hint_fractions,
                do_sample=sampling_params.get("do_sample"),
                temperature=sampling_params.get("temperature"),
                top_p=sampling_params.get("top_p"),
                top_k=sampling_params.get("top_k"),
                repetition_penalty=sampling_params.get("repetition_penalty"),
                max_tokens=MAX_TOKENS,
                max_connections=max_connections,
                timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
                checkpoint_every=checkpoint_every,
                vllm_metrics_url=f"http://localhost:{server.port}/metrics",
                build_only=False,
                run_metadata=run_metadata,
            )
    return {
        "model": model_name,
        "model_path": model_path,
        "run_metadata": run_metadata,
        "summaries": [asdict(summary) for summary in summaries],
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hinted inference with Inspect + local vLLM.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[h.value for h in HintType], required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["all"] + MODELS_TO_RUN, default="all")
    parser.add_argument(
        "--cluster",
        choices=["nlp", "sphinx", "miso"],
        default="nlp",
        help="Submit target cluster/account routing (no auto-inference).",
    )
    
    parser.add_argument("--max-jobs", type=int, default=None, help="Cap number of model jobs launched.")
    parser.add_argument("--executor", choices=["local", "submitit"], default="local")

    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=500)

    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--build-only", type=_parse_bool, default=False)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Requested GPUs per model job. "
            "Must be divisible by the model's tp from src/model_config.py."
        ),
    )

    parser.add_argument("--dry-run", type=_parse_bool, default=False)
    return parser


def _apply_job_cap(models: list[ModelSpec], max_jobs: int | None) -> list[ModelSpec]:
    if max_jobs is None:
        return models
    if max_jobs < 1:
        raise ValueError("max_jobs must be >= 1")
    return models[:max_jobs]


def _selected_models(model: str) -> list[ModelSpec]:
    if not MODELS_TO_RUN:
        raise ValueError("MODELS_TO_RUN cannot be empty")
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
        raise ValueError(
            f"num_gpus={num_gpus} is smaller than model tp={tp} for model={spec.path}"
        )
    if num_gpus % tp != 0:
        raise ValueError(
            f"num_gpus={num_gpus} must be divisible by model tp={tp} for model={spec.path}"
        )
    dp = num_gpus // tp
    return tp, dp, num_gpus


def _resolve_slurm_account(
    *,
    cluster: str,
) -> tuple[str, str, str]:
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


def _build_run_metadata(
    *,
    args: argparse.Namespace,
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
        "launcher": "runs.generate_hinted",
        "cli_args": dict(vars(args)),
        "constants": {
            "models_to_run": list(MODELS_TO_RUN),
            "hint_fractions": list(HINT_FRACTIONS),
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_tokens": MAX_TOKENS,
            "max_retries": MAX_RETRIES,
            "max_model_len": MAX_MODEL_LEN,
            "slurm_time_hours_override": SLURM_TIME_HOURS_OVERRIDE,
            "default_slurm_cpus_per_task": DEFAULT_SLURM_CPUS_PER_TASK,
            "default_slurm_mem_gb": DEFAULT_SLURM_MEM_GB,
            "eight_gpu_slurm_cpus_per_task": EIGHT_GPU_SLURM_CPUS_PER_TASK,
            "eight_gpu_slurm_mem_gb": EIGHT_GPU_SLURM_MEM_GB,
        },
        "job": {
            "executor": args.executor,
            "benchmark": args.benchmark,
            "hint_type": args.hint_type,
            "fractioner": args.fractioner,
            "build_only": args.build_only,
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
        "vllm_server": {
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "dtype": args.dtype,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "inspect_generation": {
            "do_sample": sampling_params.get("do_sample"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "top_k": sampling_params.get("top_k"),
            "repetition_penalty": sampling_params.get("repetition_penalty"),
            "max_tokens": MAX_TOKENS,
            "max_connections": args.max_connections,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "checkpoint_every": args.checkpoint_every,
            "build_only": args.build_only,
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


def _print_plan(args: argparse.Namespace, models: list[ModelSpec]) -> None:
    print("[generate_hinted] plan", flush=True)
    print(
        json.dumps(
            {
                "executor": args.executor,
                "benchmark": args.benchmark,
                "hint_type": args.hint_type,
                "fractioner": args.fractioner,
                "hint_fractions": HINT_FRACTIONS,
                "models": [m.path for m in models],
            },
            indent=2,
        ),
        flush=True,
    )

    for spec in models:
        model_name = spec.path.split("/")[-1]
        for fraction in HINT_FRACTIONS:
            path = build_hinted_inference_path(
                benchmark_name=args.benchmark,
                model=model_name,
                hint_type=args.hint_type,
                fractioner=args.fractioner,
                hint_fraction=fraction,
                data_root="data",
            )
            print(f"  output -> {path}", flush=True)


def _run_local(
    args: argparse.Namespace,
    models: list[ModelSpec],
    sampling_params_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in models:
        tp, dp, requested_gpus = _resolve_parallelism(spec, args.num_gpus)
        sampling_params = sampling_params_by_model.get(spec.path, {})
        run_metadata = _build_run_metadata(
            args=args,
            spec=spec,
            tp=tp,
            dp=dp,
            requested_gpus=requested_gpus,
            sampling_params=sampling_params,
        )
        print(f"[generate_hinted] local model={spec.path}", flush=True)
        result = _run_single_model_job(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            hint_fractions=HINT_FRACTIONS,
            model_path=spec.path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            sampling_params=sampling_params,
            run_metadata=run_metadata,
            max_connections=args.max_connections,
            checkpoint_every=args.checkpoint_every,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            build_only=args.build_only,
        )
        results.append(result)
    return results


def _run_submitit(
    args: argparse.Namespace,
    models: list[ModelSpec],
    sampling_params_by_model: dict[str, dict[str, Any]],
) -> list[Any]:
    import submitit

    submitit_dir = Path("data/submitit_logs/hinted")
    submitit_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(submitit_dir))

    jobs = []
    for spec in models:
        tp, dp, requested_gpus = _resolve_parallelism(spec, args.num_gpus)
        model_name = spec.path.split("/")[-1]
        resolved_cluster, account, partition = _resolve_slurm_account(cluster=args.cluster)
        time_hours = _resolve_slurm_time_hours(slurm_account=account)
        cpus_per_task, mem_gb = _resolve_slurm_resources(requested_gpus=requested_gpus)
        sampling_params = sampling_params_by_model.get(spec.path, {})
        run_metadata = _build_run_metadata(
            args=args,
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
        params = {
            "name": f"hinted_{model_name}",
            "slurm_account": account,
            "slurm_partition": partition,
            "slurm_gpus_per_node": requested_gpus,
            "slurm_cpus_per_task": cpus_per_task,
            "slurm_mem": f"{mem_gb}GB",
            "slurm_time": time_hours * 60,
            "timeout_min": time_hours * 60,
        }

        executor.update_parameters(**params)
        job = executor.submit(
            _run_single_model_job,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            hint_fractions=HINT_FRACTIONS,
            model_path=spec.path,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            sampling_params=sampling_params,
            run_metadata=run_metadata,
            max_connections=args.max_connections,
            checkpoint_every=args.checkpoint_every,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            build_only=args.build_only,
        )
        jobs.append(job)
        print(f"[generate_hinted] submitted job_id={job.job_id} model={spec.path}", flush=True)
    return jobs


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    models = _selected_models(args.model)
    models = _apply_job_cap(models, args.max_jobs)
    sampling_params_by_model = _load_sampling_params(models)
    _print_plan(args, models)

    if args.dry_run:
        print("[generate_hinted] dry_run=true: exiting before launch", flush=True)
        return

    if args.executor == "local":
        results = _run_local(args, models, sampling_params_by_model)
        print(json.dumps(results, indent=2), flush=True)
        return

    _run_submitit(args, models, sampling_params_by_model)


if __name__ == "__main__":
    main()


"""
MISO
python -m runs.generate_hinted \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner truncate_sentence \
    --model Qwen/Qwen3-4B \
    --executor submitit \
    --cluster miso \
    --max-connections 160 \
    --num-gpus 8 \
    --checkpoint-every 1000

python -m runs.generate_hinted \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --model Qwen/Qwen3-4B \
    --executor submitit \
    --cluster miso \
    --max-connections 160 \
    --num-gpus 8 \
    --checkpoint-every 1000

NLP
python -m runs.generate_hinted \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner truncate_word \
    --model Qwen/Qwen3-4B \
    --executor submitit \
    --cluster sphinx \
    --max-connections 48 \
    --num-gpus 1 \
    --checkpoint-every 500

CREATING HINTS

python -m runs.generate_hinted \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner truncate_word \
    --model Qwen/Qwen3-4B \
    --executor local \
    --build-only true

"""
