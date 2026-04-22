from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.hint_types import HintType
from src.hinted_inference import build_expanded_hinted_prompt_dataset, run_hinted_inference
from src.model_config import ALL_MODEL_PATHS, ModelSpec, get_model_spec
from src.storage import build_hint_generation_path, build_hinted_inference_path, read_jsonl
from src.types import ExpandedHintedPromptRecord, HintGenerationRecord, HintedInferenceRecord
from src.vllm_server import DEFAULT_HEALTH_TIMEOUT_SECONDS, VLLMServer, VLLMServerConfig

MODELS_TO_RUN = list(ALL_MODEL_PATHS)
HINT_FRACTIONS = [i / 10 for i in range(11)]
REQUEST_TIMEOUT_SECONDS = 3600
MAX_NUM_BATCHED_TOKENS = 32768
MAX_TOKENS = 32000
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
TOGETHER_SERVERLESS_PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "moonshotai/Kimi-K2.5": {"input": 0.50, "output": 2.80},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.20},
    "Qwen/Qwen3.5-397B-A17B": {"input": 0.60, "output": 3.60},
}


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _setup_vllm_env(*, port: int) -> None:
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["VLLM_API_KEY"] = "local"


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
    max_requests: int | None,
    checkpoint_every: int,
    gpu_memory_utilization: float,
    dtype: str,
    backend: str,
    build_only: bool,
) -> dict[str, Any]:
    if build_only:
        summaries = run_hinted_inference(
            benchmark_name=benchmark,
            hint_type=hint_type,
            model=model_path,
            fractioner=fractioner,
            hint_fractions=hint_fractions,
            do_sample=sampling_params.get("do_sample"),
            temperature=sampling_params.get("temperature"),
            top_p=sampling_params.get("top_p"),
            top_k=sampling_params.get("top_k"),
            repetition_penalty=sampling_params.get("repetition_penalty"),
            max_tokens=MAX_TOKENS,
            max_connections=max_connections,
            max_requests=max_requests,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            checkpoint_every=checkpoint_every,
            vllm_metrics_url=None,
            backend=backend,
            build_only=True,
            run_metadata=run_metadata,
        )
    else:
        if backend == "local-vllm":
            server_config = VLLMServerConfig(
                model_path=model_path,
                served_model_name=model_path,
                tensor_parallel_size=tensor_parallel_size,
                data_parallel_size=data_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                dtype=dtype,
            )

            with VLLMServer(server_config) as server:
                _setup_vllm_env(
                    port=server.port,
                )
                summaries = run_hinted_inference(
                    benchmark_name=benchmark,
                    hint_type=hint_type,
                    model=model_path,
                    fractioner=fractioner,
                    hint_fractions=hint_fractions,
                    do_sample=sampling_params.get("do_sample"),
                    temperature=sampling_params.get("temperature"),
                    top_p=sampling_params.get("top_p"),
                    top_k=sampling_params.get("top_k"),
                    repetition_penalty=sampling_params.get("repetition_penalty"),
                    max_tokens=MAX_TOKENS,
                    max_connections=max_connections,
                    max_requests=max_requests,
                    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                    max_retries=MAX_RETRIES,
                    checkpoint_every=checkpoint_every,
                    vllm_metrics_url=f"http://localhost:{server.port}/metrics",
                    backend=backend,
                    build_only=False,
                    run_metadata=run_metadata,
                )
        elif backend == "together-serverless":
            summaries = run_hinted_inference(
                benchmark_name=benchmark,
                hint_type=hint_type,
                model=model_path,
                fractioner=fractioner,
                hint_fractions=hint_fractions,
                do_sample=sampling_params.get("do_sample"),
                temperature=sampling_params.get("temperature"),
                top_p=sampling_params.get("top_p"),
                top_k=sampling_params.get("top_k"),
                repetition_penalty=sampling_params.get("repetition_penalty"),
                max_tokens=MAX_TOKENS,
                max_connections=max_connections,
                max_requests=max_requests,
                timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
                checkpoint_every=checkpoint_every,
                vllm_metrics_url=None,
                backend=backend,
                build_only=False,
                run_metadata=run_metadata,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend!r}")
    return {
        "model": model_path,
        "model_path": model_path,
        "run_metadata": run_metadata,
        "summaries": [asdict(summary) for summary in summaries],
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hinted inference with local vLLM or Together serverless.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[h.value for h in HintType], required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["all"] + MODELS_TO_RUN, default="all")
    parser.add_argument("--backend", choices=["local-vllm", "together-serverless"], default="local-vllm")
    parser.add_argument(
        "--cluster",
        choices=["nlp", "sphinx", "miso"],
        default="nlp",
        help="Submit target cluster/account routing (no auto-inference).",
    )
    
    parser.add_argument("--max-jobs", type=int, default=None, help="Cap number of model jobs launched.")
    parser.add_argument("--executor", choices=["local", "submitit"], default="local")

    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional global cap on the number of pending requests to launch across all hint fractions.",
    )
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


def _load_hint_ids(*, benchmark: str, hint_type: str) -> list[str]:
    path = build_hint_generation_path(
        benchmark_name=benchmark,
        hint_type=hint_type,
        data_root="data",
    )
    rows = read_jsonl(path, model_cls=HintGenerationRecord)
    typed_rows = [row for row in rows if isinstance(row, HintGenerationRecord)]
    return [row.hint_id for row in typed_rows]


def _load_hinted_rows(path: Path) -> list[HintedInferenceRecord]:
    rows = read_jsonl(path, model_cls=HintedInferenceRecord)
    return [row for row in rows if isinstance(row, HintedInferenceRecord)]


def _get_together_pricing_per_million(model_name: str) -> dict[str, float]:
    pricing = TOGETHER_SERVERLESS_PRICING_PER_MILLION.get(model_name)
    if pricing is None:
        raise ValueError(
            f"Missing Together serverless pricing for model={model_name!r}. "
            "Add it to TOGETHER_SERVERLESS_PRICING_PER_MILLION in runs/generate_hinted.py."
        )
    if "input" not in pricing or "output" not in pricing:
        raise ValueError(
            f"Incomplete Together serverless pricing for model={model_name!r}. "
            "Expected both 'input' and 'output' prices per million tokens."
        )
    return pricing


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        token_ids = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)
    return len(tokenizer.encode(prompt, add_special_tokens=True))


def _estimate_together_cost_for_model(
    *,
    args: argparse.Namespace,
    spec: ModelSpec,
) -> dict[str, Any]:
    hint_ids = _load_hint_ids(benchmark=args.benchmark, hint_type=args.hint_type)
    hint_id_set = set(hint_ids)
    pricing = _get_together_pricing_per_million(spec.path)
    tokenizer = _load_tokenizer(spec.path)
    fraction_paths = build_expanded_hinted_prompt_dataset(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        hint_fractions=HINT_FRACTIONS,
        data_root="data",
    )

    total_pending_requests = 0
    total_input_tokens = 0
    total_max_output_tokens = 0
    per_fraction: list[dict[str, Any]] = []
    remaining_request_budget = args.max_requests

    for fraction in HINT_FRACTIONS:
        expanded_rows = read_jsonl(fraction_paths[fraction], model_cls=ExpandedHintedPromptRecord)
        typed_expanded_rows = [row for row in expanded_rows if isinstance(row, ExpandedHintedPromptRecord)]
        output_path = build_hinted_inference_path(
            benchmark_name=args.benchmark,
            model=spec.path,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            hint_fraction=fraction,
            data_root="data",
        )
        existing_rows = _load_hinted_rows(output_path) if output_path.exists() else []
        completed_hint_ids = {
            row.hint.hint_id
            for row in existing_rows
            if isinstance(row.hint.hint_id, str) and row.hint.hint_id
        }
        pending_rows = [
            row
            for row in typed_expanded_rows
            if row.hint_id in hint_id_set and row.hint_id not in completed_hint_ids
        ]
        if remaining_request_budget is not None:
            pending_rows = pending_rows[:remaining_request_budget]
            remaining_request_budget -= len(pending_rows)
        input_tokens = sum(
            _count_prompt_tokens(tokenizer=tokenizer, prompt=row.prompt)
            for row in pending_rows
        )
        max_output_tokens = len(pending_rows) * MAX_TOKENS

        total_pending_requests += len(pending_rows)
        total_input_tokens += input_tokens
        total_max_output_tokens += max_output_tokens
        per_fraction.append(
            {
                "hint_fraction": fraction,
                "output_path": str(output_path),
                "expanded_prompt_path": str(fraction_paths[fraction]),
                "pending_requests": len(pending_rows),
                "input_tokens": input_tokens,
                "max_output_tokens": max_output_tokens,
            }
        )

    input_cost = None
    input_cost = (total_input_tokens / 1_000_000.0) * pricing["input"]
    output_cost = None
    output_cost = (total_max_output_tokens / 1_000_000.0) * pricing["output"]
    max_total_cost = None
    if input_cost is not None or output_cost is not None:
        max_total_cost = (input_cost or 0.0) + (output_cost or 0.0)

    return {
        "model": spec.path,
        "backend": args.backend,
        "request_count": total_pending_requests,
        "max_requests": args.max_requests,
        "input_tokens": total_input_tokens,
        "max_output_tokens": total_max_output_tokens,
        "input_cost_per_million": pricing["input"],
        "output_cost_per_million": pricing["output"],
        "estimated_input_cost_usd": input_cost,
        "max_output_cost_usd": output_cost,
        "max_total_cost_usd": max_total_cost,
        "token_count_method": "exact tokenizer count over expanded prompts for pending requests",
        "per_fraction": per_fraction,
    }


def _print_dry_run_estimates(args: argparse.Namespace, models: list[ModelSpec]) -> None:
    if args.backend != "together-serverless":
        print("[generate_hinted] dry_run=true: exiting before launch", flush=True)
        return

    estimates = [_estimate_together_cost_for_model(args=args, spec=spec) for spec in models]
    print("[generate_hinted] together dry-run estimate", flush=True)
    print(json.dumps(estimates, indent=2), flush=True)


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


def _validate_together_pricing(models: list[ModelSpec], *, backend: str) -> None:
    if backend != "together-serverless":
        return
    for spec in models:
        _get_together_pricing_per_million(spec.path)


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
            "slurm_time_hours_override": SLURM_TIME_HOURS_OVERRIDE,
            "default_slurm_cpus_per_task": DEFAULT_SLURM_CPUS_PER_TASK,
            "default_slurm_mem_gb": DEFAULT_SLURM_MEM_GB,
            "eight_gpu_slurm_cpus_per_task": EIGHT_GPU_SLURM_CPUS_PER_TASK,
            "eight_gpu_slurm_mem_gb": EIGHT_GPU_SLURM_MEM_GB,
        },
        "job": {
            "executor": args.executor,
            "backend": args.backend,
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
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "dtype": args.dtype,
            "health_timeout": DEFAULT_HEALTH_TIMEOUT_SECONDS,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "generation": {
            "do_sample": sampling_params.get("do_sample"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "top_k": sampling_params.get("top_k"),
            "repetition_penalty": sampling_params.get("repetition_penalty"),
            "max_tokens": MAX_TOKENS,
            "max_connections": args.max_connections,
            "max_requests": args.max_requests,
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
                "backend": args.backend,
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
        for fraction in HINT_FRACTIONS:
            path = build_hinted_inference_path(
                benchmark_name=args.benchmark,
                model=spec.path,
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
            max_requests=args.max_requests,
            checkpoint_every=args.checkpoint_every,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            backend=args.backend,
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
            max_requests=args.max_requests,
            checkpoint_every=args.checkpoint_every,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            backend=args.backend,
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
    _validate_together_pricing(models, backend=args.backend)
    sampling_params_by_model = _load_sampling_params(models)
    _print_plan(args, models)

    if args.dry_run:
        _print_dry_run_estimates(args, models)
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
    --fractioner mask_word \
    --model Qwen/Qwen3.5-9B \
    --executor submitit \
    --cluster miso \
    --max-connections 360 \
    --num-gpus 8 \
    --checkpoint-every 1000

^260 works well, 400 untested
^ 260 is too much for Qwen3-32B but good for 14B
^260 is not enough for gemma 12B


NLP
python -m runs.generate_hinted \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --model Qwen/Qwen3.5-4B \
    --executor submitit \
    --cluster sphinx \
    --max-connections 48 \
    --num-gpus 1 \
    --checkpoint-every 500



TOGETHER
python -m runs.generate_hinted \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --model Qwen/Qwen3.5-397B-A17B \
    --max-connections 48 \
    --checkpoint-every 1000 \
    --backend together-serverless \
    --dry-run true


"max_total_cost_usd": 128.26805549999997, (for openai/gpt-oss-120b)
"max_total_cost_usd": 42.7560185, (for openai/gpt-oss-20b)
(for Qwen/Qwen3.5-397B-A17B)
"""
