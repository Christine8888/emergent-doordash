"""Baseline evaluations CLI."""

from concurrent.futures import ThreadPoolExecutor

from experiments.registry import get_all_evals, get_external_evals, get_internal_evals
from utils.submitit_utils import launch_baseline
from utils.submitit_defaults import SubmitConfig
from utils.setup import setup_logging

logger = setup_logging()

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 1),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1),
    ("Qwen/Qwen2.5-3B-Instruct", 1),
    ("Qwen/Qwen2.5-7B-Instruct", 1),
    ("Qwen/Qwen2.5-14B-Instruct", 2),
    ("Qwen/Qwen2.5-32B-Instruct", 4),
]

CONFIG = SubmitConfig(
    partition="sphinx",
    time_hours=10,
    mem_gb=64,
    cpus_per_task=4,
)


if __name__ == "__main__":
    import argparse

    all_evals = get_all_evals()

    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--eval", type=str, default="all",
                        help="Eval(s) to run: 'all', 'external', 'internal', single name, or comma-separated list (e.g. gpqa,arc,aime)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="./baseline_results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.eval == "all":
        evals_to_run = list(all_evals.keys())
    elif args.eval == "external":
        evals_to_run = list(get_external_evals().keys())
    elif args.eval == "internal":
        evals_to_run = list(get_internal_evals().keys())
    else:
        evals_to_run = [e.strip() for e in args.eval.split(",")]
        invalid = [e for e in evals_to_run if e not in all_evals]
        if invalid:
            raise ValueError(f"Unknown eval(s): {invalid}. Available: {list(all_evals.keys())}")

    logger.info(f"Running evals: {evals_to_run}")

    with ThreadPoolExecutor(max_workers=len(evals_to_run)) as executor:
        futures = [
            executor.submit(
                launch_baseline,
                eval_names=[eval_name],
                models=MODELS,
                results_dir=args.results_dir,
                config=CONFIG,
                epochs=args.epochs,
                limit=args.limit,
            )
            for eval_name in evals_to_run
        ]
        for future in futures:
            future.result()
