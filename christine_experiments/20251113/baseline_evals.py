"""Baseline evaluations CLI."""

import os
from concurrent.futures import ThreadPoolExecutor

from experiments.registry import get_all_evals, get_external_evals, get_internal_evals
from utils.submitit_utils import launch_baseline
from utils.model_config import QWEN3_MODELS, QWEN25_MODELS, LLAMA_MODELS, GEMMA_MODELS
from utils.setup import setup_logging

logger = setup_logging()

with open('/sphinx/u/cye/emergent-doordash/hf.tok', 'r') as f:
    os.environ['HF_TOKEN'] = f.read().strip()


if __name__ == "__main__":
    import argparse

    all_evals = get_all_evals()

    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--eval", type=str, default="all",
                        help="Eval(s) to run: 'all', 'external', 'internal', single name, or comma-separated list (e.g. gpqa,arc,aime)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Eval(s) to exclude: comma-separated list (e.g. hle,math)")
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

    if args.exclude:
        exclude_set = {e.strip() for e in args.exclude.split(",")}
        evals_to_run = [e for e in evals_to_run if e not in exclude_set]

    logger.info(f"Running evals: {evals_to_run}")

    with ThreadPoolExecutor(max_workers=len(evals_to_run)) as executor:
        futures = [
            executor.submit(
                launch_baseline,
                eval_names=[eval_name],
                models=QWEN3_MODELS + QWEN25_MODELS + LLAMA_MODELS + GEMMA_MODELS,
                results_dir=args.results_dir,
                epochs=args.epochs,
                limit=args.limit,
                poll_interval=30,
            )
            for eval_name in evals_to_run
        ]
        for future in futures:
            future.result()
