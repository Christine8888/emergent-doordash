"""Baseline evaluations CLI."""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add src/ to path so imports work from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from experiments.registry import get_all_evals, get_external_evals, get_internal_evals
from utils.submitit_utils import launch_baseline
from utils.model_config import QWEN3_MODELS, QWEN25_MODELS, LLAMA_MODELS, GEMMA_MODELS
from utils.setup import setup_logging

logger = setup_logging()

# Load HF token based on current user
import getpass
_HF_TOKEN_PATHS = {
    "cye": "/sphinx/u/cye/emergent-doordash/hf.tok",
    "suzeva": "/nlp/scr/suzeva/hf.tok",
}
_token_path = _HF_TOKEN_PATHS.get(getpass.getuser(), _HF_TOKEN_PATHS["cye"])
with open(_token_path, 'r') as f:
    os.environ['HF_TOKEN'] = f.read().strip()


if __name__ == "__main__":
    import argparse

    all_evals = get_all_evals()

    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--eval", type=str, default="all",
                        help="Eval(s) to run: 'all', 'external', 'internal', single name, or comma-separated list (e.g. gpqa,arc,aime)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Eval(s) to exclude: comma-separated list (e.g. hle,math)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model(s) to run: comma-separated list (e.g. meta-llama/Llama-3.1-70B-Instruct). Default: all models")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="./baseline",
                        help="Directory to write Inspect logs and summary JSONs (default: ./baseline)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max tokens to generate per sample (default: 8192)")
    parser.add_argument("--nodelist", type=str, default=None,
                        help="Override node list (e.g. 'sphinx[10-11]' for sphinx-only)")
    parser.add_argument("--partition", type=str, default=None,
                        help="Override partition (e.g. 'sphinx' for sphinx-only)")
    parser.add_argument("--debug", action="store_true", help="Enable Inspect HTTP debug logging")
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

    # Build model list
    all_models = QWEN3_MODELS + QWEN25_MODELS + LLAMA_MODELS + GEMMA_MODELS
    if args.model:
        model_names = [m.strip() for m in args.model.split(",")]
        models_to_run = [m for m in all_models if m.path in model_names]
        not_found = [name for name in model_names if not any(m.path == name for m in all_models)]
        if not_found:
            available = [m.path for m in all_models]
            raise ValueError(f"Unknown model(s): {not_found}. Available: {available}")
    else:
        models_to_run = all_models

    # Apply nodelist/partition overrides (e.g., for sphinx-only)
    if args.nodelist or args.partition:
        from dataclasses import replace
        overrides = {}
        if args.nodelist:
            overrides["nodelist"] = args.nodelist
        if args.partition:
            overrides["partitions"] = args.partition
        models_to_run = [replace(m, **overrides) for m in models_to_run]

    logger.info(f"Running models: {[m.path for m in models_to_run]}")
    if args.nodelist:
        logger.info(f"Nodelist override: {args.nodelist}")
    if args.partition:
        logger.info(f"Partition override: {args.partition}")

    with ThreadPoolExecutor(max_workers=len(evals_to_run)) as executor:
        futures = [
            executor.submit(
                launch_baseline,
                eval_names=[eval_name],
                models=models_to_run,
                results_dir=args.results_dir,
                epochs=args.epochs,
                limit=args.limit,
                max_tokens=args.max_tokens,
                poll_interval=30,
                debug=args.debug,
            )
            for eval_name in evals_to_run
        ]
        for future in futures:
            future.result()


"""
cd /afs/cs.stanford.edu/u/suzeva/emergent-doordash/christine_experiments/20251113
python baseline_evals.py --eval bbh --model meta-llama/Llama-3.1-70B-Instruct

running at: 14409840

tail -f /afs/cs.stanford.edu/u/suzeva/emergent-doordash/christine_experiments/20251113/submitit_logs/14409840_0_log.err

squeue -j 14409840        # Check if still running
sacct -j 14409840         # Check completion status


(older one: 14407660)

NEW: 14426770
"""