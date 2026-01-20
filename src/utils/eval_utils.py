"""Minimal utilities for running evaluations."""

import asyncio
import importlib
import os
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
from tqdm.asyncio import tqdm

from inspect_ai import eval
from inspect_ai.dataset import Sample
from utils.setup import setup_env, setup_logging
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs, compute_pass_at_k
from evals.example import Example

setup_env()
logger = setup_logging()


def create_base_parser(default_log_dir: str):
    """Create argument parser with common arguments.

    Args:
        default_log_dir: Default directory for saving logs

    Returns:
        ArgumentParser with standard eval arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vllm/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_connections", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--base_port", type=int, default=9000, help="Base port for vLLM server")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fewshot", type=int, default=0, help="Number of fewshot examples")
    parser.add_argument("--hint_fraction", type=float, default=1.0, help="Fraction of hint to provide")
    return parser


def setup_vllm_env(base_port: int):
    """Set vLLM environment variables.

    Args:
        base_port: Base port for vLLM server
    """
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{base_port}/v1"
    os.environ["VLLM_API_KEY"] = "local"


def check_output_exists(filename: str):
    """Check if output file exists and exit if it does.

    Args:
        filename: Path to output file
    """
    if os.path.exists(filename):
        logger.info(f"Output file {filename} already exists. Skipping evaluation.")
        exit(0)


def run_eval(
    task,
    args,
    output_filename: str,
    scorer_name: str | None = None,
    extra_metadata: dict = None,
):
    """Run evaluation and save results.

    Args:
        task: Inspect Task object (already configured)
        args: Parsed command-line arguments
        output_filename: Path to save results JSON
        scorer_name: Name of scorer for bootstrap metrics (e.g., 'gpqa_scorer').
                     Only needed if epochs > 1.
        extra_metadata: Additional metadata to include in output

    Returns:
        EvalLog from inspection, or None if output already exists
    """
    # Check if output already exists
    if os.path.exists(output_filename):
        logger.info(f"Output file {output_filename} already exists. Skipping evaluation.")
        return None

    # Build metadata
    metadata = {
        "timeout": args.timeout,
    }
    if args.max_tokens:
        metadata["max_tokens"] = args.max_tokens
    if args.limit:
        metadata["limit"] = args.limit
    if extra_metadata:
        metadata.update(extra_metadata)

    # Run evaluation
    eval_kwargs = {
        "model": args.model,
        "log_dir": args.log_dir,
        "epochs": args.epochs,
        "limit": args.limit,
        "max_connections": args.max_connections,
        "display": "rich",
        "metadata": metadata,
    }
    if args.max_tokens:
        eval_kwargs["max_tokens"] = args.max_tokens

    log = eval(task, **eval_kwargs)

    # Extract and save results
    results = extract_scores_from_log(log[0])

    if args.epochs > 1 and scorer_name:
        bootstrap_metric = {'scorer': scorer_name, 'metric': 'accuracy'}
        results["manual_bootstrap"] = compute_bootstrap_over_epochs(log[0], bootstrap_metric)
        results["pass_at_k"] = compute_pass_at_k(log[0], bootstrap_metric)

    with open(output_filename, "w") as f:
        json.dump(results, f)

    return log[0]


def get_valid_problem_ids(jsonl_paths: list[str]) -> dict[str, set[int]] | None:
    """Get valid (problem_id, sample_idx) pairs across JSONL files.

    Args:
        jsonl_paths: List of paths to JSONL files containing Example objects

    Returns:
        Dictionary mapping problem IDs to sets of valid sample_idx values,
        or None if any file doesn't exist or has no valid entries.
        Only includes entries with non-empty string hints.
    """
    if not jsonl_paths:
        return None

    file_data: list[dict[str, set[int]]] = []

    for path in jsonl_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None

        path_data: dict[str, set[int]] = {}
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                example = Example.from_dict(data)

                if not example.has_valid_hint():
                    continue

                if example.id not in path_data:
                    path_data[example.id] = set()
                path_data[example.id].add(example.sample_idx)

        if not path_data:
            logger.warning(f"No valid entries found in {path}")
            return None

        total_samples = sum(len(indices) for indices in path_data.values())
        logger.info(f"Loaded {total_samples} valid samples for {len(path_data)} problems from {path}")
        file_data.append(path_data)

    result = file_data[0]
    for other in file_data[1:]:
        common_ids = set(result.keys()) & set(other.keys())
        result = {
            id: result[id] | other[id]
            for id in common_ids
        }

    total_samples = sum(len(indices) for indices in result.values())
    logger.info(f"Found {total_samples} valid samples for {len(result)} problems common to all {len(jsonl_paths)} files")

    return result if result else None
