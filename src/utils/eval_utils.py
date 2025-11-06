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


def get_valid_problem_ids(jsonl_paths: list[str]) -> set[str] | None:
    """Get intersection of problem IDs across multiple JSONL files with Example objects.

    Args:
        jsonl_paths: List of paths to JSONL files containing Example objects

    Returns:
        Set of IDs that appear in ALL files, or None if any file doesn't exist
        or is empty

    Example:
        >>> ids = get_valid_problem_ids([
        ...     "data/cot/gpqa.jsonl",
        ...     "data/solution/gpqa.jsonl"
        ... ])
        >>> print(f"Found {len(ids)} common problems")
    """
    if not jsonl_paths:
        return None

    id_sets = []

    for path in jsonl_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None

        ids = set()
        try:
            with open(path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        example = Example.from_dict(data)
                        ids.add(example.id)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"{path}:{line_num}: Skipping invalid line - {e}")

            if not ids:
                logger.warning(f"No valid IDs found in {path}")
                return None

            id_sets.append(ids)
            logger.info(f"Loaded {len(ids)} IDs from {path}")

        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    # Compute intersection of all ID sets
    valid_ids = id_sets[0]
    for id_set in id_sets[1:]:
        valid_ids &= id_set

    logger.info(f"Found {len(valid_ids)} problems common to all {len(jsonl_paths)} files")

    return valid_ids if valid_ids else None
