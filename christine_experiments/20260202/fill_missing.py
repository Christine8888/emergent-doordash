#!/usr/bin/env python3
"""Run evals iteratively until we have at least 1 correct trace per question."""

import os
import subprocess
import sys
from pathlib import Path

import dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from inspect_ai.log import list_eval_logs, read_eval_log

MODEL = "anthropic/claude-sonnet-4-5-20250929"
MAX_TOKENS = 16000
TEMPERATURE = 1.0
MAX_CONNECTIONS = 30
LOG_DIR = Path(__file__).parent / "logs"


def get_correct_ids(log_dir: Path, eval_name: str) -> set[str]:
    """Get all sample IDs that have at least one correct response."""
    correct_ids = set()

    log_files = list_eval_logs(str(log_dir))
    for log_info in log_files:
        if eval_name not in log_info.name:
            continue

        log = read_eval_log(log_info.name)
        if not log.samples:
            continue

        for sample in log.samples:
            if not sample.scores:
                continue
            for score in sample.scores.values():
                if score.value == "C" or score.value == 1 or score.value == 1.0:
                    correct_ids.add(str(sample.id))
                    break

    return correct_ids


def get_all_sample_ids(log_dir: Path, eval_name: str) -> set[str]:
    """Get all sample IDs seen across all logs for this eval."""
    all_ids = set()
    eval_key = eval_name.split("/")[-1].replace("_", "-")

    log_files = list_eval_logs(str(log_dir))
    for log_info in log_files:
        if eval_key not in log_info.name:
            continue
        log = read_eval_log(log_info.name)
        if log.samples:
            for s in log.samples:
                all_ids.add(str(s.id))
    return all_ids


def run_eval(eval_name: str, sample_ids: list[str] | None = None, is_multiple_choice: bool = True):
    """Run eval, optionally on specific sample IDs."""
    LOG_DIR.mkdir(exist_ok=True)

    cmd = [
        "inspect", "eval", eval_name,
        "--model", MODEL,
        "--epochs", "1",
        "--log-dir", str(LOG_DIR.absolute()),
        "--temperature", str(TEMPERATURE),
        "--max-tokens", str(MAX_TOKENS),
        "--max-connections", str(MAX_CONNECTIONS),
        "--display", "rich",
        "--retry-on-error=3",
        "--no-fail-on-error",
    ]

    if is_multiple_choice:
        cmd.extend([
            "--solver", "inspect_ai/multiple_choice",
            "-S", "cot=true",
        ])

    if sample_ids:
        cmd.extend(["--sample-id", ",".join(sample_ids)])

    print(f"\n{'='*60}")
    print(f"Running: {eval_name}")
    if sample_ids:
        print(f"Sample IDs: {len(sample_ids)} remaining")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd="/tmp", env=os.environ)
    return result.returncode


def fill_eval(eval_name: str, max_iterations: int = 10, is_multiple_choice: bool = True):
    """Run eval iteratively until all samples have at least 1 correct trace.

    Supports resume - checks existing logs first.
    """
    eval_key = eval_name.split("/")[-1]

    # Check existing logs for this eval
    all_ids = get_all_sample_ids(LOG_DIR, eval_name)
    correct_ids = get_correct_ids(LOG_DIR, eval_key)

    if all_ids:
        print(f"\n>>> Resuming {eval_name}")
        print(f"Found existing logs: {len(all_ids)} total samples, {len(correct_ids)} correct")
    else:
        # First run to get all sample IDs
        print(f"\n>>> Initial run for {eval_name}")
        run_eval(eval_name, sample_ids=None, is_multiple_choice=is_multiple_choice)
        all_ids = get_all_sample_ids(LOG_DIR, eval_name)
        correct_ids = get_correct_ids(LOG_DIR, eval_key)

    if not all_ids:
        print(f"Could not get sample IDs for {eval_name}")
        return

    print(f"Total samples: {len(all_ids)}")

    for iteration in range(1, max_iterations + 1):
        missing_ids = [id for id in all_ids if id not in correct_ids]

        print(f"\n>>> Iteration {iteration}: {len(correct_ids)}/{len(all_ids)} correct, {len(missing_ids)} remaining")

        if not missing_ids:
            print(f"All samples have correct traces!")
            break

        run_eval(eval_name, sample_ids=missing_ids, is_multiple_choice=is_multiple_choice)

        # Update counts
        correct_ids = get_correct_ids(LOG_DIR, eval_key)

    # Final stats
    correct_ids = get_correct_ids(LOG_DIR, eval_key)
    print(f"\n>>> Final: {len(correct_ids)}/{len(all_ids)} samples have correct traces")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True,
                        help="Eval to run (e.g., 'inspect_evals/commonsense_qa')")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max iterations to retry missing samples")
    parser.add_argument("--no-cot", action="store_true",
                        help="Don't use CoT solver override")
    args = parser.parse_args()

    fill_eval(args.eval, max_iterations=args.max_iterations,
              is_multiple_choice=not args.no_cot)


if __name__ == "__main__":
    main()
