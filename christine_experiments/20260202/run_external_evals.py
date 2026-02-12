#!/usr/bin/env python3
"""Run external evals with multiple epochs to collect correct traces.

For multiple-choice evals, overrides solver to use cot=True for reasoning traces.
"""

import os
import subprocess
import sys
from pathlib import Path

# Load environment variables from project root
import dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")

# External evals to run (excluding bbeh, niah, mbpp)
# Format: (eval_name, is_multiple_choice, extra_task_args)
EXTERNAL_EVALS = [
    # Multiple choice evals - override solver with cot=True
    ("inspect_evals/mmlu_0_shot", True, {}),
    ("inspect_evals/commonsense_qa", True, {}),
    ("inspect_evals/arc_easy", True, {}),
    ("inspect_evals/arc_challenge", True, {}),
    ("inspect_evals/hellaswag", True, {}),
    ("inspect_evals/piqa", True, {}),
    ("inspect_evals/winogrande", True, {}),
    ("inspect_evals/bbh", True, {}),
]

MODEL = "anthropic/claude-sonnet-4-5-20250929"
EPOCHS = 1
MAX_TOKENS = 16000
TEMPERATURE = 1.0
MAX_CONNECTIONS = 30  # concurrent API calls
LOG_DIR = Path(__file__).parent / "logs"


def run_eval(eval_name: str, is_multiple_choice: bool, task_args: dict):
    """Run a single eval with the given configuration."""
    LOG_DIR.mkdir(exist_ok=True)

    cmd = [
        "inspect", "eval", eval_name,
        "--model", MODEL,
        "--epochs", str(EPOCHS),
        "--log-dir", str(LOG_DIR.absolute()),
        "--temperature", str(TEMPERATURE),
        "--max-tokens", str(MAX_TOKENS),
        "--max-connections", str(MAX_CONNECTIONS),
        "--display", "rich",
        "--retry-on-error=3",
        "--no-fail-on-error",
    ]

    # Override solver for multiple choice evals to get CoT reasoning
    if is_multiple_choice:
        cmd.extend([
            "--solver", "inspect_ai/multiple_choice",
            "-S", "cot=true",
        ])

    for key, value in task_args.items():
        cmd.extend(["-T", f"{key}={value}"])

    print(f"\n{'='*60}")
    print(f"Running: {eval_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run from /tmp to avoid local task name conflicts
    result = subprocess.run(cmd, cwd="/tmp", env=os.environ)
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default=None,
                        help="Run specific eval (e.g., 'mmlu_0_shot')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    evals_to_run = EXTERNAL_EVALS
    if args.eval:
        evals_to_run = [(name, mc, ta) for name, mc, ta in EXTERNAL_EVALS
                        if args.eval in name]
        if not evals_to_run:
            print(f"No matching eval found for '{args.eval}'")
            print(f"Available: {[name for name, _, _ in EXTERNAL_EVALS]}")
            sys.exit(1)

    for eval_name, is_mc, task_args in evals_to_run:
        if args.dry_run:
            print(f"Would run: {eval_name} (mc={is_mc}) with {task_args}")
        else:
            returncode = run_eval(eval_name, is_mc, task_args)
            if returncode != 0:
                print(f"Warning: {eval_name} exited with code {returncode}")


if __name__ == "__main__":
    main()
