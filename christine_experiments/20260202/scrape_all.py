#!/usr/bin/env python3
"""Scrape correct traces for all external evals.

Runs all evals iteratively until each sample has at least 1 correct trace,
or max iterations reached. Resumes from partial .jsonl output files.

Usage:
    python scrape_all.py                    # Run all evals, max 10 iterations each
    python scrape_all.py --max-iterations 5 # Custom max iterations
    python scrape_all.py --eval mmlu        # Run specific eval only
"""

import json
import os
import subprocess
from pathlib import Path

import dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from inspect_ai.log import list_eval_logs, read_eval_log

# Config
MODEL = "anthropic/claude-sonnet-4-5-20250929"
MAX_CONNECTIONS = 30
TEMPERATURE = 1.0

LOG_DIR = Path(__file__).parent / "logs"
OUTPUT_DIR = Path(__file__).parent / "traces"

# Evals to run (excluding bbeh, niah, mbpp, ifeval)
# Format: (eval_name, solver_override, task_args)
#   solver_override: None = use default, "mc_cot" = multiple_choice with cot=true
#   task_args: dict of -T args to pass
EXTERNAL_EVALS = [
    ("inspect_evals/mmlu_0_shot", None, {"cot": "true"}),  # has native cot support
    ("inspect_evals/commonsense_qa", "mc_cot", {}),
    ("inspect_evals/arc_easy", "mc_cot", {}),
    ("inspect_evals/arc_challenge", "mc_cot", {}),
    ("inspect_evals/hellaswag", "mc_cot", {}),
    ("inspect_evals/piqa", "mc_cot", {}),
    ("inspect_evals/winogrande", "mc_cot", {}),
    ("inspect_evals/bbh", None, {}),  # has its own reasoning solver
]


def eval_to_filename(eval_name: str) -> str:
    """Convert eval name to output filename."""
    return eval_name.split("/")[-1].replace("_", "-") + ".jsonl"


def eval_to_log_key(eval_name: str) -> str:
    """Convert eval name to key for matching log files."""
    return eval_name.split("/")[-1].replace("_", "-")


def load_existing_ids(output_path: Path) -> set[str]:
    """Load sample IDs that already have correct traces in output file."""
    if not output_path.exists():
        return set()

    ids = set()
    with open(output_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                ids.add(str(data["id"]))
            except:
                pass
    return ids


def extract_response(sample) -> str:
    """Extract the model's response from a sample."""
    if sample.output and sample.output.completion:
        return sample.output.completion
    for msg in reversed(sample.messages):
        if hasattr(msg, 'role') and msg.role == 'assistant':
            if hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    texts = []
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            texts.append(block.text)
                    return '\n'.join(texts)
    return ""


def extract_question(sample) -> str:
    """Extract the question/input from a sample."""
    if isinstance(sample.input, str):
        return sample.input
    elif isinstance(sample.input, list):
        texts = []
        for msg in sample.input:
            if hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    texts.append(msg.content)
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            texts.append(block.text)
        return '\n'.join(texts)
    return str(sample.input)


def extract_prompt(sample) -> str:
    """Extract the full prompt from messages."""
    if not sample.messages:
        return extract_question(sample)
    texts = []
    for msg in sample.messages:
        if hasattr(msg, 'role') and msg.role == 'user':
            if hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    texts.append(msg.content)
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            texts.append(block.text)
    return '\n'.join(texts)


def is_correct(sample) -> bool:
    """Check if sample has correct score."""
    if not sample.scores:
        return False
    for score in sample.scores.values():
        if score.value == "C" or score.value == 1 or score.value == 1.0:
            return True
    return False


def process_logs_and_append(eval_name: str, output_path: Path, existing_ids: set[str]) -> set[str]:
    """Process all logs for eval, append new correct traces to output, return all correct IDs."""
    log_key = eval_to_log_key(eval_name)
    new_ids = set()

    log_files = list_eval_logs(str(LOG_DIR))
    for log_info in log_files:
        if log_key not in log_info.name:
            continue

        try:
            log = read_eval_log(log_info.name)
        except Exception as e:
            print(f"  Warning: Could not read {log_info.name}: {e}")
            continue

        if not log.samples:
            continue

        traces_to_write = []
        for sample in log.samples:
            sample_id = str(sample.id)

            # Skip if we already have this one
            if sample_id in existing_ids:
                new_ids.add(sample_id)
                continue

            if not is_correct(sample):
                continue

            response = extract_response(sample)
            if not response.strip():
                continue

            target = sample.target
            if isinstance(target, list):
                target = target[0] if target else ""

            trace = {
                "id": sample_id,
                "question": extract_question(sample),
                "target": str(target),
                "response": response,
                "hint": response,
                "prompt": extract_prompt(sample),
                "sample_idx": sample.epoch,
                "metadata": {
                    "model": log.eval.model,
                    "eval": log.eval.task,
                }
            }
            if sample.choices:
                trace["choices"] = sample.choices

            traces_to_write.append(trace)
            new_ids.add(sample_id)

        # Append new traces
        if traces_to_write:
            with open(output_path, "a") as f:
                for trace in traces_to_write:
                    f.write(json.dumps(trace) + "\n")

    return existing_ids | new_ids


def get_all_sample_ids_from_logs(eval_name: str) -> set[str]:
    """Get all sample IDs from logs for this eval."""
    log_key = eval_to_log_key(eval_name)
    all_ids = set()

    log_files = list_eval_logs(str(LOG_DIR))
    for log_info in log_files:
        if log_key not in log_info.name:
            continue
        try:
            log = read_eval_log(log_info.name)
            if log.samples:
                for s in log.samples:
                    all_ids.add(str(s.id))
        except:
            pass
    return all_ids


def run_eval(eval_name: str, sample_ids: list[str] | None = None,
              solver_override: str | None = None, task_args: dict | None = None):
    """Run eval, optionally on specific sample IDs."""
    LOG_DIR.mkdir(exist_ok=True)

    cmd = [
        "inspect", "eval", eval_name,
        "--model", MODEL,
        "--epochs", "1",
        "--log-dir", str(LOG_DIR.absolute()),
        "--max-connections", str(MAX_CONNECTIONS),
        "--temperature", str(TEMPERATURE),
        "--display", "rich",
        "--max-retries", "10",
        "--retry-on-error=10",
        "--no-fail-on-error",
    ]

    # Add solver override for CoT
    if solver_override == "mc_cot":
        cmd.extend(["--solver", "inspect_ai/multiple_choice", "-S", "cot=true"])

    # Add task args
    if task_args:
        for key, value in task_args.items():
            cmd.extend(["-T", f"{key}={value}"])

    if sample_ids:
        cmd.extend(["--sample-id", ",".join(sample_ids)])

    print(f"\n{'='*60}")
    print(f"Running: {eval_name}")
    if sample_ids:
        print(f"Samples: {len(sample_ids)} remaining")
    print(f"{'='*60}\n")

    subprocess.run(cmd, cwd="/tmp", env=os.environ)


def scrape_eval(eval_name: str, max_iterations: int,
                solver_override: str | None = None, task_args: dict | None = None):
    """Scrape a single eval until all samples have correct traces."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / eval_to_filename(eval_name)

    print(f"\n{'#'*60}")
    print(f"# {eval_name}")
    print(f"{'#'*60}")

    # Load existing correct IDs from output file
    correct_ids = load_existing_ids(output_path)
    if correct_ids:
        print(f"Resuming: {len(correct_ids)} traces already in {output_path.name}")

    # Check if we have logs to get total sample count
    all_ids = get_all_sample_ids_from_logs(eval_name)

    for iteration in range(1, max_iterations + 1):
        # If no logs yet, do initial run
        if not all_ids:
            print(f"\nIteration {iteration}: Initial run")
            run_eval(eval_name, solver_override=solver_override, task_args=task_args)
            all_ids = get_all_sample_ids_from_logs(eval_name)
            correct_ids = process_logs_and_append(eval_name, output_path, correct_ids)
        else:
            # Process any new logs
            correct_ids = process_logs_and_append(eval_name, output_path, correct_ids)
            missing_ids = [id for id in all_ids if id not in correct_ids]

            print(f"\nIteration {iteration}: {len(correct_ids)}/{len(all_ids)} complete, {len(missing_ids)} remaining")

            if not missing_ids:
                print(f"Done! All samples have correct traces.")
                break

            run_eval(eval_name, sample_ids=missing_ids, solver_override=solver_override, task_args=task_args)
            correct_ids = process_logs_and_append(eval_name, output_path, correct_ids)

    # Final stats
    if all_ids:
        print(f"\nFinal: {len(correct_ids)}/{len(all_ids)} samples have traces in {output_path.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=str, default=None,
                        help="Run specific eval only (e.g., 'mmlu' or 'inspect_evals/mmlu_0_shot')")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max iterations per eval (default: 10)")
    args = parser.parse_args()

    evals_to_run = EXTERNAL_EVALS
    if args.eval:
        evals_to_run = [(name, solver, targs) for name, solver, targs in EXTERNAL_EVALS if args.eval in name]
        if not evals_to_run:
            print(f"No matching eval for '{args.eval}'")
            print(f"Available: {[e[0] for e in EXTERNAL_EVALS]}")
            return

    for eval_name, solver_override, task_args in evals_to_run:
        scrape_eval(eval_name, args.max_iterations, solver_override=solver_override, task_args=task_args)

    print(f"\n{'='*60}")
    print("All done! Output files in:", OUTPUT_DIR)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
