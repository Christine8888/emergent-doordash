#!/usr/bin/env python3
"""Extract correct traces from eval log files."""

import json
import re
from pathlib import Path
from inspect_ai.log import list_eval_logs, read_eval_log

LOG_DIR = Path(__file__).parent / "logs"
OUTPUT_DIR = Path(__file__).parent / "traces"


def is_correct(sample) -> bool:
    """Check if a sample has a correct score."""
    if not sample.scores:
        return False
    for score in sample.scores.values():
        val = score.value
        if val == "C" or val == 1 or val == 1.0 or val is True:
            return True
    return False


def extract_response(sample) -> str:
    """Extract the model's response from a sample."""
    # Prefer output.completion as it's the cleanest
    if sample.output and sample.output.completion:
        return sample.output.completion

    # Fallback: get the last assistant message
    for msg in reversed(sample.messages):
        if hasattr(msg, 'role') and msg.role == 'assistant':
            if hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Handle ContentText blocks
                    texts = []
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            texts.append(block.text)
                        elif isinstance(block, str):
                            texts.append(block)
                    return '\n'.join(texts)
    return ""


def extract_question(sample) -> str:
    """Extract the question/input from a sample."""
    if isinstance(sample.input, str):
        return sample.input
    elif isinstance(sample.input, list):
        # Handle chat message format
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
    # Get user messages
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


def process_log_file(log_path: str) -> list[dict]:
    """Process a single log file and extract correct traces."""
    log = read_eval_log(log_path)

    if not log.samples:
        return []

    traces = []
    for sample in log.samples:
        if not is_correct(sample):
            continue

        response = extract_response(sample)
        if not response.strip():
            continue

        # Get target as string
        target = sample.target
        if isinstance(target, list):
            target = target[0] if target else ""

        trace = {
            "id": str(sample.id),
            "question": extract_question(sample),
            "target": str(target),
            "response": response,
            "hint": response,  # For external evals, hint = response
            "prompt": extract_prompt(sample),
            "sample_idx": sample.epoch,
            "metadata": {
                "model": log.eval.model,
                "eval": log.eval.task,
                "epoch": sample.epoch,
            }
        }

        # Include choices if available
        if sample.choices:
            trace["choices"] = sample.choices

        traces.append(trace)

    return traces


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--eval", type=str, default=None,
                        help="Filter by eval name")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all log files
    log_files = list_eval_logs(str(log_dir))
    print(f"Found {len(log_files)} log files in {log_dir}")

    # Group by eval name
    eval_traces = {}

    for log_info in log_files:
        log_path = log_info.name

        # Filter if requested
        if args.eval and args.eval not in log_path:
            continue

        print(f"Processing: {log_path}")
        traces = process_log_file(log_path)

        if not traces:
            print(f"  No correct traces found")
            continue

        # Extract eval name from first trace
        eval_name = traces[0]["metadata"]["eval"]
        # Clean up eval name for filename
        eval_name_clean = eval_name.replace("/", "_").replace("inspect_evals_", "")

        if eval_name_clean not in eval_traces:
            eval_traces[eval_name_clean] = []
        eval_traces[eval_name_clean].extend(traces)

        print(f"  Extracted {len(traces)} correct traces")

    # Write output files
    for eval_name, traces in eval_traces.items():
        output_path = output_dir / f"{eval_name}.jsonl"

        # Deduplicate by (id, epoch)
        seen = set()
        unique_traces = []
        for trace in traces:
            key = (trace["id"], trace["sample_idx"])
            if key not in seen:
                seen.add(key)
                unique_traces.append(trace)

        with open(output_path, "w") as f:
            for trace in unique_traces:
                f.write(json.dumps(trace) + "\n")

        print(f"Wrote {len(unique_traces)} traces to {output_path}")


if __name__ == "__main__":
    main()
