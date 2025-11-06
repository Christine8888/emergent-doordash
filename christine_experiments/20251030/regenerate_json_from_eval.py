#!/usr/bin/env python3
"""
Regenerate JSON result files from .eval files.

Usage:
    python regenerate_json_from_eval.py <eval_dir> --scorer SCORER [--metric METRIC] [--output_dir DIR]

Example:
    python regenerate_json_from_eval.py results/gpqa/0shot/Qwen2.5-0.5B-Instruct --scorer gpqa_scorer
    python regenerate_json_from_eval.py results/gpqa/0shot/Qwen2.5-0.5B-Instruct --scorer gpqa_scorer --metric accuracy
"""

import sys
import os
from pathlib import Path
from inspect_ai.log import read_eval_log
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs, compute_pass_at_k
import json
import argparse


def regenerate_json(eval_file_path, output_dir=None, scorer=None, metric=None):
    """Regenerate JSON from an .eval file."""
    eval_path = Path(eval_file_path)

    # Extract the suffix after the last underscore (e.g., j4UaD8TtpeC3bM7gWBhbog)
    stem = eval_path.stem  # filename without extension
    # Get everything after the last underscore
    suffix = stem.split('_')[-1]
    json_filename = f"{suffix}.json"

    # Determine output directory
    if output_dir:
        output_path = Path(output_dir) / json_filename
    else:
        output_path = eval_path.parent / json_filename

    print(f"Processing: {eval_path.name} -> {json_filename}")

    # Read the eval log
    log = read_eval_log(str(eval_path))

    # Extract scores
    results = extract_scores_from_log(log)

    # Compute bootstrap statistics if requested
    if scorer:
        bootstrap_metric = {'scorer': scorer}
        if metric:
            bootstrap_metric['metric'] = metric

        # Check if we have multiple epochs - try multiple ways to detect it
        n_epochs = None
        if hasattr(log.eval, 'epochs') and log.eval.epochs:
            n_epochs = log.eval.epochs

        # Also check the actual number of samples per ID
        if log.samples:
            sample_ids = {}
            for sample in log.samples:
                sample_ids[sample.id] = sample_ids.get(sample.id, 0) + 1
            if sample_ids:
                n_epochs_actual = max(sample_ids.values())
                if n_epochs is None:
                    n_epochs = n_epochs_actual
                elif n_epochs != n_epochs_actual:
                    print(f"  WARNING: log.eval.epochs={n_epochs} but actual epochs from samples={n_epochs_actual}")
                    n_epochs = n_epochs_actual

        if n_epochs is None:
            n_epochs = 1

        print(f"  Epochs detected: {n_epochs}")

        if n_epochs > 1:
            results["manual_bootstrap"] = compute_bootstrap_over_epochs(log, bootstrap_metric)
            results["pass_at_k"] = compute_pass_at_k(log, bootstrap_metric)
        else:
            print(f"  WARNING: Only 1 epoch found, skipping bootstrap/pass@k calculations")

    # Write to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  -> Wrote {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate JSON files from .eval files")
    parser.add_argument("eval_dir", type=str, help="Directory containing .eval files")
    parser.add_argument("--scorer", type=str, default=None,
                       help="Scorer name for bootstrap calculation (e.g., 'gpqa_scorer', 'hle_scorer')")
    parser.add_argument("--metric", type=str, default=None,
                       help="Metric name for bootstrap calculation (default: 'accuracy')")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for JSON files (default: same as .eval file)")

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory {eval_dir} does not exist")
        sys.exit(1)

    # Find all .eval files recursively
    eval_files = list(eval_dir.glob("**/*.eval"))

    if not eval_files:
        print(f"No .eval files found in {eval_dir}")
        sys.exit(1)

    print(f"Found {len(eval_files)} .eval files")
    if args.scorer:
        metric_str = args.metric if args.metric else 'accuracy (default)'
        print(f"Bootstrap metric: scorer='{args.scorer}', metric='{metric_str}'")
    else:
        print(f"Bootstrap metric: None")
    print("=" * 80)

    # Process each file
    for eval_file in eval_files:
        try:
            regenerate_json(eval_file, args.output_dir, args.scorer, args.metric)
        except Exception as e:
            print(f"  ERROR processing {eval_file.name}: {e}")

    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
