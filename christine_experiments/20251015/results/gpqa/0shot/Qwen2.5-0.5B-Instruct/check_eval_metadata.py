#!/usr/bin/env python3
"""
Script to check if hint_fraction metadata is stored in .eval files.

Usage:
    python check_eval_metadata.py <path_to_eval_file>

Example:
    python check_eval_metadata.py results/gpqa/0shot/Qwen2.5-0.5B-Instruct/2025-10-22T14-10-52-07-00_gpqa-diamond_j4UaD8TtpeC3bM7gWBhbog.eval
"""

import sys
from inspect_ai.log import read_eval_log
import json

def check_eval_metadata(eval_file_path):
    """Check what metadata is stored in an eval file."""
    print(f"Reading: {eval_file_path}")
    print("=" * 80)

    log = read_eval_log(eval_file_path)

    print("\n=== log.eval.task_args ===")
    print(f"Type: {type(log.eval.task_args)}")
    if log.eval.task_args:
        print(f"Keys: {list(log.eval.task_args.keys())}")
        for key, value in log.eval.task_args.items():
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            print(f"  Value: {value}")

            # Check if it's a dict with fraction
            if isinstance(value, dict):
                if 'fraction' in value:
                    print(f"  *** FOUND FRACTION: {value['fraction']} ***")
                print(f"  Dict keys: {list(value.keys())}")
    else:
        print("task_args is None or empty")

    print("\n" + "=" * 80)
    print("=== log.eval.task_args_passed ===")
    print(f"Type: {type(log.eval.task_args_passed)}")
    print(f"Value: {log.eval.task_args_passed}")

    print("\n" + "=" * 80)
    print("=== log.eval.metadata ===")
    print(f"Type: {type(log.eval.metadata)}")
    print(f"Value: {log.eval.metadata}")

    print("\n" + "=" * 80)
    print("=== log.eval.task_attribs ===")
    print(f"Type: {type(log.eval.task_attribs)}")
    print(f"Value: {log.eval.task_attribs}")

    print("\n" + "=" * 80)
    print("=== Summary ===")
    print(f"Model: {log.eval.model}")
    print(f"Task: {log.eval.task}")
    print(f"Total samples: {log.results.total_samples}")

    # Try to extract hint_fraction if it exists
    hint_fraction = None
    if log.eval.task_args and 'prefill_config' in log.eval.task_args:
        prefill_config = log.eval.task_args['prefill_config']
        if isinstance(prefill_config, dict) and 'fraction' in prefill_config:
            hint_fraction = prefill_config['fraction']

    if hint_fraction is not None:
        print(f"\n✓ HINT_FRACTION FOUND: {hint_fraction}")
    else:
        print(f"\n✗ HINT_FRACTION NOT FOUND IN EVAL FILE")

    return hint_fraction

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    eval_file = sys.argv[1]
    check_eval_metadata(eval_file)

