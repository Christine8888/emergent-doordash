from __future__ import annotations

import argparse
import ast
import json

from src.hint_types import HintType
from src.hint_generation import generate_hints

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hint records to JSONL.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[hint.value for hint in HintType], required=True)
    parser.add_argument("--num-rollouts", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--dry-run", choices=["true", "false"], required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_hints(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        first_model="claude-sonnet-4-6",
        # first_model="gpt-5.5-2026-04-23",
        first_model_attempts=1,
        num_rollouts=args.num_rollouts,
        limit=args.limit,
        max_tokens=8192,
        temperature=1.0,
        dry_run=args.dry_run == "true",
        thinking_enabled="false",
        # thinking_effort="low",
        concurrency=40,
    )


if __name__ == "__main__":
    main()

"""

python -m runs.generate_hints --benchmark hle --hint-type basic_hint_hle --num-rollouts 1 --limit 200 --dry-run false

python -m runs.generate_hints --benchmark hle --hint-type basic_hint_hle --num-rollouts 1 --dry-run false --limit 2500





python -m runs.generate_hints --benchmark hle --hint-type answer_not_revealed --num-rollouts 1 --limit 1 --dry-run false

"""
