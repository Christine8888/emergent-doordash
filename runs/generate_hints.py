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
        # first_model="claude-sonnet-4-6", # sonnet keeps revealing the answer accidentally so I'm switching to opus
        first_model="claude-opus-4-6",
        first_model_attempts=1,
        second_model="claude-opus-4-6",
        second_model_attempts=0,
        num_rollouts=args.num_rollouts,
        limit=args.limit,
        max_tokens=64000,
        temperature=1.0,
        dry_run=args.dry_run == "true",
        problem_ids=['aime2025_2026_0015'],
        thinking_enabled=True,
        thinking_effort="medium",
        concurrency=6,
    )


if __name__ == "__main__":
    main()

"""
python -m runs.generate_hints --benchmark aime2025_2026 --hint-type basic_hint --num-rollouts 10 --limit 60 --dry-run false 





python -m runs.generate_hints --benchmark aime2025_2026 --hint-type answer_not_revealed --num-rollouts 2 --limit 2 --dry-run false 
python -m runs.generate_hints --benchmark aime2025_2026 --hint-type bag_of_hints --num-rollouts 1 --limit 2 --dry-run false
"""
