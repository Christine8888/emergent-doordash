from __future__ import annotations

import argparse

from src.hint_types import HintType
from src.hint_generation import generate_hints

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hint records to JSONL.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[hint.value for hint in HintType], required=True)
    parser.add_argument("--first-model", type=str, required=True)
    parser.add_argument("--second-model", type=str, required=True)
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
        first_model=args.first_model,
        first_model_attempts=3,
        second_model=args.second_model,
        second_model_attempts=3,
        num_rollouts=args.num_rollouts,
        limit=args.limit,
        max_tokens=32000,
        temperature=1.0,
        dry_run=args.dry_run == "true",
    )


if __name__ == "__main__":
    main()
"""
dry run
python -m runs.generate_hints --benchmark aime2025_2026 --hint-type masked --first-model claude-sonnet-4-6 --second-model claude-opus-4-6 --num-rollouts 1 --limit 1000 --dry-run true

generating tiny bit of hints
python -m runs.generate_hints --benchmark aime2025_2026 --hint-type masked --first-model claude-sonnet-4-6 --second-model claude-opus-4-6 --num-rollouts 2 --limit 2 --dry-run false

python -m runs.generate_hints --benchmark aime2025_2026 --hint-type basic_hint --first-model claude-sonnet-4-6 --second-model claude-opus-4-6 --num-rollouts 2 --limit 2 --dry-run false


"""
