from __future__ import annotations

import argparse

from src.datasets import HintType
from src.hint_generation import generate_hints

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hint records to JSONL.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[hint.value for hint in HintType], required=True)
    parser.add_argument("--generator-model", type=str, required=True)
    parser.add_argument("--num-rollouts", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--dry-run", choices=["true", "false"], required=True)
    parser.add_argument("--resume", choices=["true", "false"], required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    generate_hints(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        generator_model=args.generator_model,
        num_rollouts=args.num_rollouts,
        limit=args.limit,
        max_tokens=32000,
        temperature=1.0,
        dry_run=args.dry_run == "true",
        resume=args.resume == "true",
    )


if __name__ == "__main__":
    main()
"""
python runs/generate_hints.py --benchmark aime2025_2026 --hint-type truncated --generator-model claude-sonnet-4-6 --num-rollouts 1 --limit 1--temperature 1.0 --output-path data/aime2025_2026_truncated.jsonl --dry-run true --resume true
"""