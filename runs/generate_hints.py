from __future__ import annotations

import argparse

from src.hint_types import HintType
from src.hint_generation import generate_hints

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hint records to JSONL.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", choices=[hint.value for hint in HintType], required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for hint generation.",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Enable model thinking at this effort level. Omit to disable thinking.",
    )
    parser.add_argument("--num-rollouts", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--problem-id",
        action="append",
        default=None,
        help="Generate only this problem id. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without making model calls or writing outputs.",
    )
    parser.add_argument(
        "--save-debug-responses",
        action="store_true",
        help="Save raw provider responses under data/debug/model_responses for every model call.",
    )
    parser.add_argument(
        "--hle-modality",
        choices=["all", "text-only", "with-images"],
        default="all",
        help="Only for --benchmark hle. Filter HLE problems by modality before applying --limit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    thinking_enabled = args.thinking_effort is not None
    generate_hints(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        first_model=args.model,
        num_rollouts=args.num_rollouts,
        limit=args.limit,
        problem_ids=args.problem_id,
        max_tokens=16384, # TODO
        temperature=1.0,
        dry_run=args.dry_run,
        thinking_enabled=thinking_enabled,
        thinking_effort=args.thinking_effort,
        concurrency=40,
        hle_modality=args.hle_modality,
        save_debug_responses=args.save_debug_responses,
    )


if __name__ == "__main__":
    main()

"""

python -m runs.generate_hints --hle-modality text-only --benchmark hle --hint-type basic_hint_hle --num-rollouts 1 --limit 5 --model gpt-5.5-2026-04-23 --thinking_effort low






python -m runs.generate_hints --hle-modality text-only --benchmark hle --hint-type answer_not_revealed --num-rollouts 1 --limit 5 --model claude-sonnet-4-6


"""
