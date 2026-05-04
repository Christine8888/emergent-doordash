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
    generate_hints(
        benchmark_name=args.benchmark,
        hint_type=args.hint_type,
        # first_model="claude-sonnet-4-6",
        first_model="claude-opus-4-7",
        # first_model="gpt-5.5-2026-04-23",
        first_model_attempts=1,
        num_rollouts=args.num_rollouts,
        limit=args.limit,
        max_tokens=16384, # TODO
        temperature=1.0,
        dry_run=args.dry_run == "true",
        thinking_enabled=True,
        thinking_effort="low",
        concurrency=40,
        hle_modality=args.hle_modality,
    )


if __name__ == "__main__":
    main()

"""

python -m runs.generate_hints--hle-modality text-only --benchmark hle --hint-type    --num-rollouts 1 --limit 1 --dry-run false

python -m runs.generate_hints --hle-modality text-only --benchmark hle --hint-type basic_hint_hle --num-rollouts 1 --dry-run false --limit 2500





python -m runs.generate_hints --hle-modality text-only --benchmark hle --hint-type answer_not_revealed --num-rollouts 1 --limit 5 --dry-run false


[00:01:28] [hint_generation] request benchmark=hle hint_type=answer_not_revealed problem_id=hle_00001 rollout_id=0 attempt=1 model=claude-sonnet-4-6 images=1
[00:01:29] [hint_generation][WARN] query_error benchmark=hle problem_id=hle_00001 rollout_id=0 attempt=1 model=claude-sonnet-4-6 error=Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}, 'request_id': 'req_011CabRGDCag6NLQreUqrHeB'}
"""
