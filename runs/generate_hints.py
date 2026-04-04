from __future__ import annotations

import argparse

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
    # if args.concurrency < 1:
    #     parser.error("--concurrency must be >= 1")

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
        max_tokens=32000,
        temperature=1.0,
        dry_run=args.dry_run == "true",
        concurrency=16,
    )


if __name__ == "__main__":
    main()

"""
python -m runs.generate_hints --benchmark aime2025_2026 --hint-type basic_hint --num-rollouts 10 --limit 65 --dry-run false 
python -m runs.generate_hints --benchmark aime2025_2026 --hint-type answer_not_revealed --num-rollouts 2 --limit 2 --dry-run false 


python -m runs.generate_hints --benchmark aime2025_2026 --hint-type bag_of_hints --num-rollouts 1 --limit 2 --dry-run false

"""


"""
(ed) suzeva@jagupard37:/nlp/scr/suzeva/projects/fresh-emergent-doordash$ python -m runs.generate_hints --benchmark aime2025_2026 --hint-type basic_hint --num-rollouts 10 --limit 65 --dry-run true 
[16:05:24] [hint_generation] dry_run benchmark=aime2025_2026 hint_type=basic_hint num_problems=60 rollouts=10 would_write=140 skipped=460 output=data/hint_generation/aime2025_2026/basic_hint.jsonl
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0009 missing_count=2 rollout_ids=[6, 7]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0010 missing_count=6 rollout_ids=[1, 3, 4, 5, 6, 8]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0011 missing_count=2 rollout_ids=[2, 3]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0013 missing_count=1 rollout_ids=[2]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0014 missing_count=3 rollout_ids=[0, 6, 7]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0015 missing_count=9 rollout_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0020 missing_count=5 rollout_ids=[0, 1, 6, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0026 missing_count=1 rollout_ids=[0]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0028 missing_count=8 rollout_ids=[0, 2, 3, 4, 5, 6, 7, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0029 missing_count=5 rollout_ids=[0, 1, 2, 5, 7]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0030 missing_count=9 rollout_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0039 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0040 missing_count=7 rollout_ids=[1, 3, 4, 5, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0041 missing_count=7 rollout_ids=[0, 1, 2, 3, 5, 6, 8]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0043 missing_count=4 rollout_ids=[1, 3, 5, 6]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0044 missing_count=2 rollout_ids=[0, 6]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0045 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0047 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0048 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0053 missing_count=4 rollout_ids=[0, 5, 6, 8]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0054 missing_count=3 rollout_ids=[2, 3, 6]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0057 missing_count=1 rollout_ids=[1]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0058 missing_count=1 rollout_ids=[5]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0059 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:05:24] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0060 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
(ed) suzeva@jagupard37:/nlp/scr/suzeva/projects/fresh-emergent-doordash$ 
before: 90,09



[16:11:31] [hint_generation] dry_run benchmark=aime2025_2026 hint_type=basic_hint num_problems=60 rollouts=10 would_write=113 skipped=487 output=data/hint_generation/aime2025_2026/basic_hint.jsonl
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0009 missing_count=2 rollout_ids=[6, 7]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0010 missing_count=3 rollout_ids=[3, 6, 8]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0013 missing_count=1 rollout_ids=[2]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0015 missing_count=7 rollout_ids=[1, 2, 3, 4, 5, 6, 8]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0020 missing_count=5 rollout_ids=[0, 1, 6, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0028 missing_count=6 rollout_ids=[0, 2, 3, 4, 6, 7]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0029 missing_count=5 rollout_ids=[0, 1, 2, 5, 7]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0030 missing_count=9 rollout_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0039 missing_count=9 rollout_ids=[0, 1, 3, 4, 5, 6, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0040 missing_count=7 rollout_ids=[1, 3, 4, 5, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0041 missing_count=3 rollout_ids=[1, 2, 8]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0043 missing_count=3 rollout_ids=[1, 5, 6]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0044 missing_count=1 rollout_ids=[6]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0045 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0047 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0048 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0053 missing_count=2 rollout_ids=[5, 8]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0059 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[16:11:31] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0060 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
After (1 attempt per question, need 140 more): 94,81
Accepted: 27
Rejected: 113

It cost $4.72 to get 27 nswers (0.18 per answer)
Need 113 more (~$20 dollars)
"""
