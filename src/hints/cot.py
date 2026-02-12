#!/usr/bin/env python3
"""
Sample eval problems until all have correct solutions, saving CoT as hints.

Outputs Example format with:
- id, question, target, response, hint (hint = response for CoT data)

Usage:
    python sample_cot.py --eval gpqa --output-file results.jsonl
    python sample_cot.py --eval aime --output-file results.jsonl
    python sample_cot.py --eval math --output-file results.jsonl --split train
"""
import sys
import asyncio
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hints.sample_utils import collect_samples, create_base_parser


def identity(x: str) -> str:
    """Identity function for response_to_hint (hint = response for CoT)."""
    return x


async def main():
    parser = create_base_parser("Sample eval problems until all have correct solutions")
    args, _ = parser.parse_known_args()

    await collect_samples(
        args=args,
        response_to_hint=identity,  # hint = response for CoT
        format_fn=None,  # use eval_config.format_prompt
        extract_fn=None,  # use eval_config.extract_answer
    )


if __name__ == "__main__":
    asyncio.run(main())