"""LLM-judge spoilage analysis: does a hint prefix reveal the answer?

Uses Claude Haiku 4.5 via Inspect AI's model backend to judge whether
a hint prefix directly states/reveals the final answer (not just makes
the problem easier).

Usage:
    python spoilage_judge.py --max-concurrent 50
    python spoilage_judge.py --fractions 0.6,0.8,1.0 --samples-per-fraction 100

Resume: just re-run the same command; completed judgments are skipped.
"""

import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from inspect_ai.model import get_model, ChatMessageUser, GenerateConfig

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────

DATASETS = {
    "aime_solution": _PROJECT_ROOT / "christine_experiments/data/solution/aime.jsonl",
    "gpqa_solution": _PROJECT_ROOT / "christine_experiments/data/solution/gpqa.jsonl",
    "aime_cot": _PROJECT_ROOT / "christine_experiments/data/cot/aime.jsonl",
    "gpqa_cot": _PROJECT_ROOT / "christine_experiments/data/cot/gpqa.jsonl",
}

JUDGE_MODEL = "anthropic/claude-haiku-4-5-20251001"
STOP_STRING = "ANSWER:"
SEED = 42

JUDGE_PROMPT = """\
You are evaluating whether a partial hint for a math/science problem gives away the final answer.

This hint is meant to help a student solve the problem — it should guide their thinking, show them the approach, and provide intermediate steps. That's fine and expected. What we do NOT want is for the hint to directly tell the student what the final answer is, so they can just read it off without finishing the problem themselves.

## Problem
{question}

## Correct Answer
{target}

## Partial Hint (first {fraction_pct}% of the hint)
{hint_prefix}

## Task
Could a student reading this hint determine the final answer without doing any additional reasoning? In other words, does the hint directly state or give away the final answer?

SPOILED means the hint explicitly reveals the final answer — e.g.:
- "the answer is 42"
- "therefore m + n = 42"
- "this gives us 42" as the final conclusion
- For multiple choice: "the correct answer is (D)" or "option D is right"

CLEAN means the student would still need to do some work to reach the answer, even with this hint. Examples of CLEAN hints:
- Showing the approach or method without reaching the final result
- Intermediate calculations that haven't been combined into the final answer yet
- Numbers that happen to equal the answer but appear in a different context (e.g., the answer is 6 and the hint mentions x^6 or "Step 6")
- The answer appearing as part of the problem statement itself

Respond with exactly one word: SPOILED or CLEAN"""


# ── Hint truncation (matches prefill.py logic) ─────────────────────────

def truncate_at_stop(text: str) -> str:
    if STOP_STRING not in text:
        return text
    return text[:text.index(STOP_STRING)].strip()


def get_prefix(text: str, fraction: float) -> str:
    text = truncate_at_stop(text)
    if fraction >= 1.0:
        return text
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text
    last_idx = word_indices[num_words - 1]
    return "".join(tokens[:last_idx + 1]).strip()


# ── Data loading ───────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("hint", "").strip():
                continue
            entries.append(d)
    return entries


def sample_entries(entries: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(entries) <= n:
        return entries
    return rng.sample(entries, n)


# ── Checkpoint ─────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    """Load checkpoint: {key: judgment} where key = "dataset|id|sample_idx|fraction"."""
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            key = f"{d['dataset']}|{d['id']}|{d['sample_idx']}|{d['fraction']}"
            results[key] = d
    return results


def append_result(path: Path, result: dict):
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


# ── Judge ──────────────────────────────────────────────────────────────

async def judge_one(
    entry: dict,
    dataset_name: str,
    fraction: float,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    hint_prefix = get_prefix(entry["hint"], fraction)

    prompt = JUDGE_PROMPT.format(
        question=entry["question"],
        target=entry["target"],
        fraction_pct=int(fraction * 100),
        hint_prefix=hint_prefix,
    )

    config = GenerateConfig(temperature=0.0, max_tokens=10)

    async with semaphore:
        async with get_model(model_id, config=config) as model:
            response = await model.generate(input=[ChatMessageUser(content=prompt)])

    verdict = response.completion.strip().upper()
    spoiled = "SPOILED" in verdict

    return {
        "dataset": dataset_name,
        "id": entry["id"],
        "sample_idx": entry.get("sample_idx", 0),
        "fraction": fraction,
        "target": entry["target"],
        "spoiled": spoiled,
        "verdict": verdict,
    }


# ── Main ───────────────────────────────────────────────────────────────

async def main(
    fractions: list[float],
    samples_per_fraction: int,
    max_concurrent: int,
    output_file: Path,
):
    checkpoint = load_checkpoint(output_file)
    logger.info(f"Loaded {len(checkpoint)} existing judgments from {output_file}")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Build all tasks
    tasks = []
    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Skipping {dataset_name}: {dataset_path} not found")
            continue

        entries = load_dataset(dataset_path)
        sampled = sample_entries(entries, samples_per_fraction, SEED)
        logger.info(f"{dataset_name}: {len(entries)} total, sampled {len(sampled)}")

        for fraction in fractions:
            for entry in sampled:
                key = f"{dataset_name}|{entry['id']}|{entry.get('sample_idx', 0)}|{fraction}"
                if key in checkpoint:
                    continue
                tasks.append((entry, dataset_name, fraction))

    logger.info(f"Total judgments to run: {len(tasks)}")
    if not tasks:
        logger.info("Nothing to do!")
        return

    # Run with progress
    completed = 0
    total = len(tasks)

    async def run_and_save(entry, dataset_name, fraction):
        nonlocal completed
        result = await judge_one(entry, dataset_name, fraction, JUDGE_MODEL, semaphore)
        append_result(output_file, result)
        completed += 1
        if completed % 100 == 0 or completed == total:
            logger.info(f"Progress: {completed}/{total}")
        return result

    results = await asyncio.gather(
        *[run_and_save(e, dn, f) for e, dn, f in tasks],
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.warning(f"{len(errors)} errors occurred")
        for e in errors[:5]:
            logger.warning(f"  {e}")

    # Print summary
    print_summary(output_file)


def print_summary(output_file: Path):
    results = []
    with open(output_file) as f:
        for line in f:
            results.append(json.loads(line))

    # Group by dataset and fraction
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["dataset"], r["fraction"])].append(r["spoiled"])

    datasets = sorted(set(r["dataset"] for r in results))
    fractions = sorted(set(r["fraction"] for r in results))

    print(f"\n{'Dataset':<20}", end="")
    for f in fractions:
        print(f"  f={f:.2f}", end="")
    print()
    print("-" * (20 + len(fractions) * 8))

    for ds in datasets:
        print(f"{ds:<20}", end="")
        for f in fractions:
            key = (ds, f)
            if key in groups:
                rate = sum(groups[key]) / len(groups[key])
                print(f"  {rate:.1%} ", end="")
            else:
                print(f"  {'n/a':>5} ", end="")
        print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="LLM-judge spoilage analysis")
    parser.add_argument("--fractions", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated prefill fractions")
    parser.add_argument("--samples-per-fraction", type=int, default=500,
                        help="Number of entries to sample per dataset")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Max concurrent API requests")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "spoilage_results.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--summary-only", action="store_true",
                        help="Just print summary from existing results")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_summary(output_path)
    else:
        fractions = [float(f) for f in args.fractions.split(",")]
        asyncio.run(main(
            fractions=fractions,
            samples_per_fraction=args.samples_per_fraction,
            max_concurrent=args.max_concurrent,
            output_file=output_path,
        ))
