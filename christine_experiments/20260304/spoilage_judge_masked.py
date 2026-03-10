"""LLM-judge spoilage analysis for MASKED hints.

Instead of truncating to the first N% of words, this randomly masks
(1-fraction)% of words with [MASK] tokens, matching the masked prefill mode.

Since masking is stochastic, each (sample, fraction) pair is judged with
multiple random seeds (--mask-seeds) for variance reduction.

Usage:
    python spoilage_judge_masked.py --max-concurrent 50
    python spoilage_judge_masked.py --fractions 0.4,0.6,0.8 --mask-seeds 3

Resume: just re-run the same command; completed judgments are skipped.
"""

import asyncio
import json
import logging
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
You are evaluating whether a partially masked hint for a math/science problem makes the answer obvious.

This hint has had some words randomly replaced with [MASK]. The visible words show {fraction_pct}% of the original hint.

## Problem
{question}

## Correct Answer
{target}

## Masked Hint ({fraction_pct}% of words visible)
{masked_hint}

## Task
Looking ONLY at the visible (non-masked) words, is the final answer obvious? Not "could a clever student piece it together" — is it obvious at a glance from what is actually readable?

SPOILED — the answer is obvious, or requires only an extremely trivial jump (like adding two numbers). Examples:
- "the answer is 42" or "therefore x = 42"
- "therefore m + n = 42"
- "m = 700 and n = 31" when the answer is m+n=731 — adding two stated numbers is trivial
- "the correct answer is (D)"
- "Total optically active compounds = 3" when the choices include D)3

CLEAN — the answer is NOT obvious from the visible words. Examples:
- The answer value doesn't appear in the visible words
- The answer value appears but NOT as a conclusion — e.g., inside a formula like "\binom{62}{26}" where 62 is a parameter, not the answer
- The answer value appears surrounded by [MASK] with no readable context (e.g., "[MASK] [MASK] 42 [MASK]")
- The answer value appears in an intermediate computation, subscript, coefficient, or step number
- The answer appears in the problem statement, not the hint
- Determining the answer would require inference or piecing together fragments

Think step by step about what the visible words reveal, then give your final answer as <answer>SPOILED</answer> or <answer>CLEAN</answer>."""


# ── Masking (matches prefill.py logic) ─────────────────────────────────

def truncate_at_stop(text: str) -> str:
    if STOP_STRING not in text:
        return text
    return text[:text.index(STOP_STRING)].strip()


def get_masked_text(text: str, fraction: float, mask_token: str = "[MASK]",
                    seed: str | None = None) -> str:
    text = truncate_at_stop(text)
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]

    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask == 0:
        return text

    rng = random.Random(seed) if seed is not None else random
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join(mask_token if i in mask_indices else t for i, t in enumerate(tokens)).strip()


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
    """Key = "dataset|id|sample_idx|fraction|mask_seed"."""
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            key = f"{d['dataset']}|{d['id']}|{d['sample_idx']}|{d['fraction']}|{d['mask_seed']}"
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
    mask_seed: int,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    seed_str = f"{entry['id']}_{entry.get('sample_idx', 0)}_{fraction}_{mask_seed}"
    masked_hint = get_masked_text(entry["hint"], fraction, seed=seed_str)

    prompt = (JUDGE_PROMPT
        .replace("{question}", entry["question"])
        .replace("{target}", entry["target"])
        .replace("{fraction_pct}", str(int(fraction * 100)))
        .replace("{masked_hint}", masked_hint)
    )

    config = GenerateConfig(temperature=0.0, max_tokens=2048)

    async with semaphore:
        async with get_model(model_id, config=config) as model:
            response = await model.generate(input=[ChatMessageUser(content=prompt)])

    text = response.completion
    # Extract answer from <answer> tags
    import re as _re
    match = _re.search(r'<answer>\s*(SPOILED|CLEAN)\s*</answer>', text, _re.IGNORECASE)
    verdict = match.group(1).upper() if match else ("SPOILED" if "SPOILED" in text.upper() else "CLEAN")
    spoiled = verdict == "SPOILED"

    return {
        "dataset": dataset_name,
        "id": entry["id"],
        "sample_idx": entry.get("sample_idx", 0),
        "fraction": fraction,
        "mask_seed": mask_seed,
        "target": entry["target"],
        "spoiled": spoiled,
        "verdict": verdict,
    }


# ── Main ───────────────────────────────────────────────────────────────

async def main(
    fractions: list[float],
    mask_seeds: int,
    samples_per_fraction: int,
    max_concurrent: int,
    output_file: Path,
):
    checkpoint = load_checkpoint(output_file)
    logger.info(f"Loaded {len(checkpoint)} existing judgments from {output_file}")

    semaphore = asyncio.Semaphore(max_concurrent)

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
                for ms in range(mask_seeds):
                    key = f"{dataset_name}|{entry['id']}|{entry.get('sample_idx', 0)}|{fraction}|{ms}"
                    if key in checkpoint:
                        continue
                    tasks.append((entry, dataset_name, fraction, ms))

    logger.info(f"Total judgments to run: {len(tasks)}")
    if not tasks:
        logger.info("Nothing to do!")
        return

    completed = 0
    total = len(tasks)

    async def run_and_save(entry, dataset_name, fraction, mask_seed):
        nonlocal completed
        result = await judge_one(entry, dataset_name, fraction, mask_seed, JUDGE_MODEL, semaphore)
        append_result(output_file, result)
        completed += 1
        if completed % 100 == 0 or completed == total:
            logger.info(f"Progress: {completed}/{total}")
        return result

    results = await asyncio.gather(
        *[run_and_save(e, dn, f, ms) for e, dn, f, ms in tasks],
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.warning(f"{len(errors)} errors occurred")
        for e in errors[:5]:
            logger.warning(f"  {e}")

    print_summary(output_file)


def print_summary(output_file: Path):
    results = []
    with open(output_file) as f:
        for line in f:
            results.append(json.loads(line))

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

    parser = argparse.ArgumentParser(description="LLM-judge spoilage analysis (masked)")
    parser.add_argument("--fractions", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated prefill fractions")
    parser.add_argument("--mask-seeds", type=int, default=1,
                        help="Number of random mask seeds per (sample, fraction)")
    parser.add_argument("--samples-per-fraction", type=int, default=500,
                        help="Number of entries to sample per dataset")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Max concurrent API requests")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "spoilage_results_masked.jsonl"),
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
            mask_seeds=args.mask_seeds,
            samples_per_fraction=args.samples_per_fraction,
            max_concurrent=args.max_concurrent,
            output_file=output_path,
        ))
