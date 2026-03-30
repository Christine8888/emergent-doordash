"""Build merged spoilage labels (LLM + regex) as JSONL.

This script computes regex spoilage labels for all source hint rows using:
1) masked hints; and
2) truncated-prefix hints (matching christine_experiments/20260304/spoilage_judge.py).

It joins Christine's LLM-judge masked spoilage results when available, keyed by
dataset, id, sample_idx, fraction, mask_seed.

Output rows are intended as a lightweight lookup index for downstream viewers.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_RESULTS = PROJECT_ROOT / "christine_experiments/20260304/spoilage_results_masked.jsonl"
DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "spoilage_results_llm_regex_masked.jsonl"

DATASET_TO_SOURCE = {
    "aime_solution": PROJECT_ROOT / "christine_experiments/data/solution/aime.jsonl",
    "gpqa_solution": PROJECT_ROOT / "christine_experiments/data/solution/gpqa.jsonl",
    "aime_cot": PROJECT_ROOT / "christine_experiments/data/cot/aime.jsonl",
    "gpqa_cot": PROJECT_ROOT / "christine_experiments/data/cot/gpqa.jsonl",
}

STOP_STRING = "ANSWER:"
MASK_TOKEN = "[MASK]"


def parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
    return None


def infer_rationalize_from_prompt(prompt: str) -> bool:
    return "HINT: The answer is" in prompt


def extract_source_metadata(row: dict) -> tuple[str | None, bool]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    model = row.get("model") or metadata.get("model")
    raw_rationalize = row.get("rationalize", metadata.get("rationalize"))
    rationalize = parse_bool(raw_rationalize)
    if rationalize is None:
        rationalize = infer_rationalize_from_prompt(str(row.get("prompt", "")))
    return (None if model is None else str(model), bool(rationalize))


def truncate_at_stop(text: str, stop_string: str = STOP_STRING) -> str:
    if stop_string not in text:
        return text
    return text[: text.index(stop_string)].strip()


def split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    tokens = re.split(r"(\s+)", text)
    word_indices = [i for i, tok in enumerate(tokens) if tok.strip()]
    return tokens, word_indices


def get_masked_text(text: str, fraction: float, seed: str | None) -> str:
    text = truncate_at_stop(text)
    tokens, word_indices = split_preserving_whitespace(text)
    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask <= 0:
        return text

    rng = random.Random(seed) if seed is not None else random
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join(MASK_TOKEN if i in mask_indices else tok for i, tok in enumerate(tokens)).strip()


def get_truncated_prefix_text(text: str, fraction: float) -> str:
    """Match spoilage_judge.py: truncate at ANSWER:, then keep first fraction of words."""
    text = truncate_at_stop(text)
    if fraction >= 1.0:
        return text

    tokens, word_indices = split_preserving_whitespace(text)
    if not word_indices:
        return text

    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text

    last_idx = word_indices[num_words - 1]
    return "".join(tokens[: last_idx + 1]).strip()


def target_is_spoiled(prefix_text: str, target: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(target) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, prefix_text))


def load_source_entries() -> dict[tuple[str, str, int], dict]:
    by_key: dict[tuple[str, str, int], dict] = {}
    duplicate_keys = 0

    for dataset, path in DATASET_TO_SOURCE.items():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                hint = row.get("hint", "")
                if not str(hint).strip():
                    continue

                key = (dataset, str(row["id"]), int(row.get("sample_idx", 0)))
                if key in by_key:
                    duplicate_keys += 1
                    continue
                by_key[key] = row

    if duplicate_keys:
        print(f"[load_source_entries] ignored {duplicate_keys} duplicate source key(s)")
    print(f"[load_source_entries] loaded {len(by_key)} unique source rows")
    return by_key


def load_llm_rows(llm_results_path: Path) -> tuple[dict[tuple[str, str, int, float, int], dict], list[float], list[int]]:
    llm_by_key: dict[tuple[str, str, int, float, int], dict] = {}
    fractions: set[float] = set()
    mask_seeds: set[int] = set()

    with llm_results_path.open() as f:
        for line in f:
            row = json.loads(line)
            dataset = str(row["dataset"])
            row_id = str(row["id"])
            sample_idx = int(row.get("sample_idx", 0))
            fraction = float(row["fraction"])
            mask_seed = int(row.get("mask_seed", 0))
            key = (dataset, row_id, sample_idx, fraction, mask_seed)
            llm_by_key[key] = row
            fractions.add(fraction)
            mask_seeds.add(mask_seed)

    if not fractions:
        fractions = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    if not mask_seeds:
        mask_seeds = {0}
    return llm_by_key, sorted(fractions), sorted(mask_seeds)


def build_rows(
    source_by_key: dict[tuple[str, str, int], dict],
    llm_by_key: dict[tuple[str, str, int, float, int], dict],
    fractions: list[float],
    mask_seeds: list[int],
) -> list[dict]:
    merged_rows: list[dict] = []

    for (dataset, row_id, sample_idx), source_row in source_by_key.items():
        target = str(source_row.get("target", ""))
        hint = str(source_row["hint"])
        source_model, source_rationalize = extract_source_metadata(source_row)

        for fraction in fractions:
            for mask_seed in mask_seeds:
                llm_key = (dataset, row_id, sample_idx, float(fraction), int(mask_seed))
                llm_row = llm_by_key.get(llm_key)

                mask_seed_str = f"{row_id}_{sample_idx}_{fraction}_{mask_seed}"
                masked_hint = get_masked_text(hint, fraction=fraction, seed=mask_seed_str)
                regex_spoiled_masked = target_is_spoiled(masked_hint, target)

                truncated_hint = get_truncated_prefix_text(hint, fraction=fraction)
                regex_spoiled_truncated = target_is_spoiled(truncated_hint, target)

                merged_rows.append(
                    {
                        "dataset": dataset,
                        "id": row_id,
                        "sample_idx": sample_idx,
                        "fraction": float(fraction),
                        "mask_seed": int(mask_seed),
                        "target": target,
                        "source_model": source_model,
                        "source_rationalize": source_rationalize,
                        "llm_spoiled": None if llm_row is None else bool(llm_row["spoiled"]),
                        "llm_verdict": None if llm_row is None else str(llm_row.get("verdict", "")),
                        "llm_judged": llm_row is not None,
                        # Backward-compatible alias for existing viewers.
                        "regex_spoiled": bool(regex_spoiled_masked),
                        "regex_spoiled_masked": bool(regex_spoiled_masked),
                        "regex_spoiled_truncated": bool(regex_spoiled_truncated),
                    }
                )

    return merged_rows


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build merged spoilage labels (LLM masked + regex masked/truncated)."
    )
    parser.add_argument("--llm-results", type=Path, default=DEFAULT_LLM_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    if not args.llm_results.exists():
        raise FileNotFoundError(f"LLM results file not found: {args.llm_results}")

    source_by_key = load_source_entries()
    llm_by_key, fractions, mask_seeds = load_llm_rows(args.llm_results)
    print(
        f"[load_llm_rows] loaded {len(llm_by_key)} LLM rows "
        f"across {len(fractions)} fractions and {len(mask_seeds)} mask seed(s)"
    )
    rows = build_rows(source_by_key, llm_by_key, fractions, mask_seeds)
    write_jsonl(rows, args.out)

    print(f"[build] wrote {len(rows)} rows to {args.out}")
    llm_covered = sum(1 for r in rows if r["llm_judged"])
    print(f"[build] llm-covered rows: {llm_covered} / {len(rows)}")


if __name__ == "__main__":
    main()
