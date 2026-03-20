"""Compute and plot answer spoilage rate curves for hint datasets.

For each dataset, this script:
1) judges every hint at each hint fraction;
2) verifies each question id has the same number of hints;
3) computes spoilage by averaging per-id rates so all questions are weighted equally.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint

DATASETS = [
    ("data/solution/aime.jsonl", "AIME solution"),
    ("data/solution/gpqa.jsonl", "GPQA solution"),
    ("data/cot/aime.jsonl", "AIME cot"),
    ("data/cot/gpqa.jsonl", "GPQA cot"),
]

FRACTIONS = np.arange(0.0, 1.01, 0.05)
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "christine_experiments"

# Map christine's dataset name -> our label
CHRISTINE_DATASET_MAP = {
    "aime_solution": "AIME solution",
    "gpqa_solution": "GPQA solution",
    "aime_cot":      "AIME cot",
    "gpqa_cot":      "GPQA cot",
}
CHRISTINE_RESULTS_PATH = BASE / "20260304/spoilage_results_masked.jsonl"

def _truncate_at_stop_string(text: str, stop_string: str, fraction) -> str:
    """Truncate text before stop_string, warning if not found."""
    if stop_string not in text:
        if not fraction == 0.0:
            raise ValueError(f"stop_string '{stop_string}' not found in text: {text}")
        return text
    return text[:text.index(stop_string)].strip()


def _split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    """Split text into tokens preserving whitespace, return tokens and word indices."""
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    return tokens, word_indices


def get_masked_text(text: str, fraction: float = 0.5, mask_token: str = "[MASK]",
                    stop_string: str = "ANSWER:", seed: int | str | None = None) -> str:
    """Mask random words, showing only a fraction of them. Truncates at stop_string.

    Args:
        seed: If provided, use a seeded RNG for reproducible masking.
              If None, uses the global RNG (legacy behavior).
    """
    text = _truncate_at_stop_string(text, stop_string, fraction)
    tokens, word_indices = _split_preserving_whitespace(text)

    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask == 0:
        return text

    rng = random.Random(seed) if seed is not None else random
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join(mask_token if i in mask_indices else t for i, t in enumerate(tokens)).strip()


def target_is_spoiled(prefix_text, target):
    """Check if target appears as a standalone token in the text."""
    pattern = r'(?<![A-Za-z0-9])' + re.escape(target) + r'(?![A-Za-z0-9])'
    return bool(re.search(pattern, prefix_text))


def load_entries(path):
    """Load entries from a JSONL file, dropping any rows with an empty hint field."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))

    filtered = [e for e in entries if e.get("hint", "").strip()]
    n_dropped = len(entries) - len(filtered)
    if n_dropped:
        print(f"[load_entries] {path}: dropped {n_dropped} row(s) with empty hint")

    return filtered


def analyze_duplicates(label, entries):
    """Print full details of every (id, sample_idx) duplicate pair."""
    seen = {}  # (id, sample_idx) -> first entry
    duplicates = []  # list of (key, first_entry, later_entry)

    for entry in entries:
        key = (entry["id"], entry.get("sample_idx"))
        if key in seen:
            duplicates.append((key, seen[key], entry))
        else:
            seen[key] = entry

    if not duplicates:
        print(f"[{label}] No duplicate (id, sample_idx) pairs found.")
        return

    print(f"\n{'='*80}")
    print(f"[{label}] {len(duplicates)} duplicate (id, sample_idx) pair(s)")
    print(f"{'='*80}")

    all_keys = sorted({key for key, _, _ in duplicates})
    for key in all_keys:
        pairs = [(a, b) for k, a, b in duplicates if k == key]
        print(f"\n--- id={key[0]!r}  sample_idx={key[1]!r}  ({len(pairs)} extra occurrence(s)) ---")
        # Collect all versions: first seen + all later occurrences
        versions = [seen[key]] + [b for _, b in pairs]
        for i, v in enumerate(versions):
            print(f"  [version {i}]")
            for field, val in v.items():
                val_str = str(val)
                # Truncate long fields but show enough to compare
                display = val_str if len(val_str) <= 200 else val_str[:200] + "...(truncated)"
                print(f"    {field}: {display}")
        # Highlight fields that differ across versions
        all_fields = list(dict.fromkeys(k for v in versions for k in v))
        differing = [f for f in all_fields if len({str(v.get(f)) for v in versions}) > 1]
        if differing:
            print(f"  ** Differing fields: {differing}")
        else:
            print(f"  ** All fields identical (exact duplicate)")

    print(f"\n[{label}] Summary: {len(duplicates)} duplicate rows across {len(all_keys)} unique (id, sample_idx) keys")


def validate_hint_counts_per_id(entries, dataset_label):
    """Verify each id has the same number of hint samples."""
    id_counts = Counter(entry["id"] for entry in entries)
    unique_counts = sorted(set(id_counts.values()))

    if len(unique_counts) == 1:
        print(
            f"[{dataset_label}] all {len(id_counts)} ids have "
            f"{unique_counts[0]} hints each."
        )
        return

    print(f"[{dataset_label}] WARNING: uneven hints-per-id counts: {unique_counts}")
    preview = sorted(id_counts.items(), key=lambda kv: (kv[1], kv[0]))[:10]
    print(f"[{dataset_label}] example id->count pairs: {preview}")
    print(
        f"WARNING: {dataset_label}: uneven hint counts per id; cannot ensure equal question weighting."
    )


def compute_spoilage_curve(entries, fractions):
    """Compute spoilage by averaging per-id spoilage rates (equal question weight)."""
    per_id_entries = defaultdict(list)
    for entry in entries:
        per_id_entries[entry["id"]].append(entry)

    rates = []
    for f in fractions:
        # if f == 0.0:
            # rates.append(0.0)
            # continue

        per_id_rates = []
        for id_entries in per_id_entries.values():
            spoiled = 0
            total = len(id_entries)
            for entry in id_entries:
                hint = entry["hint"]
                target = str(entry["target"])
                # Deterministic seed for reproducible masked spoilage curves.
                mask_seed = f"{entry['id']}_{entry.get('sample_idx', 0)}_{f}_0"
                if not hint:
                    raise ValueError(f'hint is empty: {pprint(entry)}')
                masked_hint = get_masked_text(
                    hint, fraction=f, stop_string="ANSWER:", seed=mask_seed
                )
                if target_is_spoiled(masked_hint, target):
                    spoiled += 1
            per_id_rates.append(spoiled / total if total > 0 else 0.0)

        rates.append(float(np.mean(per_id_rates)) if per_id_rates else 0.0)
    return rates


def load_christine_spoilage_curves():
    """Load Christine's pre-judged (LLM) spoilage results and aggregate per-id.

    Returns:
        dict mapping label -> (fractions_array, rates_list)
    """
    rows = []
    with open(CHRISTINE_RESULTS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    # group by (dataset, fraction) -> {id -> [spoiled, ...]}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = CHRISTINE_DATASET_MAP.get(row["dataset"])
        key = (label, row["fraction"])
        grouped[key][row["id"]].append(row["spoiled"])

    # Validate rollout counts per id at fraction=1.0 (all fractions share the same ids)
    for label in CHRISTINE_DATASET_MAP.values():
        per_id = grouped.get((label, 1.0), {})
        if not per_id:
            continue
        id_counts = Counter(len(v) for v in per_id.values())
        unique_counts = sorted(id_counts)
        if len(unique_counts) == 1:
            print(f"[christine/{label}] all {len(per_id)} ids have {unique_counts[0]} rollouts each.")
        else:
            print(f"[christine/{label}] WARNING: uneven rollout counts: {dict(id_counts)}")
            preview = sorted((k, len(v)) for k, v in per_id.items() if len(v) != unique_counts[-1])[:10]
            print(f"[christine/{label}] example id->count pairs: {preview}")

    # collect all fractions and labels
    all_fractions = sorted({f for _, f in grouped})
    all_labels = list(CHRISTINE_DATASET_MAP.values())

    results = {}
    for label in all_labels:
        rates = []
        for f in all_fractions:
            per_id = grouped[(label, f)]
            if f == 0.0:
                rates.append(0.0)
                continue
            # average spoilage rate per id (equal question weight)
            id_rates = [sum(v) / len(v) for v in per_id.values()]
            rates.append(float(np.mean(id_rates)))
        # unique (id, sample_idx) pairs judged for this dataset
        id_sample_pairs = frozenset(
            (row["id"], row["sample_idx"])
            for row in rows
            if CHRISTINE_DATASET_MAP.get(row["dataset"]) == label
        )
        results[label] = (np.array(all_fractions), rates, id_sample_pairs)

    return results


def main():
    results = {}      # label -> rates
    n_samples = {}    # label -> total hint count
    for rel_path, label in DATASETS:
        path = BASE / rel_path
        entries = load_entries(path)
        print(f"Loaded {len(entries)} entries for {label}")
        validate_hint_counts_per_id(entries, label)
        rates = compute_spoilage_curve(entries, FRACTIONS)
        results[label] = rates
        n_samples[label] = len(entries)

    # Print table
    # header = f"{'Dataset':<20}" + "".join(f"{'f=' + str(f):<12}" for f in FRACTIONS)
    # print("\n" + header)
    # print("-" * len(header))
    # for label, rates in results.items():
    #     row = f"{label:<20}"
    #     for f in FRACTIONS:
    #         idx = int(round(f / 0.05))
    #         row += f"{rates[idx]:<12.4f}"
    #     print(row)

    christine_results = load_christine_spoilage_curves()

    # Pre-load all entries per label so we can filter to matched subset
    all_entries = {}
    for rel_path, label in DATASETS:
        all_entries[label] = load_entries(BASE / rel_path)

    # Plot: one subplot per dataset, regex vs LLM judge
    labels = [label for _, label in DATASETS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    color_regex       = "steelblue"
    color_regex_match = "mediumseagreen"
    color_llm         = "tomato"

    for ax, label in zip(axes.flat, labels):
        rates_regex = results[label]
        n_regex = n_samples[label]
        ax.plot(FRACTIONS, rates_regex,
                color=color_regex, marker="o", markersize=3,
                label=f"regex judge (n={n_regex})")

        if label in christine_results:
            fracs_llm, rates_llm, llm_pairs = christine_results[label]
            n_llm = len(llm_pairs)
            ax.plot(fracs_llm, rates_llm,
                    color=color_llm, marker="s", markersize=3, linestyle="--",
                    label=f"LLM judge (n={n_llm})")

            # Regex curve restricted to the same (id, sample_idx) pairs as LLM judge
            # Dedupe on (id, sample_idx) so matched set is exactly llm_pairs
            seen_keys = set()
            matched_entries = []
            for e in all_entries[label]:
                key = (e["id"], e.get("sample_idx"))
                if key in llm_pairs and key not in seen_keys:
                    matched_entries.append(e)
                    seen_keys.add(key)
            if matched_entries:
                rates_matched = compute_spoilage_curve(matched_entries, FRACTIONS)
                ax.plot(FRACTIONS, rates_matched,
                        color=color_regex_match, marker="^", markersize=3, linestyle=":",
                        label=f"regex judge matched (n={len(matched_entries)})")

        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel("Hint Fraction")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spoilage Rate")

    fig.suptitle("Answer Spoilage Rate: Regex vs LLM Judge (masked hints)", fontsize=13)
    plt.tight_layout()
    out_path = REPO_ROOT / "suze_experiments/20260320/spoilage_comparison.pdf"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.show()


def inspect_aime_solution_discrepancy(fraction=0.9):
    """Print examples where LLM says spoiled but regex doesn't at a given fraction for AIME solution."""
    label = "AIME solution"

    entries = load_entries(BASE / "data/solution/aime.jsonl")
    entries_by_key = {(e["id"], e.get("sample_idx")): e for e in entries}

    rows_llm = []
    with open(CHRISTINE_RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if CHRISTINE_DATASET_MAP.get(r["dataset"]) == label and r["fraction"] == fraction:
                rows_llm.append(r)

    print(f"\n{'='*80}")
    print(f"AIME solution: cases where LLM=spoiled, regex=NOT spoiled at fraction={fraction}")
    print(f"{'='*80}")

    n_shown = 0
    for row in rows_llm:
        if not row["spoiled"]: # only check examples that christine marked as spoiled
            continue
        key = (row["id"], row["sample_idx"])
        entry = entries_by_key.get(key)
        if entry is None:
            continue
        fraction = row["fraction"]
        mask_seed = f"{entry['id']}_{entry.get('sample_idx', 0)}_{fraction}_0"
        masked = get_masked_text(entry["hint"], fraction=fraction, stop_string="ANSWER:", seed=mask_seed)
        if not target_is_spoiled(masked, str(entry["target"])):
            print(f"\n{'='*80}")
            print(f"  id={row['id']}  sample_idx={row['sample_idx']}  target={entry['target']!r}  fraction={fraction}")
            print(f"  verdict: {row['verdict']!r}")
            print(f"  problem:\n{entry.get('question', entry.get('prompt', ''))}")
            print(f"  masked hint:\n{masked}")
            n_shown += 1
            if n_shown >= 5:
                break

    print(f"\nShowed {n_shown} discrepant examples (LLM spoiled, regex not).")


def inspect_id(rel_path, target_id):
    """Print all raw rows for target_id before and after dedup, showing sample_idx coverage."""
    from pprint import pformat
    path = BASE / rel_path
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line))

    raw_for_id = [e for e in raw if e["id"] == target_id]
    raw_idxs = sorted(e.get("sample_idx") for e in raw_for_id)

    deduped = load_entries(path)
    deduped_for_id = [e for e in deduped if e["id"] == target_id]
    deduped_idxs = sorted(e.get("sample_idx") for e in deduped_for_id)

    all_idxs = sorted(set(raw_idxs))
    missing = sorted(set(range(min(all_idxs), max(all_idxs) + 1)) - set(all_idxs))

    print(f"\n{'='*80}")
    print(f"id={target_id!r}  dataset={rel_path}")
    print(f"  raw rows:    {len(raw_for_id)}  sample_idxs={raw_idxs}")
    print(f"  after dedup: {len(deduped_for_id)}  sample_idxs={deduped_idxs}")
    print(f"  missing idxs in raw: {missing}")
    print()

    for entry in sorted(raw_for_id, key=lambda e: (e.get("sample_idx"), e.get("hint", "") == "")):
        hint_preview = (entry.get("hint") or "").strip()[:120].replace("\n", " ")
        print(f"  sample_idx={entry.get('sample_idx')}  "
              f"hint_empty={not bool(hint_preview)!s:<5}  "
              f"hint_preview={hint_preview!r}")


if __name__ == "__main__":
    # Switch between modes by commenting/uncommenting below.

    # Inspect cases where LLM says spoiled but regex doesn't — change fraction here:
    inspect_aime_solution_discrepancy(fraction=1)

    # Inspect a specific id's raw vs deduped rows:
    # inspect_id("data/solution/aime.jsonl", "1997-3")

    # Inspect duplicate (id, sample_idx) pairs across all datasets:
    # for rel_path, label in DATASETS:
    #     entries = load_entries(BASE / rel_path)
    #     print(f"\nLoaded {len(entries)} entries for {label}")
    #     analyze_duplicates(label, entries)

    # Run spoilage curve analysis and plot:
    main()
