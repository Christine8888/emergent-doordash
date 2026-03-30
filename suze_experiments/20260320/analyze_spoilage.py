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

DEFAULT_FRACTIONS = np.arange(0.0, 1.01, 0.05)
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "christine_experiments"
FRACTION_ROUND_DECIMALS = 6

# Map christine's dataset name -> our label
CHRISTINE_DATASET_MAP = {
    "aime_solution": "AIME solution",
    "gpqa_solution": "GPQA solution",
    "aime_cot":      "AIME cot",
    "gpqa_cot":      "GPQA cot",
}
CHRISTINE_MASKED_RESULTS_PATH = BASE / "20260304/spoilage_results_masked.jsonl"
CHRISTINE_TRUNCATED_RESULTS_PATH = BASE / "20260304/spoilage_results.jsonl"


def normalize_mode(mode: str) -> str:
    """Map aliases onto canonical modes used internally."""
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"masked", "mask"}:
        return "masked"
    if mode_norm in {"hinted", "hint", "truncated", "truncate", "prefix"}:
        return "truncated"
    raise ValueError(f"unknown mode={mode!r}; expected masked or hinted")


def canonical_fraction(value: float) -> float:
    return round(float(value), FRACTION_ROUND_DECIMALS)


def normalize_fractions(fractions) -> np.ndarray:
    """Validate and normalize fractions to sorted unique float array in [0, 1]."""
    if fractions is None:
        raise ValueError("fractions must be provided explicitly")
    vals = [canonical_fraction(float(x)) for x in fractions]
    if not vals:
        raise ValueError("fractions cannot be empty")

    uniq_sorted = sorted(set(vals))
    for f in uniq_sorted:
        if f < 0.0 or f > 1.0:
            raise ValueError(f"fraction out of range [0,1]: {f}")
    return np.array(uniq_sorted, dtype=float)


def format_fractions_for_title(fractions: np.ndarray) -> str:
    return ", ".join(f"{float(f):g}" for f in fractions)


def normalize_datasets(datasets) -> list[tuple[str, str]]:
    """Validate datasets as a list of (relative_path, label) tuples."""
    if datasets is None:
        raise ValueError("datasets must be provided explicitly")
    normalized = []
    for item in datasets:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                f"invalid dataset entry {item!r}; expected (relative_path, label)"
            )
        rel_path, label = item
        normalized.append((str(rel_path), str(label)))
    if not normalized:
        raise ValueError("datasets cannot be empty")
    return normalized


def _truncate_at_stop_string(text: str, stop_string: str, fraction) -> str:
    """Truncate text before stop_string."""
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


def get_truncated_prefix_text(text: str, fraction: float, stop_string: str = "ANSWER:") -> str:
    """Match 20260304/spoilage_judge.py truncation logic."""
    text = _truncate_at_stop_string(text, stop_string, fraction)

    tokens, word_indices = _split_preserving_whitespace(text)
    if not word_indices:
        return text

    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text

    last_idx = word_indices[num_words - 1]
    return "".join(tokens[:last_idx + 1]).strip()


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


def write_json(data, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


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


def compute_spoilage_curve(entries, fractions, mode="masked"):
    """Compute spoilage by averaging per-id spoilage rates (equal question weight)."""
    if mode not in {"masked", "truncated"}:
        raise ValueError(f"unknown mode={mode!r}; expected masked|truncated")

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
                if not hint:
                    raise ValueError(f'hint is empty: {pprint(entry)}')

                if mode == "masked":
                    # Deterministic seed for reproducible masked spoilage curves.
                    mask_seed = f"{entry['id']}_{entry.get('sample_idx', 0)}_{f}_0"
                    judged_hint = get_masked_text(
                        hint, fraction=f, stop_string="ANSWER:", seed=mask_seed
                    )
                else:
                    judged_hint = get_truncated_prefix_text(
                        hint, fraction=f, stop_string="ANSWER:"
                    )

                if target_is_spoiled(judged_hint, target):
                    spoiled += 1
            per_id_rates.append(spoiled / total if total > 0 else 0.0)

        rates.append(float(np.mean(per_id_rates)) if per_id_rates else 0.0)
    return rates


def load_christine_spoilage_curves(
    results_path,
    source_pair_counts_by_label,
    selected_fractions=None,
    force_zero_fraction=False,
):
    """Load Christine's pre-judged (LLM) spoilage results and aggregate per-id.

    Returns:
        dict mapping label -> (fractions_array, rates_list)
    """
    rows = []
    with open(results_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # group by (dataset, fraction) -> {id -> [spoiled, ...]}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = CHRISTINE_DATASET_MAP.get(row["dataset"])
        key = (label, row["fraction"])
        grouped[key][row["id"]].append(row["spoiled"])

    # Validate rollout counts per id at fraction=1.0.
    # Only warn when LLM rows cover the full source (id, sample_idx) population.
    for label in CHRISTINE_DATASET_MAP.values():
        per_id = grouped.get((label, 1.0), {})
        if not per_id:
            continue
        judged_pairs = {
            (row["id"], row["sample_idx"])
            for row in rows
            if CHRISTINE_DATASET_MAP.get(row["dataset"]) == label
        }
        expected_pairs = int(source_pair_counts_by_label.get(label, 0))
        has_full_coverage = expected_pairs > 0 and len(judged_pairs) >= expected_pairs
        if not has_full_coverage:
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
    if selected_fractions is not None:
        allowed = {canonical_fraction(f) for f in selected_fractions}
        all_fractions = [f for f in all_fractions if canonical_fraction(f) in allowed]
    all_labels = list(CHRISTINE_DATASET_MAP.values())

    results = {}
    for label in all_labels:
        rates = []
        for f in all_fractions:
            per_id = grouped[(label, f)]
            if not per_id:
                rates.append(0.0)
                continue
            if force_zero_fraction and f == 0.0:
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


def get_matched_entries(entries, llm_pairs):
    """Filter entries to unique (id, sample_idx) pairs that appear in llm_pairs."""
    seen_keys = set()
    matched_entries = []
    for entry in entries:
        key = (entry["id"], entry.get("sample_idx"))
        if key in llm_pairs and key not in seen_keys:
            matched_entries.append(entry)
            seen_keys.add(key)
    return matched_entries


def plot_comparison(
    datasets,
    all_entries,
    regex_results,
    n_samples,
    christine_results,
    display_mode_name,
    compute_mode,
    fractions,
    out_path,
):
    """Plot one comparison panel for a single hinting mode and return plotted data."""
    labels = [label for _, label in datasets]
    if not labels:
        raise ValueError("datasets cannot be empty")

    n_panels = len(labels)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    color_regex       = "steelblue"
    color_regex_match = "mediumseagreen"
    color_llm         = "tomato"
    summary = {
        "display_mode": display_mode_name,
        "compute_mode": compute_mode,
        "fractions": [float(x) for x in fractions],
        "datasets": [],
    }

    for idx, label in enumerate(labels):
        ax = axes_arr[idx]
        rates_regex = regex_results[label]
        n_regex = n_samples[label]
        ax.plot(
            fractions,
            rates_regex,
            color=color_regex,
            linewidth=2.0,
            label=f"regex judge (n={n_regex})",
        )
        dataset_summary = {
            "label": label,
            "n_regex": int(n_regex),
            "regex_fractions": [float(x) for x in fractions],
            "regex_rates": [float(x) for x in rates_regex],
        }

        if label in christine_results:
            fracs_llm, rates_llm, llm_pairs = christine_results[label]
            n_llm = len(llm_pairs)
            ax.plot(
                fracs_llm,
                rates_llm,
                color=color_llm,
                linestyle="--",
                linewidth=2.0,
                label=f"LLM judge (n={n_llm})",
            )
            dataset_summary["n_llm"] = int(n_llm)
            dataset_summary["llm_fractions"] = [float(x) for x in fracs_llm]
            dataset_summary["llm_rates"] = [float(x) for x in rates_llm]

            # Regex curve restricted to the same (id, sample_idx) pairs as the LLM judge.
            matched_entries = get_matched_entries(all_entries[label], llm_pairs)
            if matched_entries:
                rates_matched = compute_spoilage_curve(
                    matched_entries, fractions, mode=compute_mode
                )
                ax.plot(
                    fractions,
                    rates_matched,
                    color=color_regex_match,
                    linestyle=":",
                    linewidth=2.0,
                    label=f"regex judge matched (n={len(matched_entries)})",
                )
                dataset_summary["n_regex_matched"] = int(len(matched_entries))
                dataset_summary["regex_matched_fractions"] = [float(x) for x in fractions]
                dataset_summary["regex_matched_rates"] = [float(x) for x in rates_matched]

        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=8)
        summary["datasets"].append(dataset_summary)

    for idx, ax in enumerate(axes_arr):
        if idx >= n_panels:
            ax.set_visible(False)
            continue
        row = idx // n_cols
        col = idx % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Hint Fraction")
        if col == 0:
            ax.set_ylabel("Spoilage Rate")

    frac_text = format_fractions_for_title(fractions)
    fig.suptitle(
        f"Answer Spoilage Rate: Regex vs LLM Judge ({display_mode_name} hints, fractions: {frac_text})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    return summary


def run_spoilage_plot(
    *,
    mode,
    fractions,
    datasets,
    out_path,
    json_out_path,
    show_plot,
):
    """Run analysis + plot for one hint mode.

    Args:
        mode: "masked" or "hinted" (aliases: "truncated", "prefix", etc.).
        fractions: Iterable of fractions in [0, 1].
        datasets: List of (relative_path, label) tuples to include.
        out_path: Output path for the PDF.
        json_out_path: Output path for JSON summary.
        show_plot: Whether to call plt.show() at the end.
    """
    mode_norm = normalize_mode(mode)
    mode_label = "masked" if mode_norm == "masked" else "hinted"
    fractions_arr = normalize_fractions(fractions)
    datasets_norm = normalize_datasets(datasets)

    if mode_norm == "masked":
        christine_path = CHRISTINE_MASKED_RESULTS_PATH
        force_zero_fraction = True
    else:
        christine_path = CHRISTINE_TRUNCATED_RESULTS_PATH
        force_zero_fraction = False

    all_entries = {}
    regex_results = {}       # label -> rates
    n_samples = {}           # label -> total hint count

    for rel_path, label in datasets_norm:
        path = BASE / rel_path
        entries = load_entries(path)
        print(f"Loaded {len(entries)} entries for {label}")
        validate_hint_counts_per_id(entries, label)
        all_entries[label] = entries
        regex_results[label] = compute_spoilage_curve(entries, fractions_arr, mode=mode_norm)
        n_samples[label] = len(entries)
    source_pair_counts_by_label = {
        label: len({(e["id"], e.get("sample_idx")) for e in entries})
        for label, entries in all_entries.items()
    }

    christine_results = load_christine_spoilage_curves(
        christine_path,
        source_pair_counts_by_label=source_pair_counts_by_label,
        selected_fractions=fractions_arr,
        force_zero_fraction=force_zero_fraction,
    )

    summary = plot_comparison(
        datasets=datasets_norm,
        all_entries=all_entries,
        regex_results=regex_results,
        n_samples=n_samples,
        christine_results=christine_results,
        display_mode_name=mode_label,
        compute_mode=mode_norm,
        fractions=fractions_arr,
        out_path=out_path,
    )
    summary.update(
        {
            "requested_mode": str(mode),
            "datasets": [
                {
                    "relative_path": rel_path,
                    "label": label,
                    "series": next((d for d in summary["datasets"] if d["label"] == label), {}),
                }
                for rel_path, label in datasets_norm
            ],
            "plot_path": str(out_path),
        }
    )
    write_json(summary, Path(json_out_path))
    print(f"Saved JSON to {json_out_path}")
    if show_plot:
        plt.show()
    return out_path, Path(json_out_path)


def main():
    # run_spoilage_plot(
    #     mode="masked",
    #     fractions=DEFAULT_FRACTIONS,
    #     datasets=DATASETS,
    #     out_path=REPO_ROOT / "suze_experiments/20260320/spoilage_comparison.pdf",
    #     json_out_path=REPO_ROOT / "suze_experiments/20260320/spoilage_comparison.json",
    #     show_plot=False,
    # )
    # run_spoilage_plot(
    #     mode="truncated",
    #     fractions=DEFAULT_FRACTIONS,
    #     datasets=DATASETS,
    #     out_path=REPO_ROOT / "suze_experiments/20260320/spoilage_comparison_truncation.pdf",
    #     json_out_path=REPO_ROOT / "suze_experiments/20260320/spoilage_comparison_truncation.json",
    #     show_plot=True,
    # )

    run_spoilage_plot(
        mode="truncated",
        fractions=np.arange(0.0, 1.01, 0.01),
        datasets=[
            ("data/solution/aime.jsonl", "AIME solution"),
            ("data/solution/gpqa.jsonl", "GPQA solution"),
            # ("data/cot/aime.jsonl", "AIME cot"),
            # ("data/cot/gpqa.jsonl", "GPQA cot"),
        ],
        out_path=REPO_ROOT / "suze_experiments/20260320/spoilage_comparison_truncation_granular_aime.pdf",
        json_out_path=REPO_ROOT / "suze_experiments/20260320/spoilage_comparison_truncation_granular_aime.json",
        show_plot=True,
    )


def inspect_aime_solution_discrepancy(fraction=0.9):
    """Print examples where LLM says spoiled but regex doesn't at a given fraction for AIME solution."""
    label = "AIME solution"

    entries = load_entries(BASE / "data/solution/aime.jsonl")
    entries_by_key = {(e["id"], e.get("sample_idx")): e for e in entries}

    rows_llm = []
    with open(CHRISTINE_MASKED_RESULTS_PATH) as f:
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
    # python suze_experiments/20260320/analyze_spoilage.py
    # Switch between modes by commenting/uncommenting below.

    # Inspect cases where LLM says spoiled but regex doesn't — change fraction here:
    # inspect_aime_solution_discrepancy(fraction=1)

    # Inspect a specific id's raw vs deduped rows:
    # inspect_id("data/solution/aime.jsonl", "1997-3")

    # Inspect duplicate (id, sample_idx) pairs across all datasets:
    # for rel_path, label in DATASETS:
    #     entries = load_entries(BASE / rel_path)
    #     print(f"\nLoaded {len(entries)} entries for {label}")
    #     analyze_duplicates(label, entries)

    # Run spoilage curve analysis and plot:
    main()
