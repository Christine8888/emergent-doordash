"""Compute and plot answer spoilage rate curves for hint datasets."""

import json
import re
import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("data/solution/aime.jsonl", "AIME solution"),
    ("data/solution/gpqa.jsonl", "GPQA solution"),
    ("data/cot/aime.jsonl", "AIME cot"),
    ("data/cot/gpqa.jsonl", "GPQA cot"),
]

FRACTIONS = np.arange(0.0, 1.01, 0.05)
BASE = "/Users/christineye/emergent-doordash/christine_experiments"


def split_preserving_whitespace(text):
    """Match codebase: split on whitespace, return tokens and word indices."""
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    return tokens, word_indices


def get_prefix_words(text, fraction):
    """Get the first `fraction` of words from text, using codebase word splitting."""
    tokens, word_indices = split_preserving_whitespace(text)
    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text
    last_word_idx = word_indices[num_words - 1]
    return "".join(tokens[:last_word_idx + 1]).strip()


def strip_answer_suffix(hint):
    """Split at the last occurrence of 'ANSWER:' and return text before it."""
    idx = hint.rfind("ANSWER:")
    if idx == -1:
        return hint
    return hint[:idx]


def target_is_spoiled(prefix_text, target):
    """Check if target appears as a standalone token in the prefix text."""
    pattern = r'(?<![A-Za-z0-9])' + re.escape(target) + r'(?![A-Za-z0-9])'
    return bool(re.search(pattern, prefix_text))


def load_entries(path):
    entries = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            entries.append(data)
    return entries


def compute_spoilage_curve(entries, fractions):
    rates = []
    for f in fractions:
        if f == 0.0:
            rates.append(0.0)
            continue
        spoiled = 0
        total = len(entries)
        for entry in entries:
            hint = entry["hint"]
            target = str(entry["target"])
            # Strip the ANSWER: suffix before checking
            hint_body = strip_answer_suffix(hint)
            prefix = get_prefix_words(hint_body, f)
            if target_is_spoiled(prefix, target):
                spoiled += 1
        rates.append(spoiled / total if total > 0 else 0.0)
    return rates


def main():
    results = {}
    for rel_path, label in DATASETS:
        path = f"{BASE}/{rel_path}"
        entries = load_entries(path)
        print(f"Loaded {len(entries)} entries for {label}")
        rates = compute_spoilage_curve(entries, FRACTIONS)
        results[label] = rates

    # Print table
    table_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    header = f"{'Dataset':<20}" + "".join(f"{'f=' + str(f):<12}" for f in table_fracs)
    print("\n" + header)
    print("-" * len(header))
    for label, rates in results.items():
        row = f"{label:<20}"
        for f in table_fracs:
            idx = int(round(f / 0.05))
            row += f"{rates[idx]:<12.4f}"
        print(row)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rates in results.items():
        ax.plot(FRACTIONS, rates, marker="o", markersize=3, label=label)
    ax.set_xlabel("Prefill Fraction")
    ax.set_ylabel("Spoilage Rate")
    ax.set_title("Answer Spoilage Rate by Prefill Fraction")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
