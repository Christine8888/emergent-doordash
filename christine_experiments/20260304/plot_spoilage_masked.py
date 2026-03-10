"""Plot spoilage rate curves from masked LLM-judge results."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = Path(__file__).parent / "spoilage_results_masked.jsonl"

DATASET_CONFIG = {
    "aime_solution": ("AIME", sns.color_palette("husl", 2)[0], "-"),
    "aime_cot":      ("AIME", sns.color_palette("husl", 2)[0], "--"),
    "gpqa_solution": ("GPQA", sns.color_palette("husl", 2)[1], "-"),
    "gpqa_cot":      ("GPQA", sns.color_palette("husl", 2)[1], "--"),
}


def load_results():
    groups = defaultdict(list)
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            groups[(r["dataset"], r["fraction"])].append(r["spoiled"])
    return groups


def plot_spoilage():
    groups = load_results()

    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset, (label, color, ls) in DATASET_CONFIG.items():
        fractions = sorted(set(f for (ds, f) in groups if ds == dataset))
        rates = [np.mean(groups[(dataset, f)]) * 100 for f in fractions]

        hint_type = "solution" if "solution" in dataset else "CoT"
        ax.plot(
            fractions, rates,
            color=color, linestyle=ls, marker="o", markersize=6,
            linewidth=2, alpha=0.8,
            label=f"{label} ({hint_type})",
        )

    ax.legend(framealpha=0.9)
    ax.set_xlabel("fraction of hint visible (rest masked)", fontsize=13, fontweight="bold")
    ax.set_ylabel("spoilage rate (%)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-2, 105)

    plt.tight_layout()
    fig.savefig(_PROJECT_ROOT / "plots" / "spoilage_rate_masked.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_spoilage()
