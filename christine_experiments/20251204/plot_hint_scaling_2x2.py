"""2x3 hint scaling law plots: one subplot per model, solution methods only."""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit

from src.modelx import size
from src.modelx.results import load_results


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_sigmoid_free(x, y):
    """Fit y = L + (U - L) * sigmoid(m*x + b) with all 4 params free."""
    def model(x, L, U, m, b):
        return L + (U - L) * _sigmoid(m * x + b)
    p0 = [np.min(y), np.max(y), 4.0, -2.0]
    bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
    params, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
    return params  # L, U, m, b

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DATA_DIR = str(_PROJECT_ROOT / "christine_experiments" / "20251113" / "results")
SAVE_DIR = _PROJECT_ROOT / "plots"
SAVE_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--qwen3", action="store_true", help="plot Qwen3 models only")
parser.add_argument("--cot", action="store_true", help="use cot methods instead of solution")
args = parser.parse_args()

REASONING = "cot" if args.cot else "solution"

if args.qwen3:
    MODELS = [
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B",
        "Qwen3-8B",   "Qwen3-14B",  "Qwen3-32B",
    ]
    SAVE_NAME = f"{REASONING}_scaling_laws_gpqa_qwen3.pdf"
else:
    MODELS = [
        "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct",
        "Qwen3-4B",              "gemma-3-12b-it",          "Llama-3.1-70B-Instruct",
    ]
    SAVE_NAME = f"{REASONING}_scaling_laws_gpqa.pdf"

METHODS = [
    f"{REASONING}_intext_masked",
    f"{REASONING}_intext_sequential",
    f"{REASONING}_prefill_sequential",
]

# 3 hint types -> 3 colors; reasoning type -> line style
HINT_TYPES = ["in-text masked", "in-text sequential", "pre-fill sequential"]
HINT_TYPE_COLORS = dict(zip(HINT_TYPES, sns.color_palette("husl", 3)))




def parse_method(method):
    """Split method into (reasoning_type, hint_type)."""
    parts = method.split("_", 1)  # e.g. "cot", "intext_masked"
    reasoning = parts[0]
    raw = parts[1].replace("_", " ", 1)  # "intext masked"
    hint_type = raw.replace("intext", "in-text").replace("prefill", "pre-fill")
    return reasoning, hint_type


# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
frames = {}
for method in METHODS:
    df = load_results(DATA_DIR, "gpqa", method)
    if not df.empty:
        df = df[df["model"].isin(MODELS)]
        df["method"] = method
        frames[method] = df

all_df = pd.concat(frames.values(), ignore_index=True)

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 16

fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

x_smooth = np.linspace(0, 1, 200)

for ax_idx, model in enumerate(MODELS):
    ax = axes[ax_idx]
    model_df = all_df[all_df["model"] == model]

    for method in METHODS:
        method_df = model_df[model_df["method"] == method].sort_values("hint")
        if method_df.empty:
            continue

        _, hint_type = parse_method(method)
        color = HINT_TYPE_COLORS[hint_type]

        x = method_df["hint"].values
        y = method_df["accuracy"].values
        yerr = method_df["stderr"].fillna(0).values if "stderr" in method_df.columns else None

        # data points (no connecting line)
        ax.errorbar(
            x, y, yerr=yerr,
            color=color, linestyle="none",
            marker="o", markersize=6, capsize=3, alpha=0.7,
        )

        # best-fit sigmoid with free asymptotes
        L, U, m, b = fit_sigmoid_free(x, y)
        ax.plot(x_smooth, L + (U - L) * _sigmoid(m * x_smooth + b),
                color=color, linestyle="-", linewidth=1.8, alpha=0.8)

    ax.set_title(model, fontsize=21, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

# shared axis labels
for ax in axes[3:]:
    ax.set_xlabel("hint fraction", fontsize=20, fontweight="bold")
for ax in [axes[0], axes[3]]:
    ax.set_ylabel("accuracy", fontsize=20, fontweight="bold")

# build legend: 3 color entries
legend_handles = []
for ht in HINT_TYPES:
    legend_handles.append(Line2D([0], [0], color=HINT_TYPE_COLORS[ht], linewidth=2,
                                 marker="o", markersize=6, label=ht))

fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=16,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout()
fig.subplots_adjust(bottom=0.10)

fig.savefig(SAVE_DIR / SAVE_NAME, bbox_inches="tight", dpi=300)
plt.show()
