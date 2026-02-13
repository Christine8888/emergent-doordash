"""
PCA over the benchmark *components* that make up ECI.

This script:
- loads baseline scores for the ECI component benchmarks (fit_eci.py:43-52)
- requires every model to have every component score (raises if missing)
- z-scores benchmark columns, runs PCA, and visualizes:
  - principal component weights (heatmap)
  - explained variance ratio (bar plot)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Constants / configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

BASELINE_FOLDER = PROJECT_ROOT / "christine_experiments/20251113/baseline"
OUTDIR = PROJECT_ROOT / "suze_experiments/20260212"

# Mapping from baseline eval names to ECI benchmark names (fit_eci.py:43-52)
EVAL_TO_ECI = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "mmlu_5_shot_cot": "MMLU",
    "bbh": "BBH",
    "arc_challenge": "ARC AI2",  # Epoch only uses Challenge score
    "winogrande": "Winogrande",  # 0-shot, 8192 tokens
    "math_level_5": "MATH level 5",
}

REQUIRED_BENCHMARKS = list(EVAL_TO_ECI.values())

N_COMPONENTS = 5
TOP_N_FOR_BRACE = 3

HEATMAP_FIGSIZE = (10, 5)
VAR_FIGSIZE = (5, 4)

HEATMAP_OUTFILE = OUTDIR / "pca_component_weights.png"
VAR_OUTFILE = OUTDIR / "pca_explained_variance.png"


@dataclass(frozen=True)
class PCAResult:
    components: np.ndarray  # shape: (n_components, n_features)
    explained_variance_ratio: np.ndarray  # shape: (n_components,)


def _raise_if_missing_required_scores(pivot: pd.DataFrame, required_benchmarks: list[str]) -> None:
    """Raise if any model is missing any required benchmark score."""
    missing_mask = pivot[required_benchmarks].isna()
    if not missing_mask.any().any():
        return

    missing_by_model: dict[str, list[str]] = {}
    for model, row in missing_mask.iterrows():
        missing_cols = [col for col, is_missing in row.items() if bool(is_missing)]
        if missing_cols:
            missing_by_model[str(model)] = missing_cols

    lines = [
        f"Missing required benchmark scores for {len(missing_by_model)} model(s).",
        f"Required benchmarks ({len(required_benchmarks)}): {required_benchmarks}",
        "",
    ]
    for model in sorted(missing_by_model.keys()):
        lines.append(f"- {model}: missing {missing_by_model[model]}")
    raise ValueError("\n".join(lines))


def _zscore_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    if np.any(std == 0):
        zero_cols = np.where(std.flatten() == 0)[0].tolist()
        raise ValueError(f"Cannot z-score: zero-variance columns at indices {zero_cols}")
    return (x - mean) / std, mean.flatten(), std.flatten()


def _pca_via_svd(x: np.ndarray, n_components: int) -> PCAResult:
    """PCA on already-centered data via SVD; returns components + explained variance ratio."""
    # x: (n_samples, n_features)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    n_components = min(n_components, vt.shape[0])

    # Explained variance matches sklearn:
    # explained_variance_ = (S**2) / (n_samples - 1)
    n_samples = x.shape[0]
    explained_variance = (s**2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    components = vt[:n_components]
    explained_variance_ratio = explained_variance_ratio[:n_components]

    # Deterministic sign convention: make the largest-|weight| entry positive per PC.
    for i in range(components.shape[0]):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1

    return PCAResult(components=components, explained_variance_ratio=explained_variance_ratio)


def _plot_weights_heatmap(weights: np.ndarray, benchmark_names: list[str], outfile: Path) -> None:
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    sns.heatmap(
        weights,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        ax=ax,
        xticklabels=benchmark_names,
        yticklabels=[f"PC-{i}" for i in range(1, weights.shape[0] + 1)],
    )
    ax.set_title("Principal component weights")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def _plot_explained_variance(evr: np.ndarray, outfile: Path, *, top_n: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=VAR_FIGSIZE)
    xs = np.arange(1, len(evr) + 1)
    ax.bar(xs, evr)
    ax.set_xticks(xs)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")

    top_n = min(int(top_n), len(evr))
    if top_n > 0:
        sum_top_n = float(evr[:top_n].sum())
        brace_height = float(evr[:top_n].max()) + 0.05

        # Simple bracket (matches style in the reference notebook closely enough)
        x0 = 1 - 0.4
        x1 = top_n + 0.4
        ax.plot([x0, x0, x1, x1], [brace_height - 0.02, brace_height, brace_height, brace_height - 0.02], color="gray")
        ax.text((x0 + x1) / 2, brace_height + 0.02, f"{sum_top_n:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def main() -> None:
    # python suze_experiments/20260212/pca.py
    from src.modelx import load_baseline

    rows: list[dict[str, object]] = []
    counts_by_benchmark: dict[str, int] = {}
    for eval_name, benchmark_name in EVAL_TO_ECI.items():
        df = load_baseline(str(BASELINE_FOLDER), eval_name)
        if df.empty:
            raise ValueError(f"No baseline results found for eval '{eval_name}' in {BASELINE_FOLDER}")

        if "accuracy" not in df.columns:
            raise ValueError(f"Baseline DataFrame for '{eval_name}' is missing required 'accuracy' column")

        models_with_score: set[str] = set()
        for _, r in df.iterrows():
            acc = r.get("accuracy")
            if pd.notna(acc):
                model_name = str(r["model"])
                models_with_score.add(model_name)
                rows.append(
                    {
                        "model": model_name,
                        "benchmark": str(benchmark_name),
                        "score": float(acc),
                    }
                )
        counts_by_benchmark[str(benchmark_name)] = len(models_with_score)

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError("No baseline scores loaded (unexpected).")

    pivot = scores.pivot_table(index="model", columns="benchmark", values="score", aggfunc="mean")

    print("\n=== PCA inputs (ECI component benchmarks) ===")
    print(f"Baseline folder: {BASELINE_FOLDER}")
    print(f"Required benchmarks ({len(REQUIRED_BENCHMARKS)}): {REQUIRED_BENCHMARKS}")
    print("\nModels with scores per benchmark (from baselines):")
    for bench in REQUIRED_BENCHMARKS:
        print(f"- {bench}: {counts_by_benchmark.get(bench, 0)} models")

    _raise_if_missing_required_scores(pivot, REQUIRED_BENCHMARKS)

    pivot = pivot[REQUIRED_BENCHMARKS].sort_index()
    models_used = pivot.index.tolist()
    print(f"\nModels used for PCA ({len(models_used)}):")
    for m in models_used:
        print(f"- {m}")

    x = pivot.to_numpy(dtype=float)
    xz, _, _ = _zscore_columns(x)

    n_components = min(int(N_COMPONENTS), xz.shape[1], xz.shape[0])
    res = _pca_via_svd(xz, n_components=n_components)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    _plot_weights_heatmap(res.components, REQUIRED_BENCHMARKS, HEATMAP_OUTFILE)
    _plot_explained_variance(res.explained_variance_ratio, VAR_OUTFILE, top_n=TOP_N_FOR_BRACE)

    print(f"Saved heatmap: {HEATMAP_OUTFILE}")
    print(f"Saved explained variance plot: {VAR_OUTFILE}")
    print("Explained variance ratio:", res.explained_variance_ratio.tolist())


if __name__ == "__main__":
    main()