"""PCA utilities for 20260202 experiments.

This module computes PCA over the *ECI component* benchmark baselines and returns
per-model PC scores that can be used as capability features in scaling laws.

It is intentionally lightweight (numpy/pandas/matplotlib/seaborn only) to keep
dependencies aligned with the rest of `suze_experiments/20260202/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(frozen=True)
class PCAResult:
    components: np.ndarray  # shape: (k, n_features)
    explained_variance_ratio: np.ndarray  # shape: (k,)
    benchmarks: list[str]  # feature names, length n_features
    models: list[str]  # model names, length n_samples


def _raise_if_missing_required_scores(pivot: pd.DataFrame, required_benchmarks: list[str]) -> None:
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


def _pca_via_svd(x_centered: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA on already-centered data via SVD; returns (components, explained_variance_ratio)."""
    # x_centered: (n_samples, n_features)
    _u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    k = min(int(n_components), int(vt.shape[0]))

    n_samples = x_centered.shape[0]
    explained_variance = (s**2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    components = vt[:k]
    explained_variance_ratio = explained_variance_ratio[:k]

    # Deterministic sign convention: make the largest-|weight| entry positive per PC.
    for i in range(components.shape[0]):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1

    return components, explained_variance_ratio


def default_eval_to_eci_mapping(*, baseline_folder: Path) -> dict[str, str]:
    """Return a robust eval->benchmark mapping for PCA over ECI components.

    We start from `src.modelx.EVAL_TO_ECI` but patch in common local baseline
    eval names found in this repo (e.g., `mmlu_5_shot_cot`, `winogrande`).
    """
    from src.modelx import EVAL_TO_ECI as MODELX_EVAL_TO_ECI

    mapping = dict(MODELX_EVAL_TO_ECI)

    # Enforce the canonical MMLU baseline name for this repo.
    mapping["mmlu_5_shot_cot"] = "MMLU"
    mapping.pop("mmlu_5_shot", None)

    return mapping


def compute_pc_scores(
    *,
    baseline_folder: Path,
    n_components: int,
    eval_to_benchmark: dict[str, str] | None = None,
    eval_names: list[str] | None = None,
) -> tuple[pd.DataFrame, PCAResult, dict[str, np.ndarray]]:
    """Compute PCA and per-model PC scores from baseline results.

    Returns:
      - pivot: DataFrame indexed by model with benchmark columns
      - pca: PCAResult containing components and metadata
      - pc_scores_map: {model -> np.ndarray shape (k,)} where k=min(n_components, ...)
    """
    from src.modelx import load_baseline

    baseline_folder = Path(baseline_folder)
    if eval_to_benchmark is None:
        eval_to_benchmark = default_eval_to_eci_mapping(baseline_folder=baseline_folder)

    if eval_names is not None:
        # Dedupe while preserving order.
        seen: set[str] = set()
        eval_names = [e for e in eval_names if not (e in seen or seen.add(e))]

        missing = [e for e in eval_names if e not in eval_to_benchmark]
        if missing:
            raise ValueError(
                "Unknown eval(s) requested for PCA: "
                + ", ".join(repr(m) for m in missing)
                + ". Available evals: "
                + ", ".join(sorted(eval_to_benchmark.keys()))
            )

        eval_to_benchmark = {e: eval_to_benchmark[e] for e in eval_names}

    rows: list[dict[str, object]] = []
    for eval_name, benchmark_name in eval_to_benchmark.items():
        df = load_baseline(str(baseline_folder), str(eval_name))
        if df.empty:
            continue
        if "accuracy" not in df.columns:
            raise ValueError(f"Baseline DataFrame for '{eval_name}' is missing required 'accuracy' column")

        for _, r in df.iterrows():
            acc = r.get("accuracy")
            if pd.notna(acc):
                rows.append({"model": str(r["model"]), "benchmark": str(benchmark_name), "score": float(acc)})

    scores = pd.DataFrame(rows)
    if scores.empty:
        suffix = f" (requested evals: {eval_names})" if eval_names is not None else ""
        raise ValueError(f"No baseline scores loaded from {baseline_folder}{suffix}")

    pivot = scores.pivot_table(index="model", columns="benchmark", values="score", aggfunc="mean")

    required_benchmarks = sorted(scores["benchmark"].unique().tolist())
    _raise_if_missing_required_scores(pivot, required_benchmarks)

    pivot = pivot[required_benchmarks].sort_index()
    models = pivot.index.tolist()
    benchmarks = required_benchmarks

    x = pivot.to_numpy(dtype=float)
    xz, _mean, _std = _zscore_columns(x)

    k = int(min(int(n_components), xz.shape[1], xz.shape[0]))
    components, evr = _pca_via_svd(xz, n_components=k)

    pc_scores = xz @ components.T  # (n_models, k)
    pc_scores_map = {str(m): pc_scores[i].astype(float) for i, m in enumerate(models)}

    pca = PCAResult(
        components=components,
        explained_variance_ratio=evr,
        benchmarks=benchmarks,
        models=[str(m) for m in models],
    )
    return pivot, pca, pc_scores_map


def plot_component_weights_heatmap(
    *,
    pca: PCAResult,
    outfile: Path,
    figsize: tuple[int, int] = (10, 5),
) -> Path:
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pca.components,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        ax=ax,
        xticklabels=pca.benchmarks,
        yticklabels=[f"PC-{i}" for i in range(1, pca.components.shape[0] + 1)],
    )
    ax.set_title("Principal component weights")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    return outfile


def plot_explained_variance(
    *,
    pca: PCAResult,
    outfile: Path,
    top_n_for_brace: int = 3,
    figsize: tuple[int, int] = (5, 4),
) -> Path:
    evr = np.asarray(pca.explained_variance_ratio, dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xs = np.arange(1, len(evr) + 1)
    ax.bar(xs, evr)
    ax.set_xticks(xs)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")

    top_n = min(int(top_n_for_brace), len(evr))
    if top_n > 0:
        sum_top_n = float(evr[:top_n].sum())
        brace_height = float(evr[:top_n].max()) + 0.05
        x0 = 1 - 0.4
        x1 = top_n + 0.4
        ax.plot([x0, x0, x1, x1], [brace_height - 0.02, brace_height, brace_height, brace_height - 0.02], color="gray")
        ax.text((x0 + x1) / 2, brace_height + 0.02, f"{sum_top_n:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    return outfile

