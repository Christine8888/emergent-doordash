"""Sanity-check plots for AIME runs.

This script makes a raw accuracy-vs-ECI plot by hint level:
- scatter points only (no scaling-law fit),
- one color per hint level,
- top x-axis labeling which model(s) sit at each ECI,
- strict validation that each (hint, model) appears at most once.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_eci_map(eci_file: Path) -> dict[str, float]:
    eci_df = pd.read_csv(eci_file)
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def _model_labels_by_eci(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    model_eci = df[["model", "eci"]].drop_duplicates()
    grouped = (
        model_eci.groupby("eci", as_index=False)["model"]
        .agg(lambda s: ", ".join(sorted(s.tolist())))
        .sort_values("eci")
    )
    ticks = [float(v) for v in grouped["eci"].tolist()]
    labels = grouped["model"].tolist()
    return ticks, labels


def load_results_df(
    *,
    base_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
) -> pd.DataFrame:
    # Import here so this script can run from repo root without install.
    from src.modelx import load_results

    df = load_results(
        base_folder=str(base_folder),
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )
    if df.empty:
        raise ValueError("No results found for the provided eval/solver/condition.")

    if all_models is not None:
        df = df[df["model"].isin(all_models)]
    if hint_fractions is not None:
        df = df[df["hint"].isin(hint_fractions)]

    eci_map = _load_eci_map(eci_file)
    df = df.copy()
    df["eci"] = df["model"].map(eci_map)
    df = df.dropna(subset=["eci", "accuracy", "hint"])

    if df.empty:
        raise ValueError("No rows remain after model/hint/ECI filtering.")

    return df


def _combos_with_local_eval(
    *,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
) -> set[tuple[str, float]]:
    eval_root = base_folder / eval_name / solver / condition
    combos: set[tuple[str, float]] = set()
    if not eval_root.exists():
        return combos

    for model_dir in eval_root.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for eval_file in model_dir.glob("*.eval"):
            try:
                with ZipFile(eval_file) as zf:
                    start = json.loads(zf.read("_journal/start.json"))
            except (BadZipFile, KeyError, ValueError, OSError):
                continue

            eval_info = start.get("eval", {})
            hint = eval_info.get("metadata", {}).get("hint_fraction")
            if hint is None:
                continue
            combos.add((model_name, round(float(hint), 2)))
    return combos


def add_inference_owner(
    *,
    df: pd.DataFrame,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
) -> pd.DataFrame:
    local_eval_combos = _combos_with_local_eval(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )
    out = df.copy()
    out["inference_owner"] = out.apply(
        lambda r: (
            "mine"
            if (r["model"], round(float(r["hint"]), 2)) in local_eval_combos
            else "christine"
        ),
        axis=1,
    )
    return out


def assert_unique_model_hint(df: pd.DataFrame, *, output_dir: Path | None = None) -> None:
    counts = (
        df.groupby(["hint", "model"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    dup = counts[counts["count"] > 1].sort_values(["hint", "model"])
    if dup.empty:
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dup.to_csv(output_dir / "duplicate_model_hint_counts.csv", index=False)

    sample = dup.head(10).to_dict(orient="records")
    raise ValueError(
        "Found duplicate points for the same (hint, model). "
        f"Rows with count>1: {len(dup)}. Sample: {sample}"
    )


def plot_accuracy_vs_eci_raw_by_hint(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str = "accuracy_vs_eci_by_hint_raw_points.png",
    title: str = "AIME: accuracy vs ECI by hint (raw points only)",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

    for hint in hints:
        hint_df = df[df["hint"] == hint].sort_values("eci")
        ax.scatter(
            hint_df["eci"],
            hint_df["accuracy"],
            color=colors[hint],
            label=f"h={hint:.2f}",
            alpha=0.85,
            s=65,
        )

    ax.set_xlabel("ECI")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    ticks, labels = _model_labels_by_eci(df)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels(labels, rotation=75, ha="left", fontsize=8)
    ax_top.set_xlabel("model(s) at each ECI")

    plt.tight_layout()
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_vs_hint_by_model_raw_points(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str = "accuracy_vs_hint_by_model.png",
    title: str = "AIME: accuracy vs hint by model (raw points only)",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_eci = (
        df[["model", "eci"]]
        .drop_duplicates()
        .sort_values("eci")
    )
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    model_cmap = plt.cm.coolwarm
    model_colors = {m: model_cmap(i / max(n_models - 1, 1)) for i, m in enumerate(models_sorted)}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        eci = float(model_df["eci"].iloc[0])

        ax.scatter(
            model_df["hint"],
            model_df["accuracy"],
            color=model_colors[model],
            alpha=0.85,
            s=45,
        )
        ax.set_title(f"{model}\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_vs_hint_by_model_raw_points_by_owner(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str = "accuracy_vs_hint_by_model_owner_debug.png",
    title: str = "AIME: accuracy vs hint by model (owner-colored debug)",
) -> Path:
    if "inference_owner" not in df.columns:
        raise ValueError("Missing 'inference_owner' column. Call add_inference_owner(...) first.")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_eci = (
        df[["model", "eci"]]
        .drop_duplicates()
        .sort_values("eci")
    )
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    owner_colors = {"mine": "#1f77b4", "christine": "#ff7f0e"}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        eci = float(model_df["eci"].iloc[0])

        for owner in ["mine", "christine"]:
            owner_df = model_df[model_df["inference_owner"] == owner]
            if owner_df.empty:
                continue
            ax.scatter(
                owner_df["hint"],
                owner_df["accuracy"],
                color=owner_colors[owner],
                alpha=0.85,
                s=45,
            )
        ax.set_title(f"{model}\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=owner_colors["mine"], label="mine"),
        Line2D([], [], marker="o", linestyle="", color=owner_colors["christine"], label="christine"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_aime_raw_sanity_plot(
    *,
    base_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    output_dir: Path,
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
) -> tuple[Path, Path]:
    df = load_results_df(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
    )
    assert_unique_model_hint(df, output_dir=output_dir)

    summary = (
        df.groupby(["hint"], as_index=False)
        .agg(n_models=("model", "nunique"), mean_accuracy=("accuracy", "mean"))
        .sort_values("hint")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "raw_hint_summary.csv", index=False)

    # Detailed table: one row per model, one column per hint fraction.
    # This gives a direct sanity-check view of accuracy per model per hint level.
    per_model_hint = (
        df.pivot(index="model", columns="hint", values="accuracy")
        .sort_index()
        .sort_index(axis=1)
    )
    per_model_hint.columns = [f"hint_{float(c):.2f}" for c in per_model_hint.columns]
    per_model_hint.to_csv(output_dir / "accuracy_per_model_per_hint.csv")

    # Long-form detailed table for easy filtering/joins.
    detailed_long = df[["model", "hint", "eci", "accuracy"]].sort_values(["model", "hint"])
    detailed_long.to_csv(output_dir / "accuracy_per_model_per_hint_long.csv", index=False)

    out_eci = plot_accuracy_vs_eci_raw_by_hint(df=df, output_dir=output_dir)
    out_by_model = plot_accuracy_vs_hint_by_model_raw_points(df=df, output_dir=output_dir)
    return out_eci, out_by_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base_folder = project_root / "christine_experiments/20251113/results"
    eci_file = project_root / "christine_experiments/20260129_fitting/eci_model_capabilities.csv"
    output_dir = Path(__file__).resolve().parent
    all_models = [
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
    ]
    hint_fractions = [round(i / 20.0, 2) for i in range(21)]

    out_eci, out_by_model = run_aime_raw_sanity_plot(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name="aime",
        solver="solution_intext_masked",
        condition="0shot",
        output_dir=output_dir,
        all_models=all_models,
        hint_fractions=hint_fractions,
    )

    debug_df = load_results_df(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name="aime",
        solver="solution_intext_masked",
        condition="0shot",
        all_models=all_models,
        hint_fractions=hint_fractions,
    )
    debug_df = add_inference_owner(
        df=debug_df,
        base_folder=base_folder,
        eval_name="aime",
        solver="solution_intext_masked",
        condition="0shot",
    )
    out_by_owner = plot_accuracy_vs_hint_by_model_raw_points_by_owner(
        df=debug_df,
        output_dir=output_dir,
    )

    print(f"Wrote: {out_eci}")
    print(f"Wrote: {out_by_model}")
    print(f"Wrote: {out_by_owner}")


if __name__ == "__main__":
    # python suze_experiments/20260307/aime_plots.py
    main()
