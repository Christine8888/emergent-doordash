
"""Joint scaling law plots with ECI on x-axis.

Features:
- Joint fit: ε(C, h) = σ(αC + βh + νCh + γ) or σ(αC + βh + γ)
- Individual scaling laws in C for each hint value
- Individual scaling laws in h for each model
- TRAIN/TEST split for models
- Metrics: MSE, mean midpoint error
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

# Project root (works on any machine).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory for this particular experiment run (this file's folder).
EXPERIMENT_DIR = Path(__file__).resolve().parent

# Where to write plots (created automatically in `main()`).
PLOTS_DIR = EXPERIMENT_DIR / "plots"

# This experiment script lives outside the packaged `src/` tree, and some older
# experiment code expects imports to work via repo-relative paths. Adding the repo
# root + a legacy experiments folder keeps this script runnable without needing
# `pip install -e .` or changing the repo layout.
sys.path.append(str(PROJECT_ROOT / "christine_experiments/20251204"))
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration constants
BASE_FOLDER = PROJECT_ROOT / "christine_experiments/20251113/results"
BASELINE_FOLDER = PROJECT_ROOT / "christine_experiments/20251113/baseline"
ECI_FILE = PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv"

EVAL_NAME = "gpqa"
SOLVER = "solution_intext_masked"
LABEL = "GPQA solution intext masked"

# All models
ALL_MODELS = [
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

# Train/test split (for now, use all as train)
TRAIN_MODELS = ALL_MODELS
TEST_MODELS = set()  # Empty for now

HINT_FRACTIONS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

EVAL_HINTS_FOR_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Hint transform (identity by default, can be changed)
HINT_TRANSFORM: Callable[[float], float] = lambda h: h  # math.log(h/(1-h))

# Whether to include cross term in joint fit
INCLUDE_CROSS = True

# Lower asymptote (e.g., 0.2 for random baseline on 5-choice task). Set to None for no constraint.
LOWER_ASYMPTOTE: float | None = 0.2


# Helper functions

def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save a matplotlib figure and close it.

    We default to saving (instead of `plt.show()`) so this script works in
    non-interactive runs (e.g. ssh / batch jobs) and leaves artifacts on disk.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def load_eci_map(eci_file: Path) -> dict[str, float]:
    # The ECI file is a per-model scalar capability estimate, produced by a separate
    # fitting script. We load it into a simple {model -> eci} mapping for plotting/fits.
    eci_df = pd.read_csv(eci_file)
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def print_eci_table(
    *,
    all_models: list[str],
    eci_map: dict[str, float],
    train_models: set[str],
) -> None:
    print(f"Fitted ECIs for {len(all_models)} models:")
    for model in sorted(all_models, key=lambda m: eci_map.get(m, 0)):
        eci = eci_map.get(model)
        split = "TRAIN" if model in train_models else "TEST"
        if eci is not None and not (isinstance(eci, float) and np.isnan(eci)):
            print(f"  [{split:5s}] {model:35s} {eci:6.1f}")
        else:
            print(f"  [{split:5s}] {model:35s} MISSING")


def load_and_prepare_results_df(
    *,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
    all_models: list[str],
    hint_fractions: list[float],
    eci_map: dict[str, float],
    train_models: set[str],
) -> pd.DataFrame:
    # Import inside the function so that importing this module (e.g. for reuse) does
    # not immediately require all modelx dependencies / sys.path side effects.
    from src.modelx import load_results

    print("\nLoading results...")
    df = load_results(
        base_folder=str(base_folder),
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )

    df = df[df["model"].isin(all_models) & df["hint"].isin(hint_fractions)]
    print(f"Loaded {len(df)} rows for {df['model'].nunique()} models")

    df = df.copy()
    df["eci"] = df["model"].map(eci_map)

    missing = df[df["eci"].isna()]["model"].unique()
    if len(missing) > 0:
        print(f"WARNING: Missing ECI for {len(missing)} models: {missing.tolist()}")
        df = df.dropna(subset=["eci"])

    df["split"] = df["model"].apply(lambda m: "train" if m in train_models else "test")
    return df


def format_joint_latex(result: dict, hint_transform: Callable = lambda h: h) -> str:
    """Format joint fit result as LaTeX equation."""
    params = result["params"]
    lower = result.get("lower")

    if result["include_cross"]:
        alpha, beta, gamma, delta = params
        sig = rf"\sigma({alpha:.3f} \cdot C + {beta:.3f} \cdot h + {gamma:.3f} \cdot Ch + {delta:.2f})"
    else:
        alpha, beta, delta = params
        sig = rf"\sigma({alpha:.3f} \cdot C + {beta:.3f} \cdot h + {delta:.2f})"

    if lower is not None:
        return rf"${lower:.2f} + {1-lower:.2f} \cdot {sig}$"
    else:
        return rf"${sig}$"


def fit_individual_sigmoids_by_hint(df: pd.DataFrame, eci_map: dict, fit_models: set[str] | None = None, lower: float | None = None) -> dict[float, dict]:
    """Fit individual sigmoids σ(αC + β) for each hint value using fit_sigmoid.

    Args:
        df: DataFrame with model, hint, accuracy columns
        eci_map: Dict mapping model name to ECI value
        fit_models: If provided, only fit on these models (train set)
        lower: If set, pin lower asymptote (fits L + (1-L)*σ(...))
    """
    results = {}
    for hint in sorted(df["hint"].unique()):
        hint_df = df[df["hint"] == hint]
        if fit_models is not None:
            hint_df = hint_df[hint_df["model"].isin(fit_models)]
        if len(hint_df) < 3:
            continue

        C = hint_df["model"].map(eci_map).values
        y = hint_df["accuracy"].values
        if np.allclose(y, y[0]):
            # Degenerate case: y is (nearly) constant across models at this hint.
            # A sigmoid fit is ill-posed here (infinitely many parameter settings
            # produce ~the same constant), so we skip.
            continue

        # Import locally so this helper stays usable even if you import this module
        # in a context where `src.modelx` isn't on sys.path.
        from src.modelx.fitting import fit_sigmoid, format_equation

        result = fit_sigmoid(C, y, use_log=False, lower=lower)
        if lower is not None:
            # asymptote type: params = (L, U, m, b)
            _L, _U, m, b = result["params"]
            midpoint = -b / m
        else:
            # basic type: params = (m, b)
            m, b = result["params"]
            midpoint = -b / m

        results[hint] = {
            "params": result["params"],
            "midpoint": midpoint,
            "predict": result["predict"],
            "equation": format_equation(result),
        }

    return results


def fit_individual_sigmoids_by_model(df: pd.DataFrame, hint_transform: Callable = lambda h: h, fit_models: set[str] | None = None, lower: float | None = None) -> dict[str, dict]:
    """Fit individual sigmoids σ(βh + γ) for each model using fit_sigmoid.

    Args:
        df: DataFrame with model, hint, accuracy columns
        hint_transform: Function to transform hint values
        fit_models: If provided, only fit on these models (train set)
        lower: If set, pin lower asymptote (fits L + (1-L)*σ(...))
    """
    models_to_fit = fit_models if fit_models is not None else set(df["model"].unique())
    results = {}
    for model in sorted(models_to_fit):
        model_df = df[df["model"] == model]
        if len(model_df) < 3:
            continue

        H = np.array([hint_transform(h) for h in model_df["hint"].values])
        y = model_df["accuracy"].values
        if np.allclose(y, y[0]):
            # Same degeneracy logic as above: if accuracy doesn't change with hint,
            # a sigmoid in hint-space is not identifiable / not useful.
            continue

        from src.modelx.fitting import fit_sigmoid, format_equation

        result = fit_sigmoid(H, y, use_log=False, lower=lower)
        results[model] = {
            "params": result["params"],
            "predict": result["predict"],
            "equation": format_equation(result),
        }

    return results


def compute_midpoint_errors(joint_result: dict, individual_fits: dict[float, dict], hints: list[float], hint_transform: Callable = lambda h: h) -> dict[float, float]:
    """Compute per-hint error between joint fit midpoint and individual fit midpoints.

    Returns:
        Dict mapping hint -> midpoint error (absolute difference in ECI units)
    """
    errors = {}
    params = joint_result["params"]

    for hint in hints:
        if hint not in individual_fits:
            continue
        individual_midpoint = individual_fits[hint]["midpoint"]

        # Find the joint-fit "midpoint" in C at this hint:
        # we solve for C such that predict(C, hint) = 0.5.
        #
        # For a plain sigmoid σ(z), σ(z)=0.5 happens at z=0, so we set the model's
        # logit (linear term inside σ) to zero and solve algebraically for C.
        #
        # With cross term: σ(αC + βh + γCh + δ) = 0.5  =>  αC + βh + γCh + δ = 0
        # C(α + γh) = -βh - δ
        # C = (-βh - δ) / (α + γh)
        h_t = hint_transform(hint)
        if joint_result["include_cross"]:
            alpha, beta, gamma, delta = params
            if abs(alpha + gamma * h_t) > 1e-6:
                joint_midpoint = (-beta * h_t - delta) / (alpha + gamma * h_t)
            else:
                continue
        else:
            alpha, beta, delta = params
            if abs(alpha) > 1e-6:
                joint_midpoint = (-beta * h_t - delta) / alpha
            else:
                continue

        errors[hint] = abs(joint_midpoint - individual_midpoint)

    return errors


def compute_rms(joint_result: dict, df: pd.DataFrame, eci_map: dict, models: set[str] | None = None) -> float:
    """Compute RMS error for joint fit on specified models.

    Args:
        joint_result: Result from fit_joint_sigmoid
        df: DataFrame with model, hint, accuracy columns
        eci_map: Dict mapping model name to ECI value
        models: Set of models to evaluate on (None = all models in df)

    Returns:
        RMS error
    """
    eval_df = df if models is None else df[df["model"].isin(models)]
    if len(eval_df) == 0:
        return float("nan")
    # Predict is a callable stored in the joint fit output; it expects (eci, hint).
    y_pred = np.array([joint_result["predict"](eci_map[m], h) for m, h in zip(eval_df["model"], eval_df["hint"])])
    y_actual = eval_df["accuracy"].values
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))


def run_model_sweep(
    df: pd.DataFrame,
    eci_map: dict,
    hint_fractions: list[float],
    hint_transform: Callable,
    include_cross: bool,
    lower_asymptote: float | None,
    eval_hints: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep over number of train models (sorted by ECI) and compute metrics.

    Args:
        df: DataFrame with model, hint, accuracy columns
        eci_map: Dict mapping model name to ECI value
        hint_fractions: Hint values to include
        hint_transform: Function to transform hint values
        include_cross: Whether to include cross term in joint fit
        lower_asymptote: Lower asymptote for sigmoid fits
        eval_hints: Hint values to compute midpoint errors for (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    Returns:
        DataFrame with columns: n_models, rms_train, rms_test, rms_all, midpoint_h_XX for each eval hint
    """
    if eval_hints is None:
        eval_hints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Sort models by ECI (lowest to highest)
    all_models = sorted(df["model"].unique(), key=lambda m: eci_map.get(m, 0))

    # Fit individual sigmoids on ALL models (ground truth for midpoint comparison)
    individual_by_hint_all = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=None, lower=lower_asymptote)

    results = []
    for n in range(5, len(all_models) + 1):  # Start at 5 models for stable fits
        train_models = set(all_models[:n])
        test_models = set(all_models[n:])

        # Import locally so importing this script doesn't immediately pull in the
        # fitting stack; also keeps this helper usable from other modules.
        from src.modelx.fitting import fit_joint_sigmoid

        joint_result = fit_joint_sigmoid(
            df,
            y_col="accuracy",
            hint_col="hint",
            include_cross=include_cross,
            hint_transform=hint_transform,
            use_log_x=False,
            x_values=eci_map,
            fit_models=train_models,
            lower=lower_asymptote,
        )

        # Compute RMS
        rms_train = compute_rms(joint_result, df, eci_map, train_models)
        rms_test = compute_rms(joint_result, df, eci_map, test_models) if test_models else float("nan")
        rms_all = compute_rms(joint_result, df, eci_map, None)

        # Compute midpoint errors for eval hints
        midpoint_errors = compute_midpoint_errors(joint_result, individual_by_hint_all, eval_hints, hint_transform)

        row = {
            "n_models": n,
            "rms_train": rms_train,
            "rms_test": rms_test,
            "rms_all": rms_all,
        }
        for h in eval_hints:
            row[f"midpoint_h_{h:.1f}"] = midpoint_errors.get(h, float("nan"))

        results.append(row)

    return pd.DataFrame(results)

def plot_accuracy_vs_eci_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict,
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    print("\n" + "=" * 70)
    print("PLOT 1: Accuracy vs ECI for each hint level")
    print(f"Fitting: {joint_latex}")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(12, 7))
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

    # Slightly pad the x-range so curves don't clip at the first/last model.
    eci_range = np.linspace(df["eci"].min() - 5, df["eci"].max() + 5, 100)
    for hint in hints:
        hint_df = df[df["hint"] == hint].sort_values("eci")
        train_df = hint_df[hint_df["split"] == "train"]
        ax.scatter(
            train_df["eci"],
            train_df["accuracy"],
            color=colors[hint],
            label=f"h={hint:.2f}",
            alpha=0.8,
            s=60,
            marker="o",
        )

        test_df = hint_df[hint_df["split"] == "test"]
        if len(test_df) > 0:
            # Use a different marker for test points to visually separate splits.
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color=colors[hint],
                alpha=0.8,
                s=60,
                marker="s",
                edgecolors="black",
            )

        y_fit = [joint_result["predict"](c, hint) for c in eci_range]
        ax.plot(eci_range, y_fit, "-", color=colors[hint], alpha=0.5, linewidth=2)

    ax.set_xlabel("eci", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_title(f"{label}\n{joint_latex}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "accuracy_vs_eci_by_hint.png"
    save_figure(fig, out_path)
    print(f"Saved: {out_path}")
    return out_path


def plot_individual_fits_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict,
    individual_by_hint: dict[float, dict],
    individual_by_hint_train: dict[float, dict],
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    print("\n" + "=" * 70)
    print("PLOT 2: Individual sigmoid fits per hint level")
    print("Comparing joint fit (dashed) vs individual fits (solid)")
    print("=" * 70)

    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}
    eci_range = np.linspace(df["eci"].min() - 5, df["eci"].max() + 5, 100)

    n_rows = 3
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, hint in enumerate(hints):
        ax = axes[i]
        hint_df = df[df["hint"] == hint].sort_values("eci")
        train_df = hint_df[hint_df["split"] == "train"]
        ax.scatter(train_df["eci"], train_df["accuracy"], color=colors[hint], alpha=0.8, s=40)

        test_df = hint_df[hint_df["split"] == "test"]
        if len(test_df) > 0:
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color=colors[hint],
                alpha=0.8,
                s=40,
                marker="s",
                edgecolors="black",
            )

        y_joint = [joint_result["predict"](c, hint) for c in eci_range]
        ax.plot(eci_range, y_joint, "--", color="gray", linewidth=2, label="joint (train)")

        if hint in individual_by_hint_train:
            y_indiv_train = [individual_by_hint_train[hint]["predict"](c) for c in eci_range]
            ax.plot(eci_range, y_indiv_train, "-", color="orange", linewidth=2, label="indiv (train)")

        if hint in individual_by_hint:
            y_indiv_all = [individual_by_hint[hint]["predict"](c) for c in eci_range]
            ax.plot(eci_range, y_indiv_all, "-", color=colors[hint], linewidth=2, label="indiv (all)")
            midpoint = individual_by_hint[hint]["midpoint"]
            # Midpoint in C for the per-hint fit (where predicted accuracy crosses 0.5).
            ax.axvline(midpoint, color=colors[hint], linestyle=":", alpha=0.5)

        ax.set_title(f"h = {hint:.2f}", fontsize=11)
        ax.set_xlabel("eci")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(len(hints), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{label} - Individual fits per hint\nJoint: {joint_latex}", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / "individual_fits_by_hint.png"
    save_figure(fig, out_path)
    print(f"Saved: {out_path}")
    return out_path


def plot_accuracy_vs_hint_by_model(
    *,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    joint_result: dict,
    individual_by_model: dict[str, dict],
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    print("\n" + "=" * 70)
    print("PLOT 3: Accuracy vs Hint for each model")
    print("=" * 70)

    models_sorted = sorted(df["model"].unique(), key=lambda m: eci_map.get(m, 0))
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    hint_range = np.linspace(0, 1, 100)
    model_cmap = plt.cm.coolwarm
    model_colors = {m: model_cmap(i / max(n_models - 1, 1)) for i, m in enumerate(models_sorted)}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        eci = eci_map.get(model, 0.0)

        ax.scatter(model_df["hint"], model_df["accuracy"], color=model_colors[model], alpha=0.8, s=40)

        y_joint = [joint_result["predict"](eci, h) for h in hint_range]
        ax.plot(hint_range, y_joint, "--", color="gray", linewidth=2, label="joint fit")

        if model in individual_by_model:
            # Note: the per-model fit is in "hint space" and uses HINT_TRANSFORM
            # internally (because we passed `H = hint_transform(h)` into fit_sigmoid).
            y_indiv = [individual_by_model[model]["predict"](h) for h in hint_range]
            ax.plot(hint_range, y_indiv, "-", color=model_colors[model], linewidth=2, label="individual fit")

        ax.set_title(f"{model}\neci={eci:.1f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{label} - Accuracy vs Hint per model\nJoint: {joint_latex}", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / "accuracy_vs_hint_by_model.png"
    save_figure(fig, out_path)
    print(f"Saved: {out_path}")
    return out_path


def plot_model_sweep(
    *,
    sweep_df: pd.DataFrame,
    label: str,
    eval_hints: list[float],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(sweep_df["n_models"], sweep_df["rms_train"], "o-", label="train", color="blue")
    ax.plot(sweep_df["n_models"], sweep_df["rms_test"], "s-", label="test", color="red")
    ax.plot(sweep_df["n_models"], sweep_df["rms_all"], "^-", label="all", color="green")
    ax.set_xlabel("number of train models")
    ax.set_ylabel("rms")
    ax.set_title("RMS vs number of train models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(eval_hints) - 1, 1)) for i, h in enumerate(eval_hints)}
    for h in eval_hints:
        col = f"midpoint_h_{h:.1f}"
        if col in sweep_df.columns:
            ax.plot(sweep_df["n_models"], sweep_df[col], "o-", label=f"h={h:.1f}", color=colors[h])
    ax.set_xlabel("number of train models")
    ax.set_ylabel("midpoint error (eci units)")
    ax.set_title("midpoint error vs number of train models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} (fitting joint scaling)", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / "model_sweep.png"
    save_figure(fig, out_path)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    train_models = set(TRAIN_MODELS)
    test_models = set(TEST_MODELS)

    # Create output directory for plots (so running as a script leaves artifacts).
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    eci_map = load_eci_map(ECI_FILE)
    print_eci_table(all_models=ALL_MODELS, eci_map=eci_map, train_models=train_models)

    df = load_and_prepare_results_df(
        base_folder=BASE_FOLDER,
        eval_name=EVAL_NAME,
        solver=SOLVER,
        condition="0shot",
        all_models=ALL_MODELS,
        hint_fractions=HINT_FRACTIONS,
        eci_map=eci_map,
        train_models=train_models,
    )

    from src.modelx.fitting import fit_joint_sigmoid, format_equation

    print("\n" + "=" * 70)
    print("JOINT SCALING LAW FIT")
    print("=" * 70)
    print(f"Training on {len(train_models)} models, testing on {len(test_models)} models")

    joint_result = fit_joint_sigmoid(
        df,
        y_col="accuracy",
        hint_col="hint",
        include_cross=INCLUDE_CROSS,
        hint_transform=HINT_TRANSFORM,
        use_log_x=False,
        x_values=eci_map,
        fit_models=train_models,
        lower=LOWER_ASYMPTOTE,
    )

    joint_latex = format_joint_latex(joint_result, HINT_TRANSFORM)
    joint_equation = format_equation(joint_result)
    print(f"\nEquation: {joint_equation}")
    print(f"LaTeX: {joint_latex}")

    train_df_fit = df[df["model"].isin(train_models)]
    y_pred_train = np.array(
        [
            joint_result["predict"](eci_map[m], h)
            for m, h in zip(train_df_fit["model"], train_df_fit["hint"])
        ]
    )
    y_actual_train = train_df_fit["accuracy"].values
    train_rms = np.sqrt(np.mean((y_actual_train - y_pred_train) ** 2))
    print(f"RMS (train, n={len(train_models)}): {train_rms:.4f}")

    test_models_in_data = set(df["model"].unique()) - train_models
    # `TEST_MODELS` is currently unused (often empty), so we infer "test" models as
    # everything in the dataframe not included in `train_models`.
    if test_models_in_data:
        test_df_fit = df[df["model"].isin(test_models_in_data)]
        y_pred_test = np.array(
            [
                joint_result["predict"](eci_map[m], h)
                for m, h in zip(test_df_fit["model"], test_df_fit["hint"])
            ]
        )
        y_actual_test = test_df_fit["accuracy"].values
        test_rms = np.sqrt(np.mean((y_actual_test - y_pred_test) ** 2))
        print(f"RMS (test, n={len(test_models_in_data)}): {test_rms:.4f}")

        n_train = len(train_models)
        n_test = len(test_models_in_data)
        weighted_rms = (n_train * train_rms + n_test * test_rms) / (n_train + n_test)
        print(f"RMS (weighted avg): {weighted_rms:.4f}")

    individual_by_hint = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=None, lower=LOWER_ASYMPTOTE)
    midpoint_errors_all = compute_midpoint_errors(joint_result, individual_by_hint, HINT_FRACTIONS, HINT_TRANSFORM)
    mean_midpoint_error_all = np.mean(list(midpoint_errors_all.values())) if midpoint_errors_all else float("nan")
    print(f"\nMean midpoint error (all models): {mean_midpoint_error_all:.2f}")
    if 0.0 in midpoint_errors_all:
        print(f"  h=0.00 (baseline): {midpoint_errors_all[0.0]:.2f}")

    individual_by_hint_train = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=train_models, lower=LOWER_ASYMPTOTE)
    midpoint_errors_train = compute_midpoint_errors(joint_result, individual_by_hint_train, HINT_FRACTIONS, HINT_TRANSFORM)
    mean_midpoint_error_train = np.mean(list(midpoint_errors_train.values())) if midpoint_errors_train else float("nan")
    print(f"\nMean midpoint error (train, n={len(train_models)}): {mean_midpoint_error_train:.2f}")
    if 0.0 in midpoint_errors_train:
        print(f"  h=0.00 (baseline, train): {midpoint_errors_train[0.0]:.2f}")

    if test_models_in_data:
        individual_by_hint_test = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=test_models_in_data, lower=LOWER_ASYMPTOTE)
        midpoint_errors_test = compute_midpoint_errors(joint_result, individual_by_hint_test, HINT_FRACTIONS, HINT_TRANSFORM)
        mean_midpoint_error_test = np.mean(list(midpoint_errors_test.values())) if midpoint_errors_test else float("nan")
        print(f"Mean midpoint error (test, n={len(test_models_in_data)}): {mean_midpoint_error_test:.2f}")
        if 0.0 in midpoint_errors_test:
            print(f"  h=0.00 (baseline, test): {midpoint_errors_test[0.0]:.2f}")

        n_train = len(train_models)
        n_test = len(test_models_in_data)
        weighted_midpoint_error = (n_train * mean_midpoint_error_train + n_test * mean_midpoint_error_test) / (n_train + n_test)
        print(f"Mean midpoint error (weighted avg): {weighted_midpoint_error:.2f}")

    plot_accuracy_vs_eci_by_hint(
        df=df,
        joint_result=joint_result,
        label=LABEL,
        joint_latex=joint_latex,
        output_dir=PLOTS_DIR,
    )
    plot_individual_fits_by_hint(
        df=df,
        joint_result=joint_result,
        individual_by_hint=individual_by_hint,
        individual_by_hint_train=individual_by_hint_train,
        label=LABEL,
        joint_latex=joint_latex,
        output_dir=PLOTS_DIR,
    )

    individual_by_model = fit_individual_sigmoids_by_model(df, HINT_TRANSFORM, fit_models=None, lower=LOWER_ASYMPTOTE)
    plot_accuracy_vs_hint_by_model(
        df=df,
        eci_map=eci_map,
        joint_result=joint_result,
        individual_by_model=individual_by_model,
        label=LABEL,
        joint_latex=joint_latex,
        output_dir=PLOTS_DIR,
    )

    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    print(f"Joint fit equation: {joint_equation}")
    print(f"Joint fit LaTeX: {joint_latex}")
    print(f"RMS: {joint_result['rms']:.4f}")
    print(f"Mean midpoint error: {mean_midpoint_error_all:.2f}")
    print(f"Number of train models: {len(train_models)}")
    print(f"Number of test models: {len(test_models)}")
    print(f"Cross term included: {INCLUDE_CROSS}")

    print("\n" + "=" * 70)
    print("MODEL SWEEP ANALYSIS")
    print("Sweeping number of train models (sorted by ECI, lowest to highest)")
    print("=" * 70)

    sweep_df = run_model_sweep(
        df,
        eci_map,
        HINT_FRACTIONS,
        HINT_TRANSFORM,
        INCLUDE_CROSS,
        LOWER_ASYMPTOTE,
        eval_hints=EVAL_HINTS_FOR_SWEEP,
    )
    print(f"Sweep complete: {len(sweep_df)} configurations")
    plot_model_sweep(
        sweep_df=sweep_df,
        label=LABEL,
        eval_hints=EVAL_HINTS_FOR_SWEEP,
        output_dir=PLOTS_DIR,
    )


if __name__ == "__main__":
    main()


