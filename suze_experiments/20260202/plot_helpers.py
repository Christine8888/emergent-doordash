"""Plotting helpers for 20260202 experiments.

This module is intended to be imported from `experiments.py` and contains the
reusable plotting + fitting logic (as opposed to experiment-specific configs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------ constants ------------------------------

DEFAULT_FIG_DPI = 200
DEFAULT_FIG_BBOX_INCHES = "tight"
JSON_INDENT = 2


# ------------------------------ hint transforms ------------------------------

def hint_identity(h: float) -> float:
    return h


def hint_logit(h: float) -> float:
    return math.log(h / (1.0 - h))


HINT_TRANSFORMS: dict[str, Callable[[float], float]] = {
    "identity": hint_identity,
    "logit": hint_logit,
}


# ------------------------------ io helpers ------------------------------

def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=JSON_INDENT, sort_keys=True) + "\n")


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save a matplotlib figure and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_FIG_DPI, bbox_inches=DEFAULT_FIG_BBOX_INCHES)
    plt.close(fig)


# ------------------------------ data loading ------------------------------

def load_eci_map(eci_file: Path) -> dict[str, float]:
    """Load per-model ECI values into a {model -> eci} dict."""
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
    # Import inside the function so that importing this module does not require
    # extra sys.path mutations.
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


# ------------------------------ fitting helpers ------------------------------

def format_joint_latex(result: dict, hint_transform: Callable[[float], float] = lambda h: h) -> str:
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
    return rf"${sig}$"


def fit_individual_sigmoids_by_hint(
    df: pd.DataFrame,
    eci_map: dict[str, float],
    *,
    fit_models: set[str] | None = None,
    lower: float | None = None,
) -> dict[float, dict]:
    """Fit σ(αC + β) for each hint value using `src.modelx.fitting.fit_sigmoid`."""
    results: dict[float, dict] = {}
    for hint in sorted(df["hint"].unique()):
        hint_df = df[df["hint"] == hint]
        if fit_models is not None:
            hint_df = hint_df[hint_df["model"].isin(fit_models)]
        if len(hint_df) < 3:
            continue

        C = hint_df["model"].map(eci_map).values
        y = hint_df["accuracy"].values
        if np.allclose(y, y[0]):
            # Degenerate: constant target => non-identifiable sigmoid.
            continue

        from src.modelx.fitting import fit_sigmoid, format_equation

        result = fit_sigmoid(C, y, use_log=False, lower=lower)
        if lower is not None:
            # asymptote params: (L, U, m, b)
            _L, _U, m, b = result["params"]
            midpoint = -b / m
        else:
            # basic params: (m, b)
            m, b = result["params"]
            midpoint = -b / m

        results[hint] = {
            "params": result["params"],
            "midpoint": midpoint,
            "predict": result["predict"],
            "equation": format_equation(result),
        }

    return results


def fit_individual_sigmoids_by_model(
    df: pd.DataFrame,
    hint_transform: Callable[[float], float] = lambda h: h,
    *,
    fit_models: set[str] | None = None,
    lower: float | None = None,
) -> dict[str, dict]:
    """Fit σ(βh + γ) for each model using `src.modelx.fitting.fit_sigmoid`."""
    models_to_fit = fit_models if fit_models is not None else set(df["model"].unique())
    results: dict[str, dict] = {}
    for model in sorted(models_to_fit):
        model_df = df[df["model"] == model]
        if len(model_df) < 3:
            continue

        H = np.array([hint_transform(h) for h in model_df["hint"].values])
        y = model_df["accuracy"].values
        if np.allclose(y, y[0]):
            continue

        from src.modelx.fitting import fit_sigmoid, format_equation

        result = fit_sigmoid(H, y, use_log=False, lower=lower)
        results[model] = {
            "params": result["params"],
            "predict": result["predict"],
            "equation": format_equation(result),
        }

    return results


def compute_midpoint_errors(
    joint_result: dict,
    individual_fits: dict[float, dict],
    hints: list[float],
    hint_transform: Callable[[float], float] = lambda h: h,
) -> dict[float, float]:
    """Compute |C_midpoint(joint, h) - C_midpoint(individual, h)| per hint."""
    errors: dict[float, float] = {}
    params = joint_result["params"]

    for hint in hints:
        if hint not in individual_fits:
            continue

        individual_midpoint = individual_fits[hint]["midpoint"]
        h_t = hint_transform(hint)

        if joint_result["include_cross"]:
            alpha, beta, gamma, delta = params
            denom = alpha + gamma * h_t
            if abs(denom) <= 1e-6:
                continue
            joint_midpoint = (-beta * h_t - delta) / denom
        else:
            alpha, beta, delta = params
            if abs(alpha) <= 1e-6:
                continue
            joint_midpoint = (-beta * h_t - delta) / alpha

        errors[hint] = abs(joint_midpoint - individual_midpoint)

    return errors


def compute_rms(
    joint_result: dict,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    models: set[str] | None = None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if len(eval_df) == 0:
        return float("nan")
    y_pred = np.array(
        [
            joint_result["predict"](eci_map[m], h)
            for m, h in zip(eval_df["model"], eval_df["hint"])
        ]
    )
    y_actual = eval_df["accuracy"].values
    return float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))


def run_model_sweep(
    df: pd.DataFrame,
    eci_map: dict[str, float],
    hint_fractions: list[float],
    hint_transform: Callable[[float], float],
    include_cross: bool,
    lower_asymptote: float | None,
    *,
    eval_hints: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep number of train models (sorted by ECI) and compute RMS + midpoint errors."""
    if eval_hints is None:
        eval_hints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    all_models = sorted(df["model"].unique(), key=lambda m: eci_map.get(m, 0))
    individual_by_hint_all = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=None, lower=lower_asymptote)

    from src.modelx.fitting import fit_joint_sigmoid

    rows: list[dict[str, float]] = []
    for n in range(5, len(all_models) + 1):
        train_models = set(all_models[:n])
        test_models = set(all_models[n:])

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

        rms_train = compute_rms(joint_result, df, eci_map, train_models)
        rms_test = compute_rms(joint_result, df, eci_map, test_models) if test_models else float("nan")
        rms_all = compute_rms(joint_result, df, eci_map, None)

        midpoint_errors = compute_midpoint_errors(joint_result, individual_by_hint_all, eval_hints, hint_transform)
        row: dict[str, float] = {"n_models": float(n), "rms_train": rms_train, "rms_test": rms_test, "rms_all": rms_all}
        for h in eval_hints:
            row[f"midpoint_h_{h:.1f}"] = float(midpoint_errors.get(h, float("nan")))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["n_models"] = out["n_models"].astype(int)
    return out


# ------------------------------ plotting ------------------------------

def plot_accuracy_vs_eci_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict,
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

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
    return out_path


# ------------------------------ orchestration ------------------------------

def run_joint_scaling_plots(
    *,
    base_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    label: str,
    all_models: list[str],
    num_holdout_models: int,
    hint_fractions: list[float],
    eval_hints_for_sweep: list[float],
    include_cross: bool,
    lower_asymptote: float | None,
    hint_transform: str | Callable[[float], float],
    output_dir: Path,
) -> dict[str, object]:
    """Run the joint scaling analysis and write artifacts to `output_dir`.

    Split logic:
    - Fit is performed on a train set computed by holding out the top
      `num_holdout_models` models by ECI (highest ECIs are held out).
    - The holdout set is used as the "test" set for summary metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write plots directly into the run directory (no nested `plots/`).
    plots_dir = output_dir

    if isinstance(hint_transform, str):
        hint_transform_name = hint_transform
        if hint_transform_name not in HINT_TRANSFORMS:
            raise ValueError(
                f"Unknown hint_transform={hint_transform_name!r}. "
                f"Choose one of: {sorted(HINT_TRANSFORMS.keys())}"
            )
        hint_transform_fn = HINT_TRANSFORMS[hint_transform_name]
    else:
        hint_transform_fn = hint_transform
        hint_transform_name = getattr(hint_transform_fn, "__name__", "custom")

    eci_map = load_eci_map(eci_file)

    if num_holdout_models < 0:
        raise ValueError(f"num_holdout_models must be >= 0, got {num_holdout_models}")
    if num_holdout_models > len(all_models):
        raise ValueError(
            f"num_holdout_models ({num_holdout_models}) cannot exceed number of models ({len(all_models)})"
        )

    models_sorted_by_eci = sorted(all_models, key=lambda m: eci_map.get(m, float("-inf")))
    holdout_set = set(models_sorted_by_eci[-num_holdout_models:]) if num_holdout_models > 0 else set()
    train_set = set(models_sorted_by_eci[:-num_holdout_models]) if num_holdout_models > 0 else set(all_models)

    print_eci_table(all_models=all_models, eci_map=eci_map, train_models=train_set)

    df = load_and_prepare_results_df(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
        eci_map=eci_map,
        train_models=train_set,
    )

    from src.modelx.fitting import fit_joint_sigmoid, format_equation

    joint_result = fit_joint_sigmoid(
        df,
        y_col="accuracy",
        hint_col="hint",
        include_cross=include_cross,
        hint_transform=hint_transform_fn,
        use_log_x=False,
        x_values=eci_map,
        fit_models=train_set,
        lower=lower_asymptote,
    )

    joint_latex = format_joint_latex(joint_result, hint_transform_fn)
    joint_equation = format_equation(joint_result)

    # Holdout/test set is the top-ECI models.
    test_set = holdout_set

    rms_train = compute_rms(joint_result, df, eci_map, train_set)
    rms_test = compute_rms(joint_result, df, eci_map, test_set) if test_set else float("nan")
    rms_all = compute_rms(joint_result, df, eci_map, None)

    individual_by_hint_all = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=None, lower=lower_asymptote)
    individual_by_hint_train = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=train_set, lower=lower_asymptote)
    individual_by_model = fit_individual_sigmoids_by_model(df, hint_transform_fn, fit_models=None, lower=lower_asymptote)

    midpoint_errors_all = compute_midpoint_errors(joint_result, individual_by_hint_all, hint_fractions, hint_transform_fn)
    midpoint_errors_train = compute_midpoint_errors(joint_result, individual_by_hint_train, hint_fractions, hint_transform_fn)
    mean_midpoint_error_all = float(np.mean(list(midpoint_errors_all.values()))) if midpoint_errors_all else float("nan")
    mean_midpoint_error_train = float(np.mean(list(midpoint_errors_train.values()))) if midpoint_errors_train else float("nan")

    if test_set:
        individual_by_hint_test = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=test_set, lower=lower_asymptote)
        midpoint_errors_test = compute_midpoint_errors(joint_result, individual_by_hint_test, hint_fractions, hint_transform_fn)
        mean_midpoint_error_test = float(np.mean(list(midpoint_errors_test.values()))) if midpoint_errors_test else float("nan")
    else:
        mean_midpoint_error_test = float("nan")

    plot_accuracy_vs_eci_by_hint(
        df=df,
        joint_result=joint_result,
        label=label,
        joint_latex=joint_latex,
        output_dir=plots_dir,
    )
    plot_individual_fits_by_hint(
        df=df,
        joint_result=joint_result,
        individual_by_hint=individual_by_hint_all,
        individual_by_hint_train=individual_by_hint_train,
        label=label,
        joint_latex=joint_latex,
        output_dir=plots_dir,
    )
    plot_accuracy_vs_hint_by_model(
        df=df,
        eci_map=eci_map,
        joint_result=joint_result,
        individual_by_model=individual_by_model,
        label=label,
        joint_latex=joint_latex,
        output_dir=plots_dir,
    )

    sweep_df = run_model_sweep(
        df,
        eci_map,
        hint_fractions,
        hint_transform_fn,
        include_cross,
        lower_asymptote,
        eval_hints=eval_hints_for_sweep,
    )
    plot_model_sweep(
        sweep_df=sweep_df,
        label=label,
        eval_hints=eval_hints_for_sweep,
        output_dir=plots_dir,
    )

    config_resolved: dict[str, object] = {
        "output_dir": str(output_dir),
        "base_folder": str(base_folder),
        "eci_file": str(eci_file),
        "eval_name": eval_name,
        "solver": solver,
        "condition": condition,
        "label": label,
        "all_models": all_models,
        "num_holdout_models": int(num_holdout_models),
        "train_models": sorted(train_set, key=lambda m: eci_map.get(m, float("-inf"))),
        "holdout_models": sorted(test_set, key=lambda m: eci_map.get(m, float("-inf"))),
        "hint_fractions": hint_fractions,
        "eval_hints_for_sweep": eval_hints_for_sweep,
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "hint_transform": hint_transform_name,
    }

    metrics: dict[str, object] = {
        "joint_equation": joint_equation,
        "joint_latex": joint_latex,
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "rms_train": rms_train,
        "rms_test": rms_test,
        "rms_all": rms_all,
        "mean_midpoint_error_all": mean_midpoint_error_all,
        "mean_midpoint_error_train": mean_midpoint_error_train,
        "mean_midpoint_error_test": mean_midpoint_error_test,
        "n_train_models": int(len(train_set)),
        "n_test_models": int(len(test_set)),
        "joint_params": [float(x) for x in joint_result["params"]],
        "config": config_resolved,
    }

    write_json(output_dir / "config_resolved.json", config_resolved)
    write_json(output_dir / "metrics.json", metrics)
    return metrics

