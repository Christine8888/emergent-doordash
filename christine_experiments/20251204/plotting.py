"""Plotting utilities for hint fraction experiments.

All plotting functions take DataFrames as input. Use src.modelx.load_results()
to load data into DataFrames.

Expected DataFrame columns:
- model: Model name (e.g., "Qwen2.5-7B-Instruct")
- model_size: Model size in billions (auto-populated by load_results)
- hint: Hint fraction (0.0 to 1.0)
- accuracy: Accuracy value
- stderr: Standard error (optional)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable, Tuple
import seaborn as sns

from src.modelx import size, clean_name, fit_sigmoid, fit_joint_sigmoid, format_equation

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def plot_accuracy_vs_hint(
    df: pd.DataFrame,
    title: str = "Accuracy vs Hint Fraction",
    figsize: Tuple[int, int] = (12, 7),
):
    """Plot accuracy vs hint fraction, one line per model.

    Args:
        df: DataFrame with columns: model, hint, accuracy, stderr (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = sorted(df["model"].unique(), key=lambda m: size(m))
    colors = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        model_df = df[df["model"] == model].sort_values("hint")
        yerr = model_df["stderr"].fillna(0).values if "stderr" in model_df.columns else None
        ax.errorbar(
            model_df["hint"].values,
            model_df["accuracy"].values,
            yerr=yerr,
            color=colors[i],
            linestyle="none",
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=2,
            alpha=0.8,
            label=clean_name(model),
        )

    hints = df["hint"].unique()
    ax.legend(loc="upper left", title="Models", framealpha=0.9)
    ax.set_xlabel("fraction of reasoning chain as hint", fontsize=13, fontweight="bold")
    ax.set_ylabel("eval accuracy", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(hints) + 0.05)

    plt.tight_layout()
    return fig, ax


def plot_error_vs_hint_transformed(
    df: pd.DataFrame,
    title: str = "Error Rate vs Transformed Hint",
    figsize: Tuple[int, int] = (12, 7),
    fit_scaling: bool = False,
    lower_asymptote_hint: float | None = None,
    upper_asymptote_hint: float | None = None,
    upper_bound: float = 1.0,
    hint_transform: Callable[[float], float] | None = None,
    x_label: str | None = None,
):
    """Plot error rate (1-accuracy) vs transformed hint with sigmoid fitting.

    Args:
        df: DataFrame with columns: model, hint, accuracy, stderr (optional)
        title: Plot title
        figsize: Figure size
        fit_scaling: If True, fit y = h*σ(m*x + b), else y = σ(m*x + b)
        lower_asymptote_hint: Hint value to use for lower asymptote (e.g., 1.0)
        upper_asymptote_hint: Hint value to use for upper asymptote (e.g., 0.0)
        upper_bound: Default upper bound for sigmoid (default 1.0)
        hint_transform: Function to transform hint values for x-axis
        x_label: X-axis label

    Returns:
        Figure and Axes objects
    """
    from matplotlib.lines import Line2D

    if hint_transform is None:
        hint_transform = lambda h: h
    if x_label is None:
        x_label = "transformed hint"

    fig, ax = plt.subplots(figsize=figsize)

    models = sorted(df["model"].unique(), key=lambda m: size(m))
    colors = sns.color_palette("husl", len(models))
    model_fits = {}

    # Extract asymptotes per model
    lower_asymptotes = {}
    upper_asymptotes = {}
    if lower_asymptote_hint is not None:
        for model in models:
            row = df[(df["model"] == model) & (df["hint"] == lower_asymptote_hint)]
            if len(row) > 0 and pd.notna(row["accuracy"].iloc[0]):
                lower_asymptotes[model] = 1 - row["accuracy"].iloc[0]

    if upper_asymptote_hint is not None:
        for model in models:
            row = df[(df["model"] == model) & (df["hint"] == upper_asymptote_hint)]
            if len(row) > 0 and pd.notna(row["accuracy"].iloc[0]):
                upper_asymptotes[model] = 1 - row["accuracy"].iloc[0]

    for i, model in enumerate(models):
        model_df = df[df["model"] == model].copy()

        # Exclude asymptote hints from plotting
        if lower_asymptote_hint is not None:
            model_df = model_df[model_df["hint"] != lower_asymptote_hint]
        if upper_asymptote_hint is not None:
            model_df = model_df[model_df["hint"] != upper_asymptote_hint]

        # Transform hint and compute error rate
        x_vals, y_vals, y_errs = [], [], []
        for _, row in model_df.iterrows():
            if pd.isna(row["accuracy"]):
                continue
            try:
                x_val = hint_transform(row["hint"])
                if np.isfinite(x_val):
                    x_vals.append(x_val)
                    y_vals.append(1 - row["accuracy"])
                    y_errs.append(row.get("stderr", 0) or 0)
            except (ValueError, ZeroDivisionError):
                pass

        if not x_vals:
            continue

        color = colors[i]
        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_errs,
            color=color,
            linestyle="none",
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=2,
            alpha=0.8,
        )

        # Fit sigmoid
        if len(x_vals) >= 2:
            try:
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)

                L = lower_asymptotes.get(model)
                U = upper_asymptotes.get(model, upper_bound)

                fit_result = fit_sigmoid(
                    x_arr, y_arr,
                    use_log=False,
                    scale=fit_scaling,
                    lower=L,
                    upper=U if L is not None else None,
                )
                model_fits[model] = fit_result

                # Plot fitted curve
                x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
                y_smooth = fit_result["predict"](x_smooth)
                ax.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=2, alpha=0.6)
            except Exception:
                pass

    # Create legend with equations
    model_handles = []
    model_labels = []
    for i, model in enumerate(models):
        label = clean_name(model)
        if model in model_fits:
            equation = format_equation(model_fits[model])
            label = f"{label}: {equation}"
        handle = Line2D([0], [0], color=colors[i], linewidth=2, marker="o", markersize=8)
        model_handles.append(handle)
        model_labels.append(label)

    ax.legend(model_handles, model_labels, title="Models", framealpha=0.9, loc="best")
    ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
    ax.set_ylabel("eval error rate", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_accuracy_vs_size(
    df: pd.DataFrame,
    title: str = "Accuracy vs Model Size",
    figsize: Tuple[int, int] = (12, 7),
    fit_sigmoid_curves: bool = False,
    fit_scaling: bool = False,
    fit_models: list[str] | int | None = None,
    fit_joint: bool = False,
    include_cross: bool = True,
    exclude_hints: list[float] | float | None = None,
    hint_transform: Callable[[float], float] | None = None,
):
    """Plot accuracy vs model size (log scale) with hints as different colors.

    Args:
        df: DataFrame with columns: model, model_size, hint, accuracy, stderr (optional)
        title: Plot title
        figsize: Figure size
        fit_sigmoid_curves: If True, fit per-hint sigmoids: σ(m·log(x) + b)
        fit_scaling: If True, fit y = h·σ(...), else y = σ(...)
        fit_models: Models to use for fitting (None=all, int=first N, list=specific)
        fit_joint: If True, fit joint model: σ(α·C + β·H + γ·C·H + δ)
        include_cross: If True, include cross term γ·C·H in joint fit
        exclude_hints: Hint(s) to exclude from joint fit
        hint_transform: Transform for hint in joint fit

    Returns:
        Figure and Axes objects
    """
    from matplotlib.lines import Line2D

    if hint_transform is None:
        hint_transform = lambda h: h

    # Normalize exclude_hints
    if exclude_hints is None:
        exclude_set = set()
    elif isinstance(exclude_hints, (list, tuple)):
        exclude_set = set(exclude_hints)
    else:
        exclude_set = {exclude_hints}

    fig, ax = plt.subplots(figsize=figsize)

    hints = sorted(df["hint"].unique())
    models = df["model"].unique()
    model_sizes = sorted([(m, size(m)) for m in models], key=lambda x: x[1])

    # Determine fit models
    if fit_models is None:
        fit_model_names = set(models)
    elif isinstance(fit_models, int):
        fit_model_names = set([m for m, _ in model_sizes[:fit_models]])
    else:
        fit_model_names = set(fit_models)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(hints) - 1, 1)) for i in range(len(hints))]

    # Fit joint model if requested
    joint_fit = None
    if fit_joint:
        try:
            joint_fit = fit_joint_sigmoid(
                df,
                include_cross=include_cross,
                hint_transform=hint_transform,
                exclude_hints=exclude_set,
                fit_models=fit_model_names if fit_model_names != set(models) else None,
            )
            params = joint_fit["params"]
            if include_cross:
                print(f"Joint fit: α={params[0]:.3f}, β={params[1]:.3f}, γ={params[2]:.3f}, δ={params[3]:.3f}")
            else:
                print(f"Joint fit: α={params[0]:.3f}, β={params[1]:.3f}, δ={params[2]:.3f}")
            print(f"Average RMS error = {joint_fit['rms']:.4f}")
        except Exception as e:
            print(f"Warning: Failed to fit joint model: {e}")

    hint_fits = {}

    for hint_idx, hint in enumerate(hints):
        x_vals, y_vals, y_errs = [], [], []
        x_vals_fit, y_vals_fit = [], []

        for model, sz in model_sizes:
            row = df[(df["model"] == model) & (df["hint"] == hint)]
            if len(row) > 0:
                acc = row["accuracy"].iloc[0]
                if pd.notna(acc):
                    x_vals.append(sz)
                    y_vals.append(acc)
                    y_errs.append(row["stderr"].iloc[0] if "stderr" in row.columns and pd.notna(row["stderr"].iloc[0]) else 0)
                    if model in fit_model_names:
                        x_vals_fit.append(sz)
                        y_vals_fit.append(acc)

        if not x_vals:
            continue

        color = colors[hint_idx]
        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_errs,
            color=color,
            linestyle="none",
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=2,
            alpha=0.8,
            label=f"{hint}",
        )

        # Fit per-hint sigmoid
        min_points = 3 if fit_scaling else 2
        if fit_sigmoid_curves and len(x_vals_fit) >= min_points:
            try:
                fit_result = fit_sigmoid(
                    np.array(x_vals_fit),
                    np.array(y_vals_fit),
                    use_log=True,
                    scale=fit_scaling,
                )
                hint_fits[hint] = fit_result

                x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                y_smooth = fit_result["predict"](x_smooth)
                ax.plot(x_smooth, y_smooth, color=color, linestyle="-", linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to fit sigmoid for hint={hint}: {e}")

        # Plot joint model prediction
        if joint_fit is not None:
            try:
                x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                y_predicted = joint_fit["predict"](x_smooth, hint)
                ax.plot(x_smooth, y_predicted, color=color, linestyle="--", linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to plot joint prediction for hint={hint}: {e}")

    # Create legend
    use_transform = hint_transform(0.5) != 0.5

    if fit_sigmoid_curves and hint_fits:
        hint_handles = []
        hint_labels = []

        for hint_idx, hint in enumerate(hints):
            color = colors[hint_idx]
            base_label = f"{hint} (H'={hint_transform(hint):.2f})" if use_transform else f"{hint}"

            if hint in hint_fits:
                equation = format_equation(hint_fits[hint])
                label_text = f"{base_label}: {equation}"
            else:
                label_text = base_label

            handle = Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=8)
            hint_handles.append(handle)
            hint_labels.append(label_text)

        if joint_fit is not None:
            joint_eq = format_equation(joint_fit)
            hint_handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=2))
            hint_labels.append(joint_eq)

        ax.legend(hint_handles, hint_labels, title="hint fraction", framealpha=0.9, loc="upper left")
    elif joint_fit is not None:
        hint_handles = []
        hint_labels = []

        for hint_idx, hint in enumerate(hints):
            color = colors[hint_idx]
            handle = Line2D([0], [0], color=color, linewidth=2, marker="o", markersize=8)
            hint_handles.append(handle)
            hint_labels.append(f"{hint} (H'={hint_transform(hint):.2f})" if use_transform else f"{hint}")

        joint_eq = format_equation(joint_fit)
        hint_handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=2))
        hint_labels.append(joint_eq)

        ax.legend(hint_handles, hint_labels, title="hint fraction", framealpha=0.9, loc="upper left")
    else:
        ax.legend(loc="lower right", title="hint fraction", framealpha=0.9)

    ax.set_xlabel("model size (B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("eval accuracy", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    return fig, ax


# Backwards compatibility aliases
plot_results = plot_accuracy_vs_hint
plot_results_rescaled = plot_error_vs_hint_transformed
plot_results_by_model_size = plot_accuracy_vs_size
