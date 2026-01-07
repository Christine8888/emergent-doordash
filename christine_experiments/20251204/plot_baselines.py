"""Plot baseline evaluation results across model sizes.

All plotting functions take DataFrames as input. Use src.modelx.load_baseline()
to load data.

Expected DataFrame columns:
- model: Model name
- model_size: Model size in billions
- accuracy: Accuracy value
- stderr: Standard error (optional)
- eval: Evaluation name (for multi-baseline plots)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
import seaborn as sns

from src.modelx import size, clean_name, fit_sigmoid, format_equation

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def plot_baseline(
    df: pd.DataFrame,
    title: str = "Accuracy vs Model Size",
    figsize: Tuple[int, int] = (10, 6),
    do_fit: bool = True,
    fit_scaling: bool = False,
    pin_lower_bound: float | None = None,
    upper_bound: float = 1.0,
    color: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot baseline results with accuracy vs log model size.

    Args:
        df: DataFrame with columns: model, model_size, accuracy, stderr (optional)
        title: Plot title
        figsize: Figure size
        do_fit: If True, fit a sigmoid curve
        fit_scaling: If True, fit y = h*σ(...), else y = σ(...)
        pin_lower_bound: If set, pin the lower asymptote to this value
        upper_bound: Upper bound for sigmoid (used with pin_lower_bound)
        color: Optional color override

    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    if color is None:
        color = sns.color_palette("husl", 1)[0]

    df = df.dropna(subset=["accuracy"]).sort_values("model_size")

    x_vals = df["model_size"].values
    y_vals = df["accuracy"].values
    y_errs = df["stderr"].fillna(0).values if "stderr" in df.columns else None
    labels = [clean_name(m) for m in df["model"].values]

    if len(x_vals) == 0:
        print("Warning: No data to plot")
        return fig, ax

    # Plot data points
    ax.errorbar(
        x_vals,
        y_vals,
        yerr=y_errs,
        color=color,
        linestyle="none",
        marker="o",
        markersize=10,
        capsize=5,
        linewidth=2,
        alpha=0.8,
    )

    # Add labels for each point
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            alpha=0.7,
        )

    # Fit and plot sigmoid
    fit_result = None
    if do_fit and len(x_vals) >= 2:
        try:
            fit_result = fit_sigmoid(
                np.array(x_vals),
                np.array(y_vals),
                use_log=True,
                scale=fit_scaling,
                lower=pin_lower_bound,
                upper=upper_bound if pin_lower_bound is not None else None,
            )

            x_smooth = np.logspace(
                np.log10(min(x_vals) * 0.5), np.log10(max(x_vals) * 2), 100
            )
            y_smooth = fit_result["predict"](x_smooth)
            ax.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=2, alpha=0.6)
        except Exception as e:
            print(f"Warning: Failed to fit sigmoid: {e}")

    # Add equation to plot
    if fit_result is not None:
        equation = format_equation(fit_result)
        ax.text(
            0.05,
            0.95,
            equation,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Model Size (B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    return fig, ax


def plot_multi_baseline(
    df: pd.DataFrame,
    title: str = "Baseline Evaluations vs Model Size",
    figsize: Tuple[int, int] = (12, 7),
    do_fit: bool = True,
    fit_scaling: bool = False,
    pin_lower_bound: float | None = None,
    upper_bound: float = 1.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple baseline evaluations on the same axes.

    Args:
        df: DataFrame with columns: model, model_size, accuracy, stderr (optional), eval
        title: Plot title
        figsize: Figure size
        do_fit: If True, fit sigmoid curves
        fit_scaling: If True, fit y = h*σ(...)
        pin_lower_bound: If set, pin lower asymptote
        upper_bound: Upper bound for sigmoid

    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    evals = df["eval"].unique()
    colors = sns.color_palette("husl", len(evals))

    for i, eval_name in enumerate(evals):
        eval_df = df[df["eval"] == eval_name].dropna(subset=["accuracy"]).sort_values("model_size")

        x_vals = eval_df["model_size"].values
        y_vals = eval_df["accuracy"].values
        y_errs = eval_df["stderr"].fillna(0).values if "stderr" in eval_df.columns else None

        if len(x_vals) == 0:
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
            label=eval_name,
        )

        # Fit sigmoid if requested
        if do_fit and len(x_vals) >= 2:
            try:
                fit_result = fit_sigmoid(
                    np.array(x_vals),
                    np.array(y_vals),
                    use_log=True,
                    scale=fit_scaling,
                    lower=pin_lower_bound,
                    upper=upper_bound if pin_lower_bound is not None else None,
                )

                x_smooth = np.logspace(
                    np.log10(min(x_vals) * 0.5), np.log10(max(x_vals) * 2), 100
                )
                y_smooth = fit_result["predict"](x_smooth)
                ax.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to fit sigmoid for {eval_name}: {e}")

    ax.legend(loc="lower right", title="Evaluation", framealpha=0.9)
    ax.set_xlabel("Model Size (B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    return fig, ax


# Backwards compatibility aliases
plot_baselines = plot_baseline
plot_multiple_baselines = plot_multi_baseline
