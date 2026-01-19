"""Plot baseline evaluation results across model sizes or ECI.

All plotting functions take DataFrames as input. Use src.modelx.load_baseline()
to load data.

Expected DataFrame columns:
- model: Model name
- model_size: Model size in billions
- accuracy: Accuracy value
- stderr: Standard error (optional)
- eval: Evaluation name (for multi-baseline plots)
"""

import sys
from pathlib import Path
# Add project root to path for src.modelx imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Literal
import seaborn as sns

from src.modelx import size, clean_name, model_eci, fit_sigmoid, format_equation

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def plot_baseline(
    df: pd.DataFrame,
    title: str = "Accuracy vs Model Size",
    figsize: Tuple[int, int] = (10, 6),
    x_axis: Literal["model_size", "eci"] = "model_size",
    do_fit: bool = True,
    fit_scaling: bool = False,
    pin_lower_bound: float | None = None,
    upper_bound: float = 1.0,
    color: str | None = None,
    xscale: str = "log",
    full_names: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot baseline results with accuracy vs model size or ECI.

    Args:
        df: DataFrame with columns: model, model_size, accuracy, stderr (optional)
        title: Plot title
        figsize: Figure size
        x_axis: What to plot on x-axis ("model_size" or "eci")
        do_fit: If True, fit a sigmoid curve
        fit_scaling: If True, fit y = h*σ(...), else y = σ(...)
        pin_lower_bound: If set, pin the lower asymptote to this value
        upper_bound: Upper bound for sigmoid (used with pin_lower_bound)
        color: Optional color override
        xscale: Scale for x-axis ('log' or 'linear')
        full_names: If True, use full model names for labels instead of just size

    Returns:
        Figure and Axes objects
    """
    import logging
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots(figsize=figsize)

    if color is None:
        color = sns.color_palette("husl", 1)[0]

    # Get x-axis function and label
    if x_axis == "model_size":
        get_x_value = size
        x_label = "Model Size (B)"
    elif x_axis == "eci":
        get_x_value = model_eci
        x_label = "ECI (Epoch Capabilities Index)"
    else:
        raise ValueError(f"Unknown x_axis: {x_axis}. Use 'model_size' or 'eci'")

    df = df.dropna(subset=["accuracy"]).copy()

    # Compute x values
    x_vals_list = []
    y_vals_list = []
    y_errs_list = []
    labels_list = []

    for _, row in df.iterrows():
        x_val = get_x_value(row["model"])
        if x_val is None:
            logger.warning(f"Could not get {x_axis} for model: {row['model']}")
            continue
        x_vals_list.append(x_val)
        y_vals_list.append(row["accuracy"])
        y_errs_list.append(row.get("stderr", 0) or 0)
        labels_list.append(row["model"] if full_names else clean_name(row["model"]))

    if len(x_vals_list) == 0:
        print("Warning: No data to plot")
        return fig, ax

    # Sort by x value
    sorted_data = sorted(zip(x_vals_list, y_vals_list, y_errs_list, labels_list))
    x_vals = np.array([d[0] for d in sorted_data])
    y_vals = np.array([d[1] for d in sorted_data])
    y_errs = np.array([d[2] for d in sorted_data])
    labels = [d[3] for d in sorted_data]

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
                x_vals,
                y_vals,
                use_log=(xscale == "log"),
                scale=fit_scaling,
                lower=pin_lower_bound,
                upper=upper_bound if pin_lower_bound is not None else None,
            )

            if xscale == "log":
                x_smooth = np.logspace(
                    np.log10(min(x_vals) * 0.9), np.log10(max(x_vals) * 1.1), 100
                )
            else:
                x_range = max(x_vals) - min(x_vals)
                x_smooth = np.linspace(min(x_vals) - 0.1 * x_range, max(x_vals) + 0.1 * x_range, 100)
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

    ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale(xscale)

    plt.tight_layout()
    return fig, ax


def plot_multi_baseline(
    df: pd.DataFrame,
    title: str = "Baseline Evaluations vs Model Size",
    figsize: Tuple[int, int] = (12, 7),
    x_axis: Literal["model_size", "eci"] = "model_size",
    do_fit: bool = True,
    fit_scaling: bool = False,
    pin_lower_bound: float | None = None,
    upper_bound: float = 1.0,
    xscale: str = "log",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple baseline evaluations on the same axes.

    Args:
        df: DataFrame with columns: model, model_size, accuracy, stderr (optional), eval
        title: Plot title
        figsize: Figure size
        x_axis: What to plot on x-axis ("model_size" or "eci")
        do_fit: If True, fit sigmoid curves
        fit_scaling: If True, fit y = h*σ(...)
        pin_lower_bound: If set, pin lower asymptote
        upper_bound: Upper bound for sigmoid
        xscale: Scale for x-axis ('log' or 'linear')

    Returns:
        Figure and Axes objects
    """
    import logging
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots(figsize=figsize)

    # Get x-axis function and label
    if x_axis == "model_size":
        get_x_value = size
        x_label = "Model Size (B)"
    elif x_axis == "eci":
        get_x_value = model_eci
        x_label = "ECI (Epoch Capabilities Index)"
    else:
        raise ValueError(f"Unknown x_axis: {x_axis}. Use 'model_size' or 'eci'")

    evals = df["eval"].unique()
    colors = sns.color_palette("husl", len(evals))

    for i, eval_name in enumerate(evals):
        eval_df = df[df["eval"] == eval_name].dropna(subset=["accuracy"])

        # Compute x values
        x_vals_list = []
        y_vals_list = []
        y_errs_list = []

        for _, row in eval_df.iterrows():
            x_val = get_x_value(row["model"])
            if x_val is None:
                logger.warning(f"Could not get {x_axis} for model: {row['model']}")
                continue
            x_vals_list.append(x_val)
            y_vals_list.append(row["accuracy"])
            y_errs_list.append(row.get("stderr", 0) or 0)

        if len(x_vals_list) == 0:
            continue

        # Sort by x value
        sorted_data = sorted(zip(x_vals_list, y_vals_list, y_errs_list))
        x_vals = np.array([d[0] for d in sorted_data])
        y_vals = np.array([d[1] for d in sorted_data])
        y_errs = np.array([d[2] for d in sorted_data])

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
                    x_vals,
                    y_vals,
                    use_log=(xscale == "log"),
                    scale=fit_scaling,
                    lower=pin_lower_bound,
                    upper=upper_bound if pin_lower_bound is not None else None,
                )

                if xscale == "log":
                    x_smooth = np.logspace(
                        np.log10(min(x_vals) * 0.9), np.log10(max(x_vals) * 1.1), 100
                    )
                else:
                    x_range = max(x_vals) - min(x_vals)
                    x_smooth = np.linspace(min(x_vals) - 0.1 * x_range, max(x_vals) + 0.1 * x_range, 100)
                y_smooth = fit_result["predict"](x_smooth)
                ax.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to fit sigmoid for {eval_name}: {e}")

    ax.legend(loc="lower right", title="Evaluation", framealpha=0.9)
    ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale(xscale)

    plt.tight_layout()
    return fig, ax


# Backwards compatibility aliases
plot_baselines = plot_baseline
plot_multiple_baselines = plot_multi_baseline


# ============================================================================
# Example: GPQA accuracy vs ECI
# ============================================================================
if __name__ == "__main__":
    from src.modelx import load_baseline

    # Load GPQA baseline results
    baseline_folder = "/Users/christineye/emergent-doordash/christine_experiments/20251113/baseline"
    df = load_baseline(baseline_folder, "aime")

    # Plot vs ECI
    fig, ax = plot_baseline(
        df,
        title="AIME Accuracy vs Size",
        x_axis="eci",
        xscale="log",
        full_names=True,
        pin_lower_bound = 0.0
    )
    plt.show()
