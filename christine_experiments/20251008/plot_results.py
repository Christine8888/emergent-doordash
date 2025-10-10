#!/usr/bin/env python3
"""Plot evaluation results from experiment runs."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


# ============= CONFIGURATION =============

# Base directory containing results
BASE_FOLDER = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251007/math"
FILENAME_TEMPLATE = "math_{condition}_{hint}.json"
GRADER_FIELD = "expression_exact_match_sympy"
# BASE_FOLDER = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251007/gpqa"
# FILENAME_TEMPLATE = "gpqa_diamond_{condition}_{hint}.json"
# GRADER_FIELD = "choice"

# Models to plot
MODELS = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    # "Qwen2.5-32B-Instruct",
]

CONDITIONS = ["5shot"]  # , "5shot"]
MAIN_CONDITION = "5shot"
ACCURACY_FIELD = "accuracy"
STDERR_FIELD = "stderr"
HINT_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8]

# Output configuration
OUTPUT_FILE = "results_plot.png"
DPI = 300

# =========================================


def clean_model_name(model: str) -> str:
    """Clean up model names for display.

    Examples:
        Qwen2.5-0.5B-Instruct -> 0.5B
        Qwen2.5-7B-Instruct -> 7B
    """
    # Extract the size (e.g., "0.5B", "7B", "32B")
    parts = model.split("-")
    for part in parts:
        if "B" in part:
            return part
    return model


def load_result(base_folder: str, condition: str, model: str, hint: float,
                filename_template: str) -> Optional[Dict]:
    """Load a single result JSON file.

    Args:
        base_folder: Base directory containing results
        condition: Condition name (e.g., '0shot', '5shot')
        model: Model name
        hint: Hint fraction
        filename_template: Template for filename

    Returns:
        Dictionary with result data, or None if file not found
    """
    filename = filename_template.format(condition=condition, hint=hint)
    filepath = Path(base_folder) / condition / model / filename

    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def extract_accuracy_and_stderr(result: Dict, grader_field: str = 'choice',
                                accuracy_field: str = 'accuracy',
                                stderr_field: str = 'stderr') -> Tuple[Optional[float], Optional[float]]:
    """Extract accuracy and stderr from result dictionary.

    Args:
        result: Result dictionary
        grader_field: Field name for grader results
        accuracy_field: Field name for accuracy value
        stderr_field: Field name for stderr value

    Returns:
        Tuple of (accuracy, stderr), or (None, None) if not found
    """
    if result is None:
        return None, None

    if grader_field in result:
        accuracy = result[grader_field].get(accuracy_field)
        stderr = result[grader_field].get(stderr_field)
        return accuracy, stderr

    return None, None


def load_all_results(base_folder: str, models: List[str], conditions: List[str],
                    hints: List[float], filename_template: str, grader_field: str = 'choice',
                    accuracy_field: str = 'accuracy', stderr_field: str = 'stderr') -> Dict:
    """Load all results into a nested dictionary.

    Args:
        base_folder: Base directory containing results
        models: List of model names
        conditions: List of conditions
        hints: List of hint fractions
        filename_template: Template for filename
        grader_field: Field name for grader results
        accuracy_field: Field name for accuracy value
        stderr_field: Field name for stderr value

    Returns:
        Nested dictionary: {model: {condition: {hint: (accuracy, stderr)}}}
    """
    results = {}

    for model in models:
        results[model] = {}
        for condition in conditions:
            results[model][condition] = {}
            for hint in hints:
                result = load_result(base_folder, condition, model, hint, filename_template)
                accuracy, stderr = extract_accuracy_and_stderr(result, grader_field, accuracy_field, stderr_field)
                results[model][condition][hint] = (accuracy, stderr)

    return results


def get_marker_style(condition: str, conditions: List[str]) -> str:
    """Get marker style for a condition.

    Args:
        condition: Condition name
        conditions: List of all conditions

    Returns:
        Matplotlib marker style string
    """
    # Markers: circle, triangle up, triangle down, square, diamond, star, etc.
    marker_styles = ['o', '^', 'v', 's', 'D', '*', 'P', 'X']
    idx = conditions.index(condition) % len(marker_styles)
    return marker_styles[idx]


def plot_results(results: Dict, models: List[str], conditions: List[str],
                hints: List[float], main_condition: str,
                title: str = "Accuracy vs Hint Fraction",
                figsize: Tuple[int, int] = (12, 7)):
    """Plot results with models in different colors and conditions as marker styles.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        conditions: List of conditions
        hints: List of hint fractions
        main_condition: Main condition to show in legend
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get color palette
    colors = sns.color_palette("husl", len(models))

    # Plot each model
    for model_idx, model in enumerate(models):
        color = colors[model_idx]
        clean_name = clean_model_name(model)

        for condition in conditions:
            # Extract data for this model-condition pair
            x_vals = []
            y_vals = []
            y_errs = []

            for hint in hints:
                accuracy, stderr = results[model][condition][hint]
                if accuracy is not None:
                    x_vals.append(hint)
                    y_vals.append(accuracy)
                    y_errs.append(stderr if stderr is not None else 0)

            if not x_vals:
                continue

            # Get marker style
            marker_style = get_marker_style(condition, conditions)

            # Only add label for main condition
            label = clean_name if condition == main_condition else None

            # Plot with error bars (no connecting lines)
            ax.errorbar(x_vals, y_vals, yerr=y_errs,
                       color=color, linestyle='none',
                       marker=marker_style, markersize=8, capsize=4,
                       linewidth=2, alpha=0.8, label=label)

    # Create custom legend for marker styles
    # Model legend (colors)
    model_legend = ax.legend(loc='upper left', title='Models', framealpha=0.9)
    ax.add_artist(model_legend)

    # Marker style legend (conditions)
    style_handles = []
    for condition in conditions:
        marker_style = get_marker_style(condition, conditions)
        handle = Line2D([0], [0], color='black', linestyle='none',
                       marker=marker_style, markersize=8, label=condition)
        style_handles.append(handle)

    style_legend = ax.legend(handles=style_handles, loc='lower right',
                            title='eval conditions', framealpha=0.9)

    # Formatting
    ax.set_xlabel('fraction of reasoning chain as hint', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(hints) + 0.05)

    plt.tight_layout()
    return fig, ax


def extract_model_size(model: str) -> float:
    """Extract model size as a float from model name.

    Examples:
        Qwen2.5-0.5B-Instruct -> 0.5
        Qwen2.5-7B-Instruct -> 7.0
        Qwen2.5-32B-Instruct -> 32.0
    """
    parts = model.split("-")
    for part in parts:
        if "B" in part:
            # Remove 'B' and convert to float
            return float(part.replace("B", ""))
    return 0.0


def get_hint_color(hint: float, hints: List[float]) -> Tuple[float, float, float]:
    """Get color for a hint fraction using a colormap.

    Args:
        hint: Hint fraction
        hints: List of all hint fractions

    Returns:
        RGB color tuple
    """
    # Use a colormap (e.g., viridis, plasma, coolwarm)
    cmap = plt.cm.viridis
    # Normalize hint to [0, 1] range
    if len(hints) > 1:
        norm_hint = (hint - min(hints)) / (max(hints) - min(hints))
    else:
        norm_hint = 0.5
    return cmap(norm_hint)


def plot_results_by_model_size(results: Dict, models: List[str], conditions: List[str],
                                hints: List[float], main_hint: float,
                                title: str = "Accuracy vs Model Size",
                                figsize: Tuple[int, int] = (12, 7)):
    """Plot results with model size on x-axis, hints as colors, conditions as markers.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        conditions: List of conditions
        hints: List of hint fractions
        main_hint: Main hint fraction to show in legend
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract model sizes and sort
    model_sizes = [(model, extract_model_size(model)) for model in models]
    model_sizes.sort(key=lambda x: x[1])

    # Plot for each condition
    for condition in conditions:
        marker_style = get_marker_style(condition, conditions)

        for hint in hints:
            # Extract data for this condition-hint pair across models
            x_vals = []
            y_vals = []
            y_errs = []

            for model, size in model_sizes:
                accuracy, stderr = results[model][condition][hint]
                if accuracy is not None:
                    x_vals.append(size)
                    y_vals.append(accuracy)
                    y_errs.append(stderr if stderr is not None else 0)

            if not x_vals:
                continue

            # Get color for this hint
            color = get_hint_color(hint, hints)

            # Only add label for main hint
            if hint == main_hint:
                label = f"{condition} (hint={hint})"
            else:
                label = None

            # Plot with error bars (no connecting lines)
            ax.errorbar(x_vals, y_vals, yerr=y_errs,
                       color=color, linestyle='none',
                       marker=marker_style, markersize=8, capsize=4,
                       linewidth=2, alpha=0.8, label=label)

    # Create custom legends
    # Condition legend (marker styles)
    condition_handles = []
    for condition in conditions:
        marker_style = get_marker_style(condition, conditions)
        handle = Line2D([0], [0], color='gray', linestyle='none',
                       marker=marker_style, markersize=8, label=condition)
        condition_handles.append(handle)

    condition_legend = ax.legend(handles=condition_handles, loc='upper left',
                                 title='eval conditions', framealpha=0.9)
    ax.add_artist(condition_legend)

    # Hint fraction legend (colors)
    hint_handles = []
    for hint in hints:
        color = get_hint_color(hint, hints)
        handle = mpatches.Patch(color=color, label=f"{hint}")
        hint_handles.append(handle)

    hint_legend = ax.legend(handles=hint_handles, loc='lower right',
                           title='fraction of reasoning chain as hint', framealpha=0.9)

    # Formatting
    ax.set_xlabel('model size (B)', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')  # Log scale for model size

    plt.tight_layout()
    return fig, ax


def main():
    """Main function to load data and generate plots."""
    print("Loading results...")
    results = load_all_results(
        base_folder=BASE_FOLDER,
        models=MODELS,
        conditions=CONDITIONS,
        hints=HINT_FRACTIONS,
        filename_template=FILENAME_TEMPLATE,
        grader_field=GRADER_FIELD,
        accuracy_field=ACCURACY_FIELD,
        stderr_field=STDERR_FIELD
    )
    print("Done!")

    # Generate plot by model size
    # X-axis: Model size (in billions)
    # Y-axis: Accuracy
    # Colors: Different hint fractions
    # Markers: Different conditions
    print(f"Generating plot...")
    fig, ax = plot_results_by_model_size(
        results=results,
        models=MODELS,
        conditions=CONDITIONS,
        hints=HINT_FRACTIONS,
        main_hint=0.8,  # Only label this hint fraction in the legend
        title="MATH: accuracy vs model size"
    )

    # Save figure
    output_path = Path(BASE_FOLDER).parent / OUTPUT_FILE
    print(f"Saving plot to {output_path}...")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved successfully!")

    plt.close(fig)


if __name__ == "__main__":
    main()
