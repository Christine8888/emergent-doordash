# %%

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


# %%

def clean_model_name(model: str) -> str:
    """Clean up model names for display.

    Examples:
        Qwen2.5-0.5B-Instruct -> 0.5B
        Qwen2.5-7B-Instruct -> 7B
    """
    # Extract the size (e.g., "0.5B", "7B", "32B")
    parts = model.upper().split("-")
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


def extract_accuracy_and_stderr(result: Dict, grader_field: str = 'manual_bootstrap', accuracy_field: str = 'accuracy', stderr_field: str = 'stderr') -> Tuple[Optional[float], Optional[float]]:
    """Extract accuracy and stderr from result dictionary.

    Args:
        result: Result dictionary

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
                    hints: List[float], filename_template: str, grader_field: str = 'manual_bootstrap', accuracy_field: str = 'accuracy', stderr_field: str = 'stderr') -> Dict:
    """Load all results into a nested dictionary.

    Args:
        base_folder: Base directory containing results
        models: List of model names
        conditions: List of conditions
        hints: List of hint fractions
        filename_template: Template for filename

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

# %%


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
    from matplotlib.lines import Line2D

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

# %%
def plot_results_rescaled(results: Dict, models: List[str], conditions: List[str],
                hints: List[float], main_condition: str,
                title: str = "Accuracy vs Hint Fraction",
                figsize: Tuple[int, int] = (12, 7),
                fit_scaling: bool = False):
    """Plot results with models in different colors and conditions as marker styles.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        conditions: List of conditions
        hints: List of hint fractions
        main_condition: Main condition to show in legend
        title: Plot title
        figsize: Figure size
        fit_scaling: If True, fit y = h*sigmoid(m*log(x) + b), else y = sigmoid(m*log(x) + b)
    """
    from scipy.optimize import curve_fit
    
    def sigmoid(x, m, b):
        """Sigmoid function: 1 / (1 + exp(-(m*log(x) + b)))"""
        return 1 / (1 + np.exp(-(m * np.log(x) + b)))
    
    def scaled_sigmoid(x, h, m, b):
        """Scaled sigmoid function: h / (1 + exp(-(m*log(x) + b)))"""
        return h / (1 + np.exp(-(m * np.log(x) + b)))
    
    fig, ax = plt.subplots(figsize=figsize)

    # Get color palette
    colors = sns.color_palette("husl", len(models))

    # Store fitted parameters for legend
    model_params = {}

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
                    x_vals.append(1 / (1 - hint))
                    y_vals.append(1 - accuracy)
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
            
            # Fit sigmoid and plot curve (only for main condition to get params once)
            if len(x_vals) >= 2 and condition == main_condition:
                try:
                    x_arr = np.array(x_vals)
                    y_arr = np.array(y_vals)
                    
                    if fit_scaling:
                        # Fit the scaled sigmoid: h * sigmoid(m*log(x) + b)
                        # Initial guess: h = max(y), m = 1, b = 0
                        params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr, 
                                             p0=[np.max(y_arr), 1, 0],
                                             maxfev=10000)
                        h_fit, m_fit, b_fit = params
                        model_params[model] = (h_fit, m_fit, b_fit)
                        
                        # Generate smooth curve for plotting
                        x_smooth = np.logspace(np.log10(min(x_vals)), 
                                              np.log10(max(x_vals)), 100)
                        y_smooth = scaled_sigmoid(x_smooth, *params)
                    else:
                        # Fit the standard sigmoid
                        params, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[1, 0],
                                             maxfev=10000)
                        m_fit, b_fit = params
                        model_params[model] = (m_fit, b_fit)
                        
                        # Generate smooth curve for plotting
                        x_smooth = np.logspace(np.log10(min(x_vals)), 
                                              np.log10(max(x_vals)), 100)
                        y_smooth = sigmoid(x_smooth, *params)
                    
                    # Plot fitted curve (same color, dashed, no label)
                    ax.plot(x_smooth, y_smooth, color=color, 
                           linestyle='--', linewidth=2, alpha=0.6)
                except:
                    # If fitting fails, just skip the curve
                    pass

    # Create custom legend for marker styles
    from matplotlib.lines import Line2D

    # Model legend (colors) with equations
    model_handles = []
    model_labels = []
    for model_idx, model in enumerate(models):
        color = colors[model_idx]
        clean_name = clean_model_name(model)
        
        # Add equation if we have fitted parameters
        if model in model_params:
            if fit_scaling:
                h_fit, m_fit, b_fit = model_params[model]
                equation = f"{h_fit:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
            else:
                m_fit, b_fit = model_params[model]
                equation = f"σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
            label_text = f"{clean_name}: {equation}"
        else:
            label_text = clean_name
        
        handle = Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=8)
        model_handles.append(handle)
        model_labels.append(label_text)
    
    model_legend = ax.legend(model_handles, model_labels, title='Models', 
                            framealpha=0.9, loc='best')
    ax.add_artist(model_legend)

    # Marker style legend (conditions)
    style_handles = []
    for condition in conditions:
        marker_style = get_marker_style(condition, conditions)
        handle = Line2D([0], [0], color='black', linestyle='none',
                       marker=marker_style, markersize=8, label=condition)
        style_handles.append(handle)
    
    # Formatting
    ax.set_xlabel('1 / (1 - hint)', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval error rate', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    return fig, ax

# %%


def extract_model_size(model: str) -> float:
    """Extract model size as a float from model name.

    Examples:
        Qwen2.5-0.5B-Instruct -> 0.5
        Qwen2.5-7B-Instruct -> 7.0
        Qwen2.5-32B-Instruct -> 32.0
    """
    parts = model.upper().split("-")
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
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

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


# ## Load and Plot Results

# %%
# ============= CONFIGURATION =============

# Base directory containing results
# BASE_FOLDER = "/sphinx/u/cye/emergent-doordash/christine_experiments/20251007/math"
# FILENAME_TEMPLATE = "math_{condition}_{hint}.json"
# GRADER_FIELD = "expression_exact_match_sympy"
BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251015/results/gpqa"
FILENAME_TEMPLATE = "gpqa_diamond_{condition}_{hint}.json"
GRADER_FIELD = "manual_bootstrap"

# Models to plot
MODELS = [
    # "Qwen2.5-0.5B-Instruct",
    # "Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct",
    # "Qwen2.5-14B-Instruct",
    # "Qwen2.5-32B-Instruct",
    "gemma-3-0.27b-it",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    # "OLMo-2-0425-1B-Instruct",
    # "OLMo-2-1124-7B-Instruct",
    # "OLMo-2-0325-32B-Instruct",
    # "OLMo-2-1124-13B-Instruct",
]

CONDITIONS = ["0shot"]
MAIN_CONDITION = "0shot"
ACCURACY_FIELD = "accuracy"
STDERR_FIELD = "stderr"
HINT_FRACTIONS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# =========================================


# %%


# Load all results
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


# %%


# Plot results by model size
# X-axis: Model size (in billions)
# Y-axis: Accuracy
# Colors: Different hint fractions
# Markers: Different conditions
fig2, ax2 = plot_results_by_model_size(
    results=results,
    models=MODELS,
    conditions=CONDITIONS,
    hints=HINT_FRACTIONS,
    main_hint=0.8,  # Only label this hint fraction in the legend
    title="GPQA: accuracy vs model size"
)

plt.show()



# %%


# Plot results
fig, ax = plot_results_rescaled(
    results=results,
    models=MODELS,
    conditions=CONDITIONS,
    hints=HINT_FRACTIONS,
    main_condition=MAIN_CONDITION,
    title="GPQA Diamond: accuracy vs hint fraction",
    fit_scaling=False
)

plt.show()

# %%
BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251015/results/aime"
FILENAME_TEMPLATE = "aime_{condition}_{hint}.json"
GRADER_FIELD = "manual_bootstrap"

# Models to plot
MODELS = [
    # "Qwen2.5-0.5B-Instruct",
    # "Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct",
    # "Qwen2.5-14B-Instruct",
    # "Qwen2.5-32B-Instruct",
]

CONDITIONS = ["0shot"]
MAIN_CONDITION = "0shot"
ACCURACY_FIELD = "accuracy"
STDERR_FIELD = "stderr"
HINT_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8]

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

fig2, ax2 = plot_results_by_model_size(
    results=results,
    models=MODELS,
    conditions=CONDITIONS,
    hints=HINT_FRACTIONS,
    main_hint=0.8,  # Only label this hint fraction in the legend
    title="AIME: accuracy vs model size"
)

plt.show()
# %%

plot_results(
    results=results,
    models=MODELS,
    conditions=CONDITIONS,
    hints=HINT_FRACTIONS,
    main_condition=MAIN_CONDITION,
    title="AIME: accuracy vs hint fraction"
)
# %%

fig, ax = plot_results_rescaled(
    results=results,
    models=MODELS,
    conditions=CONDITIONS,
    hints=HINT_FRACTIONS,
    main_condition=MAIN_CONDITION,
    title="AIME: accuracy vs hint fraction",
    fit_scaling=True
)

plt.show()
# %%

def is_id_json(filename):
    """Check if filename is a 22-character ID JSON file."""
    import re
    pattern = r'^[A-Za-z0-9]{22}\.json$'
    return bool(re.match(pattern, filename))

def load_pass_at_k_by_hint(base_folder: str, model: str, condition: str = "0shot"):
    """Load pass@k data grouped by hint fraction.

    Args:
        base_folder: Base directory containing results (e.g., "results/gpqa")
        model: Model name (e.g., "gemma-3-4b-it")
        condition: Condition name (e.g., "0shot")

    Returns:
        Dictionary: {hint_fraction: {k: (accuracy, stderr)}}
    """
    model_folder = Path(base_folder) / condition / model

    if not model_folder.exists():
        print(f"Warning: Folder not found: {model_folder}")
        return {}

    results_by_hint = {}

    for json_file in model_folder.glob("*.json"):
        if not is_id_json(json_file.name):
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        if "hint_fraction" not in data or "pass_at_k" not in data:
            continue

        hint = data["hint_fraction"]
        if hint not in results_by_hint:
            results_by_hint[hint] = {}

        for k_str, metrics in data["pass_at_k"].items():
            k = int(k_str)
            accuracy = metrics.get("accuracy")
            stderr = metrics.get("stderr")

            results_by_hint[hint][k] = (accuracy, stderr)

    return results_by_hint

def plot_pass_at_k_by_hint(results_by_hint: Dict, hints: List[float],
                            model_name: str,
                            title: str = "Pass@k vs Hint Fraction",
                            figsize: Tuple[int, int] = (12, 7)):
    """Plot pass@k accuracy for different hint fractions.

    Args:
        results_by_hint: Results dictionary from load_pass_at_k_by_hint
        hints: List of hint fractions to plot
        model_name: Model name for title
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("viridis", len(hints))

    for i, hint in enumerate(hints):
        if hint not in results_by_hint:
            print(f"Warning: No data for hint={hint}")
            continue

        k_values = []
        accuracies = []
        stderrs = []

        for k in sorted(results_by_hint[hint].keys()):
            accuracy, stderr = results_by_hint[hint][k]
            if accuracy is not None:
                k_values.append(k)
                accuracies.append(accuracy)
                stderrs.append(stderr if stderr is not None else 0)

        if not k_values:
            continue

        color = colors[i]
        label = f"{hint}"

        ax.errorbar(k_values, accuracies, yerr=stderrs,
                   color=color, linestyle='-',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=label)

    ax.set_xlabel('k (number of attempts)', fontsize=13, fontweight='bold')
    ax.set_ylabel('pass@k accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', title='hint fraction', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
# %%

# Example usage: Plot pass@k by hint fraction for a single model
# Uncomment and customize the following code:

MODEL_TO_PLOT = "Qwen2.5-32B-Instruct"
HINTS_TO_PLOT = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]  # Specify which hint fractions to show

results_by_hint = load_pass_at_k_by_hint(
    base_folder=BASE_FOLDER,
    model=MODEL_TO_PLOT,
    condition="0shot"
)

fig, ax = plot_pass_at_k_by_hint(
    results_by_hint=results_by_hint,
    hints=HINTS_TO_PLOT,
    model_name=clean_model_name(MODEL_TO_PLOT),
    title=f"GPQA: Pass@k for {clean_model_name(MODEL_TO_PLOT)}"
)

plt.show()
# %%
