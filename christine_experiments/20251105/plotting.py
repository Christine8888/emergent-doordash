"""Plotting utilities for hint fraction experiments."""

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


def clean_model_name(model: str) -> str:
    """Clean up model names for display.

    Examples:
        Qwen2.5-0.5B-Instruct -> 0.5B
        Qwen2.5-7B-Instruct -> 7B
    """
    parts = model.upper().split("-")
    for part in parts:
        if "B" in part:
            return part
    return model


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
            return float(part.replace("B", ""))
    return 0.0


def load_result(base_folder: str, model: str, hint: float,
                filename_template: str, condition: str = "0shot",
                solver: Optional[str] = None) -> Optional[Dict]:
    """Load a single result JSON file.

    Args:
        base_folder: Base directory containing results
        model: Model name
        hint: Hint fraction
        filename_template: Template for filename (use {hint} placeholder)
        condition: Condition name (e.g., '0shot')
        solver: Optional solver name for new folder structure (e.g., 'solution')

    Returns:
        Dictionary with result data, or None if file not found
    """
    filename = filename_template.format(hint=hint)

    if solver is not None:
        filepath = Path(base_folder) / solver / condition / model / filename
    else:
        filepath = Path(base_folder) / condition / model / filename

    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def extract_accuracy_and_stderr(result: Dict, grader_field: str = 'manual_bootstrap',
                                accuracy_field: str = 'accuracy',
                                stderr_field: str = 'stderr') -> Tuple[Optional[float], Optional[float]]:
    """Extract accuracy and stderr from result dictionary.

    Args:
        result: Result dictionary
        grader_field: Field containing grader results
        accuracy_field: Field name for accuracy
        stderr_field: Field name for stderr

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


def load_all_results(base_folder: str, models: List[str], hints: List[float],
                    filename_template: str, condition: str = "0shot",
                    solver: Optional[str] = None,
                    grader_field: str = 'manual_bootstrap',
                    accuracy_field: str = 'accuracy',
                    stderr_field: str = 'stderr') -> Dict:
    """Load all results into a nested dictionary.

    Args:
        base_folder: Base directory containing results
        models: List of model names
        hints: List of hint fractions
        filename_template: Template for filename
        condition: Condition name (e.g., '0shot')
        solver: Optional solver name for new folder structure
        grader_field: Field containing grader results
        accuracy_field: Field name for accuracy
        stderr_field: Field name for stderr

    Returns:
        Nested dictionary: {model: {hint: (accuracy, stderr)}}
    """
    results = {}

    for model in models:
        results[model] = {}
        for hint in hints:
            result = load_result(base_folder, model, hint, filename_template,
                               condition=condition, solver=solver)
            accuracy, stderr = extract_accuracy_and_stderr(result, grader_field,
                                                          accuracy_field, stderr_field)
            results[model][hint] = (accuracy, stderr)

    return results


def plot_results(results: Dict, models: List[str], hints: List[float],
                title: str = "Accuracy vs Hint Fraction",
                figsize: Tuple[int, int] = (12, 7)):
    """Plot results with models in different colors.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        hints: List of hint fractions
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(models))

    for model_idx, model in enumerate(models):
        color = colors[model_idx]
        clean_name = clean_model_name(model)

        x_vals = []
        y_vals = []
        y_errs = []

        for hint in hints:
            accuracy, stderr = results[model][hint]
            if accuracy is not None:
                x_vals.append(hint)
                y_vals.append(accuracy)
                y_errs.append(stderr if stderr is not None else 0)

        if not x_vals:
            continue

        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=clean_name)

    ax.legend(loc='upper left', title='Models', framealpha=0.9)
    ax.set_xlabel('fraction of reasoning chain as hint', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(hints) + 0.05)

    plt.tight_layout()
    return fig, ax


def plot_results_rescaled(results: Dict, models: List[str], hints: List[float],
                         title: str = "Error Rate vs Inverse Hint",
                         figsize: Tuple[int, int] = (12, 7),
                         fit_scaling: bool = False):
    """Plot results with rescaled axes and sigmoid fitting.

    X-axis: 1/(1-hint) on log scale
    Y-axis: 1-accuracy (error rate)
    Fits sigmoid function: σ(m*log(x) + b) or h*σ(m*log(x) + b)

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        hints: List of hint fractions
        title: Plot title
        figsize: Figure size
        fit_scaling: If True, fit y = h*σ(m*log(x) + b), else y = σ(m*log(x) + b)
    """
    from scipy.optimize import curve_fit

    def sigmoid(x, m, b):
        return 1 / (1 + np.exp(-(m * np.log(x) + b)))

    def scaled_sigmoid(x, h, m, b):
        return h / (1 + np.exp(-(m * np.log(x) + b)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(models))
    model_params = {}

    for model_idx, model in enumerate(models):
        color = colors[model_idx]
        clean_name = clean_model_name(model)

        x_vals = []
        y_vals = []
        y_errs = []

        for hint in hints:
            accuracy, stderr = results[model][hint]
            if accuracy is not None:
                x_vals.append(1 / (1 - hint))
                y_vals.append(1 - accuracy)
                y_errs.append(stderr if stderr is not None else 0)

        if not x_vals:
            continue

        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=clean_name)

        # Fit sigmoid and plot curve
        if len(x_vals) >= 2:
            try:
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)

                if fit_scaling:
                    params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr,
                                         p0=[np.max(y_arr), 1, 0],
                                         maxfev=10000)
                    h_fit, m_fit, b_fit = params
                    model_params[model] = (h_fit, m_fit, b_fit)

                    x_smooth = np.logspace(np.log10(min(x_vals)),
                                          np.log10(max(x_vals)), 100)
                    y_smooth = scaled_sigmoid(x_smooth, *params)
                else:
                    params, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[1, 0],
                                         maxfev=10000)
                    m_fit, b_fit = params
                    model_params[model] = (m_fit, b_fit)

                    x_smooth = np.logspace(np.log10(min(x_vals)),
                                          np.log10(max(x_vals)), 100)
                    y_smooth = sigmoid(x_smooth, *params)

                ax.plot(x_smooth, y_smooth, color=color,
                       linestyle='--', linewidth=2, alpha=0.6)
            except:
                pass

    # Create legend with equations
    from matplotlib.lines import Line2D
    model_handles = []
    model_labels = []

    for model_idx, model in enumerate(models):
        color = colors[model_idx]
        clean_name = clean_model_name(model)

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

    ax.legend(model_handles, model_labels, title='Models', framealpha=0.9, loc='best')
    ax.set_xlabel('1 / (1 - hint)', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval error rate', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    return fig, ax


def plot_results_by_model_size(results: Dict, models: List[str], hints: List[float],
                               title: str = "Accuracy vs Model Size",
                               figsize: Tuple[int, int] = (12, 7)):
    """Plot results with model size on x-axis and hints as different colors.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        hints: List of hint fractions
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_sizes = [(model, extract_model_size(model)) for model in models]
    model_sizes.sort(key=lambda x: x[1])

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(hints) - 1)) if len(hints) > 1 else cmap(0.5)
              for i in range(len(hints))]

    for hint_idx, hint in enumerate(hints):
        x_vals = []
        y_vals = []
        y_errs = []

        for model, size in model_sizes:
            accuracy, stderr = results[model][hint]
            if accuracy is not None:
                x_vals.append(size)
                y_vals.append(accuracy)
                y_errs.append(stderr if stderr is not None else 0)

        if not x_vals:
            continue

        color = colors[hint_idx]
        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=f"{hint}")

    ax.legend(loc='lower right', title='hint fraction', framealpha=0.9)
    ax.set_xlabel('model size (B)', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    return fig, ax


def is_id_json(filename):
    """Check if filename is a 22-character ID JSON file."""
    import re
    pattern = r'^[A-Za-z0-9]{22}\.json$'
    return bool(re.match(pattern, filename))


def load_pass_at_k_by_hint(base_folder: str, model: str,
                           condition: str = "0shot",
                           solver: Optional[str] = None):
    """Load pass@k data grouped by hint fraction.

    Args:
        base_folder: Base directory containing results
        model: Model name
        condition: Condition name (e.g., "0shot")
        solver: Optional solver name for new folder structure

    Returns:
        Dictionary: {hint_fraction: {k: (accuracy, stderr)}}
    """
    if solver is not None:
        model_folder = Path(base_folder) / solver / condition / model
    else:
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
        ax.errorbar(k_values, accuracies, yerr=stderrs,
                   color=color, linestyle='-',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=f"{hint}")

    ax.set_xlabel('k (number of attempts)', fontsize=13, fontweight='bold')
    ax.set_ylabel('pass@k accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', title='hint fraction', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
