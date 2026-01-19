"""Plotting utilities for hint fraction experiments."""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def get_model_eci(model: str) -> Optional[float]:
    """Get ECI score for a model.

    Uses the model_eci function from src.modelx which loads fitted ECI
    from christine_experiments/20260107/ if available, otherwise falls
    back to Epoch's pre-computed values.
    """
    import sys
    sys.path.insert(0, '/Users/christineye/emergent-doordash/src')

    try:
        from modelx import model_eci
        return model_eci(model)
    except Exception as e:
        logger.warning(f"Could not get ECI for {model}: {e}")
        return None


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
                         fit_scaling: bool = False,
                         fit_type: str = 'sigmoid',
                         hint_transform: Optional[callable] = None,
                         xlabel: Optional[str] = None,
                         xscale: str = 'log'):
    """Plot results with rescaled axes and curve fitting.

    X-axis: hint_transform(hint) on log scale (default: 1/(1-hint))
    Y-axis: 1-accuracy (error rate)
    Fits sigmoid or exponential function.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        hints: List of hint fractions
        title: Plot title
        figsize: Figure size
        fit_scaling: If True, fit with scaling parameter h (e.g., y = h*σ(...))
        fit_type: Type of curve to fit. Options:
                  - 'sigmoid': σ(m*log(x) + b) or h*σ(m*log(x) + b)
                  - 'exponential': h*(1 - a^x) - exponential decay
        hint_transform: Optional function to transform hint values for x-axis.
                       Default is lambda h: 1 / (1 - h)
        xlabel: Optional x-axis label. Default is '1 / (1 - hint)'
        xscale: Scale for x-axis ('log' or 'linear'). Default is 'log'
    """
    from scipy.optimize import curve_fit

    # Default hint transform: 1 / (1 - hint)
    def default_hint_transform(h):
        return 1 / (1 - h)
    
    if hint_transform is None:
        hint_transform = default_hint_transform

    def sigmoid(x, m, b):
        return 1 / (1 + np.exp(-(m * np.log(x) + b)))

    def scaled_sigmoid(x, h, m, b):
        return h / (1 + np.exp(-(m * np.log(x) + b)))

    def exponential(x, h, a, k):
        """Exponential: h * a^(k*x)"""
        return h * np.power(a, k * x)

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
                x_vals.append(hint_transform(hint))
                y_vals.append(1 - accuracy)
                y_errs.append(stderr if stderr is not None else 0)

        if not x_vals:
            continue

        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=clean_name)

        # Fit curve and plot
        if len(x_vals) >= 2:
            try:
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)

                # Generate smooth x values based on scale
                if xscale == 'log':
                    x_smooth = np.logspace(np.log10(min(x_vals)),
                                          np.log10(max(x_vals)), 100)
                else:
                    x_smooth = np.linspace(min(x_vals), max(x_vals), 100)

                if fit_type == 'exponential':
                    # Exponential: h * a^(k*x)
                    params, _ = curve_fit(exponential, x_arr, y_arr,
                                         p0=[np.max(y_arr), 0.5, 1.0],
                                         maxfev=10000)
                    h_fit, a_fit, k_fit = params
                    model_params[model] = ('exponential', h_fit, a_fit, k_fit)
                    y_smooth = exponential(x_smooth, *params)
                elif fit_scaling:
                    params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr,
                                         p0=[np.max(y_arr), 1, 0],
                                         maxfev=10000)
                    h_fit, m_fit, b_fit = params
                    model_params[model] = ('scaled_sigmoid', h_fit, m_fit, b_fit)
                    y_smooth = scaled_sigmoid(x_smooth, *params)
                else:
                    params, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[1, 0],
                                         maxfev=10000)
                    m_fit, b_fit = params
                    model_params[model] = ('sigmoid', m_fit, b_fit)
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
            params = model_params[model]
            fit_kind = params[0]
            if fit_kind == 'exponential':
                h_fit, a_fit, k_fit = params[1], params[2], params[3]
                equation = f"{h_fit:.3f}·{a_fit:.3f}^({k_fit:.2f}x)"
            elif fit_kind == 'scaled_sigmoid':
                h_fit, m_fit, b_fit = params[1], params[2], params[3]
                equation = f"{h_fit:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
            else:  # sigmoid
                m_fit, b_fit = params[1], params[2]
                equation = f"σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
            label_text = f"{clean_name}: {equation}"
        else:
            label_text = clean_name

        handle = Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=8)
        model_handles.append(handle)
        model_labels.append(label_text)

    ax.legend(model_handles, model_labels, title='Models', framealpha=0.9, loc='best')
    ax.set_xlabel(xlabel if xlabel is not None else '1 / (1 - hint)', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval error rate', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale(xscale)

    plt.tight_layout()
    return fig, ax


def plot_results_by_model_size(results: Dict, models: List[str], hints: List[float],
                               title: str = "Accuracy vs Model Size",
                               figsize: Tuple[int, int] = (12, 7),
                               fit_sigmoid: bool = False,
                               fit_scaling: bool = False,
                               fit_models = None,
                               fit_joint: bool = False,
                               include_cross: bool = True,
                               exclude_hint = None,
                               transform_hint: bool = False):
    """Plot results with model size on x-axis and hints as different colors.

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        hints: List of hint fractions
        title: Plot title
        figsize: Figure size
        fit_sigmoid: If True, fit sigmoid curves to the data
        fit_scaling: If True, fit y = h*σ(m*log(x) + b), else y = σ(m*log(x) + b)
        fit_models: Models to use for fitting. Can be:
            - None: use all models (default)
            - int: use first N models (sorted by size)
            - List[str]: use specific model names
        fit_joint: If True, fit joint model
        include_cross: If True, fit σ(α*C + β*H + γ*C*H + δ), else σ(α*C + β*H + δ)
            where C is log(model_size) and H is hint variable
        exclude_hint: Hint fraction(s) to exclude from joint fit. Can be:
            - None: include all hints (default)
            - float: exclude single hint value (e.g., 0)
            - List[float]: exclude multiple hint values (e.g., [0, 0.1])
        transform_hint: If True, use H = 1/(1-hint) instead of H = hint
    """
    from scipy.optimize import curve_fit

    def hint_transform(hint_val):
        """Transform hint fraction if transform_hint is True."""
        if transform_hint:
            return np.log(1 / (1 - hint_val))
        else:
            return hint_val

    def sigmoid(x, m, b):
        return 1 / (1 + np.exp(-(m * np.log(x) + b)))

    def scaled_sigmoid(x, h, m, b):
        return h / (1 + np.exp(-(m * np.log(x) + b)))

    def joint_sigmoid_with_cross(CH, alpha, beta, gamma, delta):
        """Joint sigmoid model: σ(α*C + β*H + γ*C*H + δ)
        where C is log(model_size) and H is hint_fraction
        CH is a 2xN array where CH[0] = C and CH[1] = H
        """
        C, H = CH
        z = alpha * C + beta * H + gamma * C * H + delta
        return 1 / (1 + np.exp(-z))

    def joint_sigmoid_no_cross(CH, alpha, beta, delta):
        """Joint sigmoid model: σ(α*C + β*H + δ)
        where C is log(model_size) and H is hint_fraction
        CH is a 2xN array where CH[0] = C and CH[1] = H
        """
        C, H = CH
        z = alpha * C + beta * H + delta
        return 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=figsize)

    model_sizes = [(model, extract_model_size(model)) for model in models]
    model_sizes.sort(key=lambda x: x[1])

    # Determine which models to use for fitting
    if fit_models is None:
        fit_model_names = set(models)
    elif isinstance(fit_models, int):
        fit_model_names = set([model for model, _ in model_sizes[:fit_models]])
    else:
        fit_model_names = set(fit_models)

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(hints) - 1)) if len(hints) > 1 else cmap(0.5)
              for i in range(len(hints))]

    # Fit joint model if requested
    joint_params = None
    if fit_joint:
        C_all = []
        H_all = []
        y_all = []

        # Convert exclude_hint to a set for easy checking
        if exclude_hint is None:
            exclude_set = set()
        elif isinstance(exclude_hint, (list, tuple)):
            exclude_set = set(exclude_hint)
        else:
            exclude_set = {exclude_hint}

        for hint in hints:
            if hint in exclude_set:
                continue
            for model, size in model_sizes:
                accuracy, stderr = results[model][hint]
                if accuracy is not None and model in fit_model_names:
                    C_all.append(np.log(size))
                    H_all.append(hint_transform(hint))
                    y_all.append(accuracy)

        min_points = 4 if include_cross else 3
        if len(C_all) >= min_points:
            try:
                CH = np.array([C_all, H_all])
                y_arr = np.array(y_all)

                if include_cross:
                    joint_params, _ = curve_fit(joint_sigmoid_with_cross, CH, y_arr,
                                               p0=[1, 1, 0, 0],
                                               maxfev=10000)
                    alpha, beta, gamma, delta = joint_params
                    y_pred = joint_sigmoid_with_cross(CH, *joint_params)
                    print(f"Joint fit: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
                else:
                    joint_params, _ = curve_fit(joint_sigmoid_no_cross, CH, y_arr,
                                               p0=[1, 1, 0],
                                               maxfev=10000)
                    alpha, beta, delta = joint_params
                    y_pred = joint_sigmoid_no_cross(CH, *joint_params)
                    print(f"Joint fit: α={alpha:.3f}, β={beta:.3f}, δ={delta:.3f}")

                # Calculate average RMS error
                rms = np.sqrt(np.mean((y_arr - y_pred) ** 2))
                print(f"Average RMS error = {rms:.4f}")
            except Exception as e:
                print(f"Warning: Failed to fit joint model: {e}")
                joint_params = None

    hint_params = {}

    for hint_idx, hint in enumerate(hints):
        x_vals = []
        y_vals = []
        y_errs = []
        x_vals_fit = []
        y_vals_fit = []

        for model, size in model_sizes:
            accuracy, stderr = results[model][hint]
            if accuracy is not None:
                x_vals.append(size)
                y_vals.append(accuracy)
                y_errs.append(stderr if stderr is not None else 0)

                if model in fit_model_names:
                    x_vals_fit.append(size)
                    y_vals_fit.append(accuracy)

        if not x_vals:
            continue

        color = colors[hint_idx]
        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=f"{hint}")

        # Fit sigmoid and plot curve
        min_points_needed = 3 if fit_scaling else 2
        if fit_sigmoid and len(x_vals_fit) >= min_points_needed:
            try:
                x_arr = np.array(x_vals_fit)
                y_arr = np.array(y_vals_fit)

                if fit_scaling:
                    params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr,
                                         p0=[np.max(y_arr), 1, 0],
                                         bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]),
                                         maxfev=10000)
                    h_fit, m_fit, b_fit = params
                    hint_params[hint] = (h_fit, m_fit, b_fit)

                    x_smooth = np.logspace(np.log10(min(x_vals)),
                                          np.log10(max(x_vals)), 100)
                    y_smooth = scaled_sigmoid(x_smooth, *params)
                else:
                    params, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[1, 0],
                                         maxfev=10000)
                    m_fit, b_fit = params
                    hint_params[hint] = (m_fit, b_fit)

                    x_smooth = np.logspace(np.log10(min(x_vals)),
                                          np.log10(max(x_vals)), 100)
                    y_smooth = sigmoid(x_smooth, *params)

                ax.plot(x_smooth, y_smooth, color=color,
                       linestyle='-', linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to fit sigmoid for hint={hint}: {e}")

        # Plot predicted sigmoid from joint model
        if fit_joint and joint_params is not None:
            try:
                x_smooth = np.logspace(np.log10(min(x_vals)),
                                      np.log10(max(x_vals)), 100)
                C_smooth = np.log(x_smooth)
                H_smooth = np.full_like(C_smooth, hint_transform(hint))
                CH_smooth = np.array([C_smooth, H_smooth])

                if include_cross:
                    y_predicted = joint_sigmoid_with_cross(CH_smooth, *joint_params)
                else:
                    y_predicted = joint_sigmoid_no_cross(CH_smooth, *joint_params)

                ax.plot(x_smooth, y_predicted, color=color,
                       linestyle='--', linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to plot joint prediction for hint={hint}: {e}")

    # Create legend
    from matplotlib.lines import Line2D

    if fit_sigmoid and hint_params:
        hint_handles = []
        hint_labels = []

        for hint_idx, hint in enumerate(hints):
            color = colors[hint_idx]

            # Create base label with transformed value if applicable
            if transform_hint:
                base_label = f"{hint} (H'={hint_transform(hint):.2f})"
            else:
                base_label = f"{hint}"

            if hint in hint_params:
                if fit_scaling:
                    h_fit, m_fit, b_fit = hint_params[hint]
                    equation = f"{h_fit:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
                else:
                    m_fit, b_fit = hint_params[hint]
                    equation = f"σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
                label_text = f"{base_label}: {equation}"
            else:
                label_text = base_label

            handle = Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=8)
            hint_handles.append(handle)
            hint_labels.append(label_text)

        # Add note about dashed lines if using joint fit
        if fit_joint and joint_params is not None:
            H_label = "H'" if transform_hint else "H"
            if include_cross:
                alpha, beta, gamma, delta = joint_params
                joint_eq = f"σ({alpha:.2f}C {beta:+.2f}{H_label} {gamma:+.2f}C{H_label} {delta:+.2f})"
            else:
                alpha, beta, delta = joint_params
                joint_eq = f"σ({alpha:.2f}C {beta:+.2f}{H_label} {delta:+.2f})"
            if transform_hint:
                joint_eq += " (H'=1/(1-H)-1)"
            hint_handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2))
            hint_labels.append(f'{joint_eq}')

        ax.legend(hint_handles, hint_labels, title='hint fraction', framealpha=0.9, loc='upper left')
    elif fit_joint and joint_params is not None:
        # Show legend with joint fit equation even if fit_sigmoid is False
        hint_handles = []
        hint_labels = []

        for hint_idx, hint in enumerate(hints):
            color = colors[hint_idx]
            handle = Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=8)
            hint_handles.append(handle)
            if transform_hint:
                hint_labels.append(f"{hint} (H'={hint_transform(hint):.2f})")
            else:
                hint_labels.append(f"{hint}")

        H_label = "H'" if transform_hint else "H"
        if include_cross:
            alpha, beta, gamma, delta = joint_params
            joint_eq = f"σ({alpha:.2f}C {beta:+.2f}{H_label} {gamma:+.2f}C{H_label} {delta:+.2f})"
        else:
            alpha, beta, delta = joint_params
            joint_eq = f"σ({alpha:.2f}C {beta:+.2f}{H_label} {delta:+.2f})"
        hint_handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2))
        hint_labels.append(f'{joint_eq}')

        ax.legend(hint_handles, hint_labels, title='hint fraction', framealpha=0.9, loc='upper left')
    else:
        ax.legend(loc='lower right', title='hint fraction', framealpha=0.9)

    ax.set_xlabel('model size (B)', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    return fig, ax


def plot_by_x_axis(
    results: Dict,
    models: List[str],
    hints: List[float],
    x_axis: Literal["model_size", "eci"] = "model_size",
    title: str = "Accuracy vs X",
    figsize: Tuple[int, int] = (12, 7),
    fit_sigmoid: bool = False,
    fit_scaling: bool = False,
    fit_models=None,
    fit_joint: bool = False,
    include_cross: bool = True,
    exclude_hint=None,
    transform_hint: bool = False,
    xscale: str = "log",
):
    """Plot results with configurable x-axis (model size or ECI).

    Args:
        results: Results dictionary from load_all_results
        models: List of model names
        hints: List of hint fractions
        x_axis: What to plot on x-axis:
            - "model_size": Model size in billions (extracted from name)
            - "eci": Epoch Capabilities Index score
        title: Plot title
        figsize: Figure size
        fit_sigmoid: If True, fit sigmoid curves to the data
        fit_scaling: If True, fit y = h*σ(m*log(x) + b), else y = σ(m*log(x) + b)
        fit_models: Models to use for fitting. Can be:
            - None: use all models (default)
            - int: use first N models (sorted by x value)
            - List[str]: use specific model names
        fit_joint: If True, fit joint model
        include_cross: If True, fit σ(α*C + β*H + γ*C*H + δ), else σ(α*C + β*H + δ)
            where C is log(x_value) and H is hint variable
        exclude_hint: Hint fraction(s) to exclude from joint fit
        transform_hint: If True, use H = log(1/(1-hint)) instead of H = hint
        xscale: Scale for x-axis ('log' or 'linear'). Default is 'log'

    Returns:
        (fig, ax) tuple
    """
    from scipy.optimize import curve_fit

    def hint_transform(hint_val):
        if transform_hint:
            return np.log(1 / (1 - hint_val))
        else:
            return hint_val

    def sigmoid(x, m, b):
        return 1 / (1 + np.exp(-(m * np.log(x) + b)))

    def scaled_sigmoid(x, h, m, b):
        return h / (1 + np.exp(-(m * np.log(x) + b)))

    def joint_sigmoid_with_cross(CH, alpha, beta, gamma, delta):
        C, H = CH
        z = alpha * C + beta * H + gamma * C * H + delta
        return 1 / (1 + np.exp(-z))

    def joint_sigmoid_no_cross(CH, alpha, beta, delta):
        C, H = CH
        z = alpha * C + beta * H + delta
        return 1 / (1 + np.exp(-z))

    # Get x values based on x_axis type
    if x_axis == "model_size":
        get_x_value = extract_model_size
        x_label = "model size (B)"
    elif x_axis == "eci":
        get_x_value = get_model_eci
        x_label = "ECI (Epoch Capabilities Index)"
    else:
        raise ValueError(f"Unknown x_axis: {x_axis}. Use 'model_size' or 'eci'")

    # Build list of (model, x_value) pairs, filtering out None
    model_x_values = []
    for model in models:
        x_val = get_x_value(model)
        if x_val is not None:
            model_x_values.append((model, x_val))
        else:
            logger.warning(f"Could not get {x_axis} for model: {model}")

    model_x_values.sort(key=lambda x: x[1])

    if not model_x_values:
        raise ValueError(f"No valid {x_axis} values found for any model")

    # Determine which models to use for fitting
    if fit_models is None:
        fit_model_names = set(m for m, _ in model_x_values)
    elif isinstance(fit_models, int):
        fit_model_names = set([model for model, _ in model_x_values[:fit_models]])
    else:
        fit_model_names = set(fit_models)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(hints) - 1)) if len(hints) > 1 else cmap(0.5)
              for i in range(len(hints))]

    # Fit joint model if requested
    joint_params = None
    if fit_joint:
        C_all = []
        H_all = []
        y_all = []

        if exclude_hint is None:
            exclude_set = set()
        elif isinstance(exclude_hint, (list, tuple)):
            exclude_set = set(exclude_hint)
        else:
            exclude_set = {exclude_hint}

        for hint in hints:
            if hint in exclude_set:
                continue
            for model, x_val in model_x_values:
                accuracy, stderr = results[model][hint]
                if accuracy is not None and model in fit_model_names:
                    C_all.append(np.log(x_val))
                    H_all.append(hint_transform(hint))
                    y_all.append(accuracy)

        min_points = 4 if include_cross else 3
        if len(C_all) >= min_points:
            try:
                CH = np.array([C_all, H_all])
                y_arr = np.array(y_all)

                if include_cross:
                    joint_params, _ = curve_fit(joint_sigmoid_with_cross, CH, y_arr,
                                               p0=[1, 1, 0, 0], maxfev=10000)
                    alpha, beta, gamma, delta = joint_params
                    y_pred = joint_sigmoid_with_cross(CH, *joint_params)
                    print(f"Joint fit: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
                else:
                    joint_params, _ = curve_fit(joint_sigmoid_no_cross, CH, y_arr,
                                               p0=[1, 1, 0], maxfev=10000)
                    alpha, beta, delta = joint_params
                    y_pred = joint_sigmoid_no_cross(CH, *joint_params)
                    print(f"Joint fit: α={alpha:.3f}, β={beta:.3f}, δ={delta:.3f}")

                rms = np.sqrt(np.mean((y_arr - y_pred) ** 2))
                print(f"Average RMS error = {rms:.4f}")
            except Exception as e:
                print(f"Warning: Failed to fit joint model: {e}")
                joint_params = None

    hint_params = {}

    for hint_idx, hint in enumerate(hints):
        x_vals = []
        y_vals = []
        y_errs = []
        x_vals_fit = []
        y_vals_fit = []

        for model, x_val in model_x_values:
            accuracy, stderr = results[model][hint]
            if accuracy is not None:
                x_vals.append(x_val)
                y_vals.append(accuracy)
                y_errs.append(stderr if stderr is not None else 0)

                if model in fit_model_names:
                    x_vals_fit.append(x_val)
                    y_vals_fit.append(accuracy)

        if not x_vals:
            continue

        color = colors[hint_idx]
        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=f"{hint}")

        # Fit sigmoid
        min_points_needed = 3 if fit_scaling else 2
        if fit_sigmoid and len(x_vals_fit) >= min_points_needed:
            try:
                x_arr = np.array(x_vals_fit)
                y_arr = np.array(y_vals_fit)

                if fit_scaling:
                    params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr,
                                         p0=[np.max(y_arr), 1, 0],
                                         bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]),
                                         maxfev=10000)
                    hint_params[hint] = params
                    x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                    y_smooth = scaled_sigmoid(x_smooth, *params)
                else:
                    params, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[1, 0], maxfev=10000)
                    hint_params[hint] = params
                    x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                    y_smooth = sigmoid(x_smooth, *params)

                ax.plot(x_smooth, y_smooth, color=color, linestyle='-', linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to fit sigmoid for hint={hint}: {e}")

        # Plot joint model prediction
        if fit_joint and joint_params is not None:
            try:
                x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                C_smooth = np.log(x_smooth)
                H_smooth = np.full_like(C_smooth, hint_transform(hint))
                CH_smooth = np.array([C_smooth, H_smooth])

                if include_cross:
                    y_predicted = joint_sigmoid_with_cross(CH_smooth, *joint_params)
                else:
                    y_predicted = joint_sigmoid_no_cross(CH_smooth, *joint_params)

                ax.plot(x_smooth, y_predicted, color=color, linestyle='--', linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to plot joint prediction for hint={hint}: {e}")

    # Legend
    from matplotlib.lines import Line2D
    hint_handles = []
    hint_labels = []

    for hint_idx, hint in enumerate(hints):
        color = colors[hint_idx]
        handle = Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=8)
        hint_handles.append(handle)

        if transform_hint:
            label = f"{hint} (H'={hint_transform(hint):.2f})"
        else:
            label = f"{hint}"

        if hint in hint_params:
            params = hint_params[hint]
            if fit_scaling:
                h_fit, m_fit, b_fit = params
                equation = f"{h_fit:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
            else:
                m_fit, b_fit = params
                equation = f"σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
            label = f"{label}: {equation}"

        hint_labels.append(label)

    if fit_joint and joint_params is not None:
        H_label = "H'" if transform_hint else "H"
        if include_cross:
            alpha, beta, gamma, delta = joint_params
            joint_eq = f"σ({alpha:.2f}C {beta:+.2f}{H_label} {gamma:+.2f}C{H_label} {delta:+.2f})"
        else:
            alpha, beta, delta = joint_params
            joint_eq = f"σ({alpha:.2f}C {beta:+.2f}{H_label} {delta:+.2f})"
        hint_handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2))
        hint_labels.append(f'{joint_eq}')

    ax.legend(hint_handles, hint_labels, title='hint fraction', framealpha=0.9, loc='upper left')
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale(xscale)

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
