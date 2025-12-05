# %%
"""Plot baseline evaluation results across model sizes."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============= UTILITIES =============

def clean_model_name(model: str) -> str:
    """Clean up model names for display (e.g., Qwen2.5-7B-Instruct -> 7B)."""
    parts = model.upper().split("-")
    for part in parts:
        if "B" in part:
            return part
    return model


def extract_model_size_billions(model: str) -> float:
    """Extract model size in billions from model name."""
    parts = model.upper().split("-")
    for part in parts:
        if "B" in part:
            return float(part.replace("B", ""))
    return 0.0




# ============= DATA LOADING =============

def load_baseline_results(
    base_folder: str,
    eval_name: str,
    models: List[str],
    grader_field: str,
    accuracy_field: str = "accuracy",
    stderr_field: str = "stderr"
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Load baseline results for multiple models.
    
    Args:
        base_folder: Base directory containing baseline results (e.g., .../baseline)
        eval_name: Name of the evaluation (e.g., 'ifeval', 'math', 'mmlu_0_shot')
        models: List of model names to load
        grader_field: Field containing grader results (e.g., 'instruction_following', 'expression_exact_match_sympy')
        accuracy_field: Field name for accuracy within grader_field
        stderr_field: Field name for stderr within grader_field
    
    Returns:
        Dictionary: {model: (accuracy, stderr)}
    """
    results = {}
    
    for model in models:
        filepath = Path(base_folder) / eval_name / model / f"{eval_name}.json"
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            results[model] = (None, None)
            continue
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if grader_field in data:
            accuracy = data[grader_field].get(accuracy_field)
            stderr = data[grader_field].get(stderr_field)
            results[model] = (accuracy, stderr)
        else:
            print(f"Warning: Grader field '{grader_field}' not found in {filepath}")
            results[model] = (None, None)
    
    return results


def load_multiple_baselines(
    base_folder: str,
    eval_configs: List[Dict],
    models: List[str]
) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """Load multiple baseline evaluations.
    
    Args:
        base_folder: Base directory containing baseline results
        eval_configs: List of dicts with keys: 'eval_name', 'grader_field', 'accuracy_field', 'stderr_field', 'label'
        models: List of model names to load
    
    Returns:
        Dictionary: {label: {model: (accuracy, stderr)}}
    """
    all_results = {}
    
    for config in eval_configs:
        label = config.get('label', config['eval_name'])
        results = load_baseline_results(
            base_folder=base_folder,
            eval_name=config['eval_name'],
            models=models,
            grader_field=config['grader_field'],
            accuracy_field=config.get('accuracy_field', 'accuracy'),
            stderr_field=config.get('stderr_field', 'stderr')
        )
        all_results[label] = results
    
    return all_results


# ============= SIGMOID FITTING =============

def sigmoid(x, m, b):
    """Basic sigmoid: σ(m*log(x) + b)"""
    return 1 / (1 + np.exp(-(m * np.log(x) + b)))


def scaled_sigmoid(x, h, m, b):
    """Scaled sigmoid: h*σ(m*log(x) + b)"""
    return h / (1 + np.exp(-(m * np.log(x) + b)))


def make_bounded_sigmoid(L, U):
    """Create sigmoid with fixed bounds: L + (U-L)*σ(m*log(x) + b)"""
    def bounded_sigmoid(x, m, b):
        return L + (U - L) / (1 + np.exp(-(m * np.log(x) + b)))
    return bounded_sigmoid


def make_lower_bounded_sigmoid(L):
    """Create sigmoid with fixed lower bound: L + h*σ(m*log(x) + b)"""
    def lower_bounded_sigmoid(x, h, m, b):
        return L + h / (1 + np.exp(-(m * np.log(x) + b)))
    return lower_bounded_sigmoid


# ============= PLOTTING =============

def plot_baselines(
    results: Dict[str, Tuple[Optional[float], Optional[float]]],
    models: List[str],
    title: str = "Accuracy vs Model Size",
    figsize: Tuple[int, int] = (10, 6),
    fit_sigmoid: bool = True,
    fit_scaling: bool = False,
    pin_lower_bound: Optional[float] = None,
    upper_bound: float = 1.0,
    color: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot baseline results with accuracy vs log model size (in billions).
    
    Args:
        results: Dictionary {model: (accuracy, stderr)} from load_baseline_results
        models: List of model names (determines order)
        title: Plot title
        figsize: Figure size
        fit_sigmoid: If True, fit a sigmoid curve
        fit_scaling: If True, fit y = h*σ(...), else y = σ(...)
        pin_lower_bound: If set, pin the lower asymptote to this value
        upper_bound: Upper bound for sigmoid (used with pin_lower_bound when not fit_scaling)
        color: Optional color for the plot
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if color is None:
        color = sns.color_palette("husl", 1)[0]
    
    # Extract data
    x_vals = []
    y_vals = []
    y_errs = []
    labels = []
    
    for model in models:
        accuracy, stderr = results.get(model, (None, None))
        if accuracy is not None:
            x_vals.append(extract_model_size_billions(model))
            y_vals.append(accuracy)
            y_errs.append(stderr if stderr is not None else 0)
            labels.append(clean_model_name(model))
    
    if not x_vals:
        print("Warning: No data to plot")
        return fig, ax
    
    # Plot data points
    ax.errorbar(x_vals, y_vals, yerr=y_errs,
               color=color, linestyle='none',
               marker='o', markersize=10, capsize=5,
               linewidth=2, alpha=0.8)
    
    # Add labels for each point
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(label, (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, alpha=0.7)
    
    # Fit and plot sigmoid
    fit_params = None
    if fit_sigmoid and len(x_vals) >= 2:
        try:
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            
            if pin_lower_bound is not None:
                L = pin_lower_bound
                if fit_scaling:
                    # Fit: L + h*σ(m*log(x) + b), h in [0, upper_bound - L]
                    fit_func = make_lower_bounded_sigmoid(L)
                    h_max = upper_bound - L
                    params, _ = curve_fit(fit_func, x_arr, y_arr,
                                         p0=[min(np.max(y_arr) - L, h_max), 1, 0],
                                         bounds=([0, -np.inf, -np.inf], [h_max, np.inf, np.inf]),
                                         maxfev=10000)
                    h_fit, m_fit, b_fit = params
                    fit_params = ('bounded_scaled', L, h_fit, m_fit, b_fit)
                else:
                    # Fit: L + (U-L)*σ(m*log(x) + b)
                    U = upper_bound
                    fit_func = make_bounded_sigmoid(L, U)
                    params, _ = curve_fit(fit_func, x_arr, y_arr,
                                         p0=[1, 0], maxfev=10000)
                    m_fit, b_fit = params
                    fit_params = ('bounded', L, U, m_fit, b_fit)
            else:
                if fit_scaling:
                    # Fit: h*σ(m*log(x) + b)
                    params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr,
                                         p0=[np.max(y_arr), 1, 0], maxfev=10000)
                    h_fit, m_fit, b_fit = params
                    fit_params = ('scaled', h_fit, m_fit, b_fit)
                else:
                    # Fit: σ(m*log(x) + b)
                    params, _ = curve_fit(sigmoid, x_arr, y_arr,
                                         p0=[1, 0], maxfev=10000)
                    m_fit, b_fit = params
                    fit_params = ('basic', m_fit, b_fit)
            
            # Plot fitted curve
            x_smooth = np.logspace(np.log10(min(x_vals) * 0.5),
                                  np.log10(max(x_vals) * 2), 100)
            
            if fit_params[0] == 'bounded_scaled':
                _, L, h_fit, m_fit, b_fit = fit_params
                y_smooth = make_lower_bounded_sigmoid(L)(x_smooth, h_fit, m_fit, b_fit)
            elif fit_params[0] == 'bounded':
                _, L, U, m_fit, b_fit = fit_params
                y_smooth = make_bounded_sigmoid(L, U)(x_smooth, m_fit, b_fit)
            elif fit_params[0] == 'scaled':
                _, h_fit, m_fit, b_fit = fit_params
                y_smooth = scaled_sigmoid(x_smooth, h_fit, m_fit, b_fit)
            else:
                _, m_fit, b_fit = fit_params
                y_smooth = sigmoid(x_smooth, m_fit, b_fit)
            
            ax.plot(x_smooth, y_smooth, color=color,
                   linestyle='--', linewidth=2, alpha=0.6)
            
        except Exception as e:
            print(f"Warning: Failed to fit sigmoid: {e}")
    
    # Create legend with equation
    if fit_params is not None:
        if fit_params[0] == 'bounded_scaled':
            _, L, h_fit, m_fit, b_fit = fit_params
            equation = f"{L:.3f} + {h_fit:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
        elif fit_params[0] == 'bounded':
            _, L, U, m_fit, b_fit = fit_params
            equation = f"{L:.3f} + {U-L:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
        elif fit_params[0] == 'scaled':
            _, h_fit, m_fit, b_fit = fit_params
            equation = f"{h_fit:.3f}·σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
        else:
            _, m_fit, b_fit = fit_params
            equation = f"σ({m_fit:.2f}·log(x) {b_fit:+.2f})"
        
        ax.text(0.05, 0.95, equation, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Model Size (B)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    return fig, ax


def plot_multiple_baselines(
    all_results: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    models: List[str],
    title: str = "Baseline Evaluations vs Model Size",
    figsize: Tuple[int, int] = (12, 7),
    fit_sigmoid: bool = True,
    fit_scaling: bool = False,
    pin_lower_bound: Optional[float] = None,
    upper_bound: float = 1.0
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple baseline evaluations on the same axes.
    
    Args:
        all_results: Dictionary {label: {model: (accuracy, stderr)}}
        models: List of model names
        title: Plot title
        figsize: Figure size
        fit_sigmoid: If True, fit sigmoid curves
        fit_scaling: If True, fit y = h*σ(...)
        pin_lower_bound: If set, pin lower asymptote
        upper_bound: Upper bound for sigmoid
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(all_results))
    
    for (label, results), color in zip(all_results.items(), colors):
        x_vals = []
        y_vals = []
        y_errs = []
        
        for model in models:
            accuracy, stderr = results.get(model, (None, None))
            if accuracy is not None:
                x_vals.append(extract_model_size_billions(model))
                y_vals.append(accuracy)
                y_errs.append(stderr if stderr is not None else 0)
        
        if not x_vals:
            continue
        
        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                   color=color, linestyle='none',
                   marker='o', markersize=8, capsize=4,
                   linewidth=2, alpha=0.8, label=label)
        
        # Fit sigmoid if requested
        if fit_sigmoid and len(x_vals) >= 2:
            try:
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)
                
                if pin_lower_bound is not None:
                    L = pin_lower_bound
                    if fit_scaling:
                        fit_func = make_lower_bounded_sigmoid(L)
                        h_max = upper_bound - L
                        params, _ = curve_fit(fit_func, x_arr, y_arr,
                                             p0=[min(np.max(y_arr) - L, h_max), 1, 0],
                                             bounds=([0, -np.inf, -np.inf], [h_max, np.inf, np.inf]),
                                             maxfev=10000)
                        x_smooth = np.logspace(np.log10(min(x_vals) * 0.5),
                                              np.log10(max(x_vals) * 2), 100)
                        y_smooth = fit_func(x_smooth, *params)
                    else:
                        U = upper_bound
                        fit_func = make_bounded_sigmoid(L, U)
                        params, _ = curve_fit(fit_func, x_arr, y_arr,
                                             p0=[1, 0], maxfev=10000)
                        x_smooth = np.logspace(np.log10(min(x_vals) * 0.5),
                                              np.log10(max(x_vals) * 2), 100)
                        y_smooth = fit_func(x_smooth, *params)
                else:
                    if fit_scaling:
                        params, _ = curve_fit(scaled_sigmoid, x_arr, y_arr,
                                             p0=[np.max(y_arr), 1, 0], maxfev=10000)
                        x_smooth = np.logspace(np.log10(min(x_vals) * 0.5),
                                              np.log10(max(x_vals) * 2), 100)
                        y_smooth = scaled_sigmoid(x_smooth, *params)
                    else:
                        params, _ = curve_fit(sigmoid, x_arr, y_arr,
                                             p0=[1, 0], maxfev=10000)
                        x_smooth = np.logspace(np.log10(min(x_vals) * 0.5),
                                              np.log10(max(x_vals) * 2), 100)
                        y_smooth = sigmoid(x_smooth, *params)
                
                ax.plot(x_smooth, y_smooth, color=color,
                       linestyle='--', linewidth=2, alpha=0.6)
            except Exception as e:
                print(f"Warning: Failed to fit sigmoid for {label}: {e}")
    
    ax.legend(loc='lower right', title='Evaluation', framealpha=0.9)
    ax.set_xlabel('Model Size (B)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    return fig, ax


# ============= EXAMPLE USAGE =============

if __name__ == "__main__":
    # Configuration
    BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251113/baseline"
    
    MODELS = [
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
    ]
    
    # Example 1: Single evaluation (IFEval)
    results = load_baseline_results(
        base_folder=BASE_FOLDER,
        eval_name="ifeval",
        models=MODELS,
        grader_field="instruction_following",
        accuracy_field="final_acc",
        stderr_field="final_stderr"
    )
    
    fig, ax = plot_baselines(
        results=results,
        models=MODELS,
        title="IFEval: Accuracy vs Model Size",
        fit_sigmoid=True,
        fit_scaling=True,
        pin_lower_bound=None  # Set to e.g. 0.25 to pin lower bound
    )
    plt.show()

# %%
# ============= INTERACTIVE USAGE =============

BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251113/baseline"

MODELS = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
]

# %%
# Plot IFEval
results_ifeval = load_baseline_results(
    base_folder=BASE_FOLDER,
    eval_name="ifeval",
    models=MODELS,
    grader_field="instruction_following",
    accuracy_field="final_acc",
    stderr_field="final_stderr"
)

fig, ax = plot_baselines(
    results=results_ifeval,
    models=MODELS,
    title="IFEval: Accuracy vs Model Size",
    fit_sigmoid=True,
    fit_scaling=True,
    pin_lower_bound=None
)
plt.show()

# %%
# Plot MATH
results_math = load_baseline_results(
    base_folder=BASE_FOLDER,
    eval_name="math",
    models=MODELS,
    grader_field="expression_exact_match_sympy",
    accuracy_field="accuracy",
    stderr_field="stderr"
)

fig, ax = plot_baselines(
    results=results_math,
    models=MODELS,
    title="MATH: Accuracy vs Model Size",
    fit_sigmoid=True,
    fit_scaling=True,
    pin_lower_bound=None
)
plt.show()

# %%
# Plot multiple evaluations together
eval_configs = [
    {
        'eval_name': 'ifeval',
        'grader_field': 'instruction_following',
        'accuracy_field': 'final_acc',
        'stderr_field': 'final_stderr',
        'label': 'IFEval'
    },
    {
        'eval_name': 'math',
        'grader_field': 'expression_exact_match_sympy',
        'accuracy_field': 'accuracy',
        'stderr_field': 'stderr',
        'label': 'MATH'
    }
]

all_results = load_multiple_baselines(
    base_folder=BASE_FOLDER,
    eval_configs=eval_configs,
    models=MODELS
)

fig, ax = plot_multiple_baselines(
    all_results=all_results,
    models=MODELS,
    title="Baseline Evaluations vs Model Size",
    fit_sigmoid=True,
    fit_scaling=True
)
plt.show()

