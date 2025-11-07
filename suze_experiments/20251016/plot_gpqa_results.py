#!/usr/bin/env python3
"""Plot GPQA evaluation results for OLMo models."""

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

# ============= CONFIGURATION =============

# Base directory containing results
BASE_FOLDER = "/afs/cs.stanford.edu/u/suzeva/emergent-doordash/suze_experiments/20251016/results/gpqa/0shot"

# Models to plot
MODELS = [
    "OLMo-2-0425-1B",
    "OLMo-2-0425-1B-SFT",
]

# Result filename
FILENAME = "gpqa_diamond_0shot_0.0.json"

# Field to extract accuracy from
GRADER_FIELD = "choice"
ACCURACY_FIELD = "accuracy"
STDERR_FIELD = "stderr"

# Output configuration
OUTPUT_FILE = "gpqa_results_plot.png"
DPI = 300

# =========================================


def load_result(base_folder: str, model: str, filename: str) -> Optional[Dict]:
    """Load a single result JSON file.
    
    Args:
        base_folder: Base directory containing results
        model: Model name
        filename: Filename to load
        
    Returns:
        Dictionary with result data, or None if file not found
    """
    filepath = Path(base_folder) / model / filename
    
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_accuracy_and_stderr(result: Dict, 
                                grader_field: str = 'choice',
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


def clean_model_name(model: str) -> str:
    """Clean up model names for display.
    
    Examples:
        OLMo-2-0425-1B -> OLMo-1B
        OLMo-2-0425-1B-SFT -> OLMo-1B-SFT
    """
    if "SFT" in model:
        return "OLMo-1B-SFT"
    return "OLMo-1B"


def plot_model_comparison(models: List[str], 
                          accuracies: List[float], 
                          stderrs: List[float],
                          title: str = "GPQA Accuracy Comparison",
                          figsize: Tuple[int, int] = (12, 7)):
    """Plot comparison of model accuracies using errorbar style.
    
    Args:
        models: List of model names
        accuracies: List of accuracy values
        stderrs: List of stderr values
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean model names
    display_names = [clean_model_name(m) for m in models]
    
    # Get color palette
    colors = sns.color_palette("husl", len(models))
    
    # Create positions with more spacing
    x_pos = np.arange(len(models)) * 2  # Multiply by 2 for wider spacing
    
    # Plot each model with errorbar
    for i, (model, accuracy, stderr) in enumerate(zip(display_names, accuracies, stderrs)):
        ax.errorbar(x_pos[i], accuracy, yerr=stderr,
                   color=colors[i], linestyle='none',
                   marker='o', markersize=10, capsize=5,
                   linewidth=2, alpha=0.8, label=model)
    
    # Formatting
    ax.set_xlabel('model', fontsize=13, fontweight='bold')
    ax.set_ylabel('eval accuracy', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, title='Models')
    
    # Adjust axis limits to zoom in on data
    ax.set_xlim(-1, x_pos[-1] + 1)
    y_min = min(accuracies) - max(stderrs) * 3
    y_max = max(accuracies) + max(stderrs) * 3
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig, ax


def main():
    """Main function to load data and generate plot."""
    print("Loading results...")
    
    accuracies = []
    stderrs = []
    valid_models = []
    
    for model in MODELS:
        result = load_result(BASE_FOLDER, model, FILENAME)
        accuracy, stderr = extract_accuracy_and_stderr(
            result, GRADER_FIELD, ACCURACY_FIELD, STDERR_FIELD
        )
        
        if accuracy is not None:
            accuracies.append(accuracy)
            stderrs.append(stderr if stderr is not None else 0)
            valid_models.append(model)
            print(f"  {model}: accuracy={accuracy:.4f}, stderr={stderr:.4f}")
        else:
            print(f"  {model}: No data available")
    
    if not valid_models:
        print("Error: No valid results found!")
        return
    
    print(f"\nGenerating plot for {len(valid_models)} model(s)...")
    fig, ax = plot_model_comparison(
        models=valid_models,
        accuracies=accuracies,
        stderrs=stderrs,
        title="GPQA: accuracy vs model"
    )
    
    # Save figure
    output_path = Path(BASE_FOLDER).parent.parent.parent / OUTPUT_FILE
    print(f"Saving plot to {output_path}...")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved successfully!")
    
    plt.close(fig)


if __name__ == "__main__":
    # python suze_experiments/20251016/plot_gpqa_results.py
    main()

