# %%
import matplotlib.pyplot as plt
import os
import sys
# append parent directory to sys.path
sys.path.append("/Users/christineye/emergent-doordash/christine_experiments/20251105")
from plotting import (
    load_all_results,
    plot_results,
    plot_results_rescaled,
    plot_results_by_model_size,
    load_pass_at_k_by_hint,
    plot_pass_at_k_by_hint,
    clean_model_name
)

# %%
# ============= CONFIGURATION =============

# Choose folder structure:
# Option 1: New structure (with solver subfolder)
BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251030/results/hle"
SOLVER = None #"solution"  # Set to None for old structure
FILENAME_TEMPLATE = "hle_cot_0shot_{hint}.json"

# Option 2: Old structure (no solver subfolder)
# BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251030/results/hle"
# SOLVER = None
# FILENAME_TEMPLATE = "hle_0shot_{hint}.json"

CONDITION = "0shot"
GRADER_FIELD = "manual_bootstrap"
ACCURACY_FIELD = "accuracy"
STDERR_FIELD = "stderr"

# Models to plot
MODELS = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
]

HINT_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# =========================================

# %%
# Load all results
print("Loading results...")
results = load_all_results(
    base_folder=BASE_FOLDER,
    models=MODELS,
    hints=HINT_FRACTIONS,
    filename_template=FILENAME_TEMPLATE,
    condition=CONDITION,
    solver=SOLVER,
    grader_field=GRADER_FIELD,
    accuracy_field=ACCURACY_FIELD,
    stderr_field=STDERR_FIELD
)
print("Done!")

# %%
# Plot 1: Accuracy vs Hint Fraction
fig, ax = plot_results(
    results=results,
    models=MODELS,
    hints=HINT_FRACTIONS,
    title="HLE, using CoT-based hints"
)
plt.show()

# %%
# Plot 2: Rescaled plot with sigmoid fitting
fig, ax = plot_results_rescaled(
    results=results,
    models=MODELS,
    hints=HINT_FRACTIONS,
    title="HLE, COT-based hints",
    fit_scaling=False
)
plt.show()

# %%
# Plot 3: Accuracy vs Model Size
fig, ax = plot_results_by_model_size(
    results=results,
    models=MODELS,
    hints=HINT_FRACTIONS,
    title="HLE, using CoT-based hints"
)
plt.show()

# %%
# ============= ADDITIONAL EXAMPLES =============

# Example: GPQA with Gemma models
# BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251015/results/gpqa"
# SOLVER = None
# FILENAME_TEMPLATE = "gpqa_diamond_0shot_{hint}.json"
# CONDITION = "0shot"
# GRADER_FIELD = "manual_bootstrap"
#
# MODELS = [
#     "gemma-3-0.27b-it",
#     "gemma-3-1b-it",
#     "gemma-3-4b-it",
#     "gemma-3-12b-it",
#     "gemma-3-27b-it",
# ]
#
# HINT_FRACTIONS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#
# results = load_all_results(
#     base_folder=BASE_FOLDER,
#     models=MODELS,
#     hints=HINT_FRACTIONS,
#     filename_template=FILENAME_TEMPLATE,
#     condition=CONDITION,
#     solver=SOLVER,
#     grader_field=GRADER_FIELD
# )
#
# fig, ax = plot_results_rescaled(
#     results=results,
#     models=MODELS,
#     hints=HINT_FRACTIONS,
#     title="GPQA Diamond (Gemma): Error Rate vs Inverse Hint",
#     fit_scaling=False
# )
# plt.show()

# %%
# Example: Pass@k plotting for a single model
# MODEL_TO_PLOT = "Qwen2.5-32B-Instruct"
# HINTS_TO_PLOT = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# results_by_hint = load_pass_at_k_by_hint(
#     base_folder=BASE_FOLDER,
#     model=MODEL_TO_PLOT,
#     condition=CONDITION,
#     solver=SOLVER
# )
#
# fig, ax = plot_pass_at_k_by_hint(
#     results_by_hint=results_by_hint,
#     hints=HINTS_TO_PLOT,
#     model_name=clean_model_name(MODEL_TO_PLOT),
#     title=f"GPQA: Pass@k for {clean_model_name(MODEL_TO_PLOT)}"
# )
# plt.show()

# %%
