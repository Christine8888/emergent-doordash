# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# append parent directory to sys.path
sys.path.append("/Users/christineye/emergent-doordash/christine_experiments/20251204")
from plotting import (
    load_all_results,
    plot_results,
    plot_results_rescaled,
    plot_results_by_model_size,
    load_pass_at_k_by_hint,
    plot_pass_at_k_by_hint,
    clean_model_name
)

%load_ext autoreload
%autoreload 2

# %%
# ============= CONFIGURATION =============

# Choose folder structure:
# Option 1: New structure (with solver subfolder)
BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251113/results/gpqa"
SOLVER = "cot_intext_masked"  
LABEL = "using in-text CoT (masked)"
FILENAME_TEMPLATE = "gpqa_" + SOLVER + "_0shot_{hint}.json"

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

HINT_FRACTIONS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
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


# # %%
# # export data as CSV
# with open("gpqa_cot_prefill_data.csv", "w") as f:
#     f.write("model,hint_fraction,accuracy,stderr\n")
#     for model in results:
#         for hint_fraction in results[model]:
#             f.write(f"{model},{hint_fraction},{results[model][hint_fraction][0]},{results[model][hint_fraction][1]}\n")

# %%
# Plot 2: Rescaled plot with sigmoid fitting
# Using log(H/(1-H)) transform (logit transform)
# - H=0 -> -infinity (pinned as upper asymptote for error rate)
# - H=1 -> +infinity (pinned as lower asymptote for error rate)
# - H=0.5 -> 0
fig, ax = plot_results_rescaled(
    results=results,
    models=MODELS,
    hints=HINT_FRACTIONS,
    title="GPQA, " + LABEL,
    fit_scaling=False,
    force_lower_asymptote=1.0,  # H=1.0 -> +inf, use as lower asymptote
    force_upper_asymptote=0.0,  # H=0.0 -> -inf, use as upper asymptote (per-model)
    hint_transform=lambda h: np.log(h / (1 - h)),  # logit transform
    x_label="log(H / (1 - H))"
)
plt.show()

# %%
# Plot 1: Accuracy vs Hint Fraction
fig, ax = plot_results(
    results=results,
    models=MODELS,
    hints=HINT_FRACTIONS,
    title="GPQA, using solution-based hints"
)
plt.show()
# %%
# Plot 3: Accuracy vs Model Size
fig, ax = plot_results_by_model_size(
    results=results,
    models=MODELS,
    hints=HINT_FRACTIONS,
    title="GPQA, pre-filling CoT",
    fit_sigmoid = False,
    fit_joint = True,
    fit_scaling = False,
    include_cross = True,
    fit_models=[],
    exclude_hint=[],
    transform_hint=True,
)
plt.show()

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
