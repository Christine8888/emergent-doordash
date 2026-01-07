# %%
import matplotlib.pyplot as plt
import numpy as np

from src.modelx import load_results, clean_name
from plotting import (
    plot_accuracy_vs_hint,
    plot_error_vs_hint_transformed,
    plot_accuracy_vs_size,
)

%load_ext autoreload
%autoreload 2

# %%
# ============= CONFIGURATION =============

BASE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251113/results"
EVAL_NAME = "gpqa"
SOLVER = "solution_prefill_sequential"
LABEL = "using solution prefill (sequential)"

# Models to plot (filter after loading)
MODELS = [
#    "Qwen2.5-0.5B-Instruct",
#     "Qwen2.5-1.5B-Instruct",
#     "Qwen2.5-3B-Instruct",
#     "Qwen2.5-7B-Instruct",
#     "Qwen2.5-14B-Instruct",
#     "Qwen2.5-32B-Instruct",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    # "Qwen3-4B",
    # "Qwen3-8B",
    # "Qwen3-14B",
    # "Qwen3-32B",
    # "Llama-3.1-8B-Instruct",
    # "Llama-3.1-70B-Instruct",
    # "gemma-3-12b-it",
    # "gemma-3-4b-it",
    # "gemma-3-27b-it",
]

HINT_FRACTIONS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# =========================================

# %%
# Load all results into DataFrame
print("Loading results...")
df = load_results(
    base_folder=BASE_FOLDER,
    eval_name=EVAL_NAME,
    solver=SOLVER,
    condition="0shot",
)

# Filter to selected models and hints
df = df[df["model"].isin(MODELS) & df["hint"].isin(HINT_FRACTIONS)]
print(f"Loaded {len(df)} rows for {df['model'].nunique()} models")


# # %%
# # export data as CSV
# df.to_csv("gpqa_cot_prefill_data.csv", index=False)

# %%
# Plot 2: Rescaled plot with sigmoid fitting
# Using log(H/(1-H)) transform (logit transform)
# - H=0 -> -infinity (pinned as upper asymptote for error rate)
# - H=1 -> +infinity (pinned as lower asymptote for error rate)
# - H=0.5 -> 0
fig, ax = plot_error_vs_hint_transformed(
    df,
    title="GPQA, " + LABEL,
    fit_scaling=False,
    lower_asymptote_hint=1.0,  # H=1.0 -> +inf, use as lower asymptote
    upper_asymptote_hint=0.0,  # H=0.0 -> -inf, use as upper asymptote (per-model)
    hint_transform=lambda h: np.log(h / (1 - h)),  # logit transform
    x_label="log(H / (1 - H))"
)
plt.show()

# %%
# Plot 1: Accuracy vs Hint Fraction
fig, ax = plot_accuracy_vs_hint(
    df,
    title="GPQA, using solution-based hints"
)
plt.show()

# %%
# Plot 3: Accuracy vs Model Size
fig, ax = plot_accuracy_vs_size(
    df,
    title="GPQA, pre-filling CoT",
    fit_sigmoid_curves=False,
    fit_joint=True,
    fit_scaling=False,
    include_cross=True,
    fit_models=None,
    exclude_hints=[],
    hint_transform=lambda h: np.log(1 / (1 - h)),  # transform_hint=True equivalent
)
plt.show()

# %%
# Example: Pass@k plotting for a single model (commented out)
# from plotting import load_pass_at_k_by_hint, plot_pass_at_k_by_hint
# MODEL_TO_PLOT = "Qwen2.5-32B-Instruct"
# HINTS_TO_PLOT = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#
# results_by_hint = load_pass_at_k_by_hint(
#     base_folder=BASE_FOLDER + "/" + EVAL_NAME,
#     model=MODEL_TO_PLOT,
#     condition="0shot",
#     solver=SOLVER
# )
#
# fig, ax = plot_pass_at_k_by_hint(
#     results_by_hint=results_by_hint,
#     hints=HINTS_TO_PLOT,
#     model_name=clean_name(MODEL_TO_PLOT),
#     title=f"GPQA: Pass@k for {clean_name(MODEL_TO_PLOT)}"
# )
# plt.show()

# %%
