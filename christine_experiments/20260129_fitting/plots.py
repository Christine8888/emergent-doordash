# %%
"""Joint scaling plots with ECI on x-axis for GPQA with all hint levels."""

import sys
from pathlib import Path

# Project root (works on any machine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "christine_experiments/20251204"))
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.modelx import load_results, load_baseline, clean_name, size, model_eci
from plotting import plot_by_x_axis, plot_accuracy_vs_hint
from plot_baselines import plot_baseline

%load_ext autoreload
%autoreload 2

# %%
# Configuration
BASE_FOLDER = str(PROJECT_ROOT / "christine_experiments/20251113/results")
BASELINE_FOLDER = str(PROJECT_ROOT / "christine_experiments/20251113/baseline")
ECI_FILE = str(PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv")

EVAL_NAME = "gpqa"
SOLVER = "solution_intext_masked"
LABEL = "solution intext masked"

MODELS = [
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
]

HINT_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# %%
# Load fitted ECI values
eci_df = pd.read_csv(ECI_FILE)
eci_map = dict(zip(eci_df["model"], eci_df["Cm"]))

print("Fitted ECIs for models:")
for model in sorted(MODELS, key=lambda m: eci_map.get(m, 0)):
    eci = eci_map.get(model)
    if eci:
        print(f"  {model:35s} {eci:6.1f}")
    else:
        print(f"  {model:35s} MISSING")

# %%
# Load results
print("\nLoading results...")
df = load_results(
    base_folder=BASE_FOLDER,
    eval_name=EVAL_NAME,
    solver=SOLVER,
    condition="0shot",
)

# Filter to selected models and hints
df = df[df["model"].isin(MODELS) & df["hint"].isin(HINT_FRACTIONS)]
print(f"Loaded {len(df)} rows for {df['model'].nunique()} models")

# Add ECI column
df["eci"] = df["model"].map(eci_map)

# Check for missing ECIs
missing = df[df["eci"].isna()]["model"].unique()
if len(missing) > 0:
    print(f"WARNING: Missing ECI for {len(missing)} models: {missing.tolist()}")
    df = df.dropna(subset=["eci"])
# %%
# Plot 2: Accuracy vs ECI (linear scale), all hint levels
# Custom plotting since plot_by_x_axis uses model_eci() which reads from old cache
fig, ax = plt.subplots(figsize=(10, 6))

hints = sorted(df["hint"].unique())
cmap = plt.cm.viridis
colors = {h: cmap(i / (len(hints) - 1)) for i, h in enumerate(hints)}

for hint in hints:
    hint_df = df[df["hint"] == hint].sort_values("eci")
    ax.plot(
        hint_df["eci"],
        hint_df["accuracy"],
        "o-",
        color=colors[hint],
        label=f"hint={hint:.1f}",
        alpha=0.7,
    )

ax.set_xlabel("ECI (Epoch Capabilities Index)")
ax.set_ylabel("Accuracy")
ax.set_title(f"GPQA Accuracy vs ECI, {LABEL}")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot 3: Accuracy vs ECI with joint sigmoid fit
from src.modelx import fit_joint_sigmoid, format_equation

# Prepare data for joint fitting
fit_df = df[["model", "eci", "hint", "accuracy"]].copy()
fit_df = fit_df.rename(columns={"eci": "model_size"})  # fit_joint_sigmoid expects model_size

# Fit joint model: σ(α·ECI + β·H + γ·ECI·H + δ)
joint_fit = fit_joint_sigmoid(
    fit_df,
    x_col="model_size",
    y_col="accuracy",
    hint_col="hint",
    use_log_x=False,  # ECI is already on capability scale, no log needed
    include_cross=True,
    hint_transform=lambda h: h,  # identity transform for hints
)

if joint_fit:
    print(f"Joint fit: α={joint_fit['params'][0]:.3f}, β={joint_fit['params'][1]:.3f}, "
          f"γ={joint_fit['params'][2]:.3f}, δ={joint_fit['params'][3]:.3f}")
    print(f"RMS error: {joint_fit['rms']:.4f}")

# Plot with joint fit curves
fig, ax = plt.subplots(figsize=(10, 6))

eci_range = np.linspace(df["eci"].min() - 5, df["eci"].max() + 5, 100)

for hint in hints:
    hint_df = df[df["hint"] == hint].sort_values("eci")
    ax.scatter(
        hint_df["eci"],
        hint_df["accuracy"],
        color=colors[hint],
        label=f"hint={hint:.1f}",
        alpha=0.7,
        s=50,
    )

    # Plot joint fit curve
    if joint_fit:
        y_fit = joint_fit["predict"](eci_range, hint)
        ax.plot(eci_range, y_fit, "-", color=colors[hint], alpha=0.5, linewidth=1.5)

ax.set_xlabel("ECI (Epoch Capabilities Index)")
ax.set_ylabel("Accuracy")
ax.set_title(f"GPQA Accuracy vs ECI, {LABEL}\nJoint fit: σ(α·ECI + β·H + γ·ECI·H + δ)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot 4: Baseline comparison
baseline_df = load_baseline(BASELINE_FOLDER, "gpqa")
baseline_df["eci"] = baseline_df["model"].map(eci_map)
baseline_df = baseline_df.dropna(subset=["eci"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(baseline_df["eci"], baseline_df["accuracy"], s=80, alpha=0.8)

for _, row in baseline_df.iterrows():
    ax.annotate(
        clean_name(row["model"]),
        (row["eci"], row["accuracy"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )

ax.set_xlabel("ECI")
ax.set_ylabel("Accuracy")
ax.set_title("GPQA Baseline Accuracy vs ECI")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
