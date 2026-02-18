# %%
"""Fit ECI from baseline data.

Two modes:
- "simple": Use Epoch's pre-fitted D and α, only estimate C for your models
- "full": Re-fit all parameters (C, D, α) jointly with Epoch's data
"""

import json
import sys
from pathlib import Path
from typing import Any

# Project root (works on any machine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.modelx import (
    load_baseline,
    load_epoch_benchmark_scores,
    load_epoch_params,
    load_epoch_eci,
    fit_eci,
    estimate_eci_from_epoch_params,
)

%load_ext autoreload
%autoreload 2

# %%
# Configuration
MODE = "simple"  # "simple" or "full"
PRIORITIZE_EPOCH = False  # If True, use Epoch's ECI when available instead of fitting

BASELINE_FOLDER = PROJECT_ROOT / "christine_experiments/20251113/baseline"
OUTPUT_DIR = PROJECT_ROOT / "christine_experiments/20260129_fitting"
EPOCH_ECI_FILE = PROJECT_ROOT / "src/modelx/eci/eci_scores.json"

# Mapping from baseline eval names to ECI benchmark names
EVAL_TO_ECI = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "mmlu_5_shot_cot": "MMLU",
    "bbh": "BBH",
    "arc_challenge": "ARC AI2",  # Epoch only uses Challenge score
    "winogrande": "Winogrande",  # 0-shot, 8192 tokens
    # "math_level_5": "MATH level 5",
}

# All user models are expected to have scores for all of these benchmarks.
REQUIRED_ECI_BENCHMARKS = sorted(set(EVAL_TO_ECI.values()))

# For full mode only: anchor models for rescaling
ANCHOR1 = ("claude-3-5-sonnet-20240620", 130.0)
ANCHOR2 = ("gpt-5-2025-08-07_medium", 150.0)

# For full mode only: benchmarks to exclude
EXCLUDE_BENCHMARKS = ["OTIS Mock AIME 2024-2025"]


def _raise_if_missing_required_scores(
    scores: pd.DataFrame,
    required_benchmarks: list[str],
    *,
    only_models: list[str] | None = None,
    label: str,
) -> None:
    """Raise if any model is missing any required benchmark score."""
    required = set(required_benchmarks)
    models = sorted(set(scores["model"].unique())) if only_models is None else sorted(set(only_models))

    missing_by_model: dict[str, list[str]] = {}
    for model in models:
        have = set(scores.loc[scores["model"] == model, "benchmark"].unique())
        missing = sorted(required - have)
        if missing:
            missing_by_model[model] = missing

    if not missing_by_model:
        return

    lines = [
        f"{label}: missing required benchmark scores for {len(missing_by_model)} model(s).",
        f"Required benchmarks ({len(required_benchmarks)}): {required_benchmarks}",
        "",
    ]
    for model in sorted(missing_by_model.keys()):
        missing = missing_by_model[model]
        lines.append(f"- {model}: missing {len(missing)} -> {missing}")
    raise ValueError("\n".join(lines))


# %%
# Load Epoch's pre-computed ECI scores for comparison
epoch_eci = load_epoch_eci() # this loads existing eci scores
print(epoch_eci)
print(f"Loaded {len(epoch_eci)} Epoch ECI scores for comparison")

# %%
# Load user baseline scores
print("\nLoading baseline results...")
user_rows = []

for eval_name, eci_benchmark in EVAL_TO_ECI.items():
    df = load_baseline(str(BASELINE_FOLDER), eval_name) # loads benchmark scores for baselines: no hinting, just model performance on a specific benchmark
    if df.empty:
        print(f"  {eval_name}: no results")
        continue

    n_models = len(df)
    print(f"  {eval_name} -> {eci_benchmark}: {n_models} models")

    for _, row in df.iterrows():
        if pd.notna(row.get("accuracy")):
            user_rows.append({
                "model": row["model"],
                "benchmark": eci_benchmark,
                "score": row["accuracy"],
            })

user_scores = pd.DataFrame(user_rows)
user_models = list(user_scores["model"].unique())
print(f"\nTotal user scores: {len(user_scores)}")
print(f"User models: {len(user_models)}")

# Enforce completeness: every user model must have every required ECI benchmark score.
_raise_if_missing_required_scores(
    user_scores,
    REQUIRED_ECI_BENCHMARKS,
    only_models=user_models,
    label="User baseline results",
)

# %%
# Fit ECI based on mode
print(f"\n{'='*70}")
print(f"MODE: {MODE}")
print(f"{'='*70}")

if MODE == "simple":
    # Simple mode: use Epoch's pre-fitted D and α, only estimate C
    print("\nUsing Epoch's pre-fitted benchmark parameters (D, α)")
    print("Only estimating model capabilities (C)")

    eci_scores = estimate_eci_from_epoch_params(
        user_scores,
        min_benchmarks=len(REQUIRED_ECI_BENCHMARKS),
    )

    print(f"\nEstimated ECI for {len(eci_scores)} models")

elif MODE == "full":
    # Full mode: re-fit everything jointly with Epoch data
    print("\nRe-fitting all parameters (C, D, α) jointly")
    print(f"Excluding benchmarks: {EXCLUDE_BENCHMARKS}")

    # Load Epoch benchmark scores and combine
    print("\nLoading Epoch benchmark scores...")
    epoch_scores = load_epoch_benchmark_scores(only_eci_models=True)
    print(f"Epoch scores: {len(epoch_scores)} ({epoch_scores['model'].nunique()} models)")

    combined = pd.concat([epoch_scores, user_scores], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "benchmark"], keep="last")
    print(f"Combined: {len(combined)} scores, {combined['model'].nunique()} models")

    # Enforce completeness on all models used for fitting.
    required_for_fit = sorted(set(REQUIRED_ECI_BENCHMARKS) - set(EXCLUDE_BENCHMARKS))
    _raise_if_missing_required_scores(
        combined,
        required_for_fit,
        label="Combined Epoch + user data (for fitting)",
    )

    # Fit
    result = fit_eci(
        combined,
        anchor_benchmark="Winogrande",
        reg_strength=0.1,
        min_benchmarks=len(required_for_fit),
        exclude_benchmarks=EXCLUDE_BENCHMARKS,
    )
    print(f"RMSE: {result['rmse']:.4f}")

    # Rescale using two anchors
    print(f"\nRescaling with anchors:")
    print(f"  {ANCHOR1[0]} = {ANCHOR1[1]}")
    print(f"  {ANCHOR2[0]} = {ANCHOR2[1]}")

    raw1 = result["Cm"].get(ANCHOR1[0])
    raw2 = result["Cm"].get(ANCHOR2[0])

    if raw1 is None or raw2 is None:
        raise ValueError(f"Anchor models not found in fit results")

    scale = (ANCHOR2[1] - ANCHOR1[1]) / (raw2 - raw1)
    offset = ANCHOR1[1] - scale * raw1
    print(f"Transform: ECI = {scale:.4f} * raw + {offset:.2f}")

    eci_scores = {m: scale * c + offset for m, c in result["Cm"].items()}

    # Debug: Compare fitted benchmark params to Epoch's (on same scale)
    print("\nBenchmark parameters (fitted vs Epoch, rescaled):")
    epoch_params = load_epoch_params()
    anchor = result["anchor_benchmark"]
    k = epoch_params["slope"].get(anchor, 0.0454)
    d_offset = epoch_params["difficulty"].get(anchor, 109.75)

    print(f"{'Benchmark':<25} {'D_fit':>8} {'D_epoch':>8} {'α_fit':>8} {'α_epoch':>8}")
    for bench in sorted(result["Db"].keys()):
        d_fit_scaled = result["Db"][bench] / k + d_offset
        a_fit_scaled = result["ab"][bench] * k
        d_epoch = epoch_params["difficulty"].get(bench, None)
        a_epoch = epoch_params["slope"].get(bench, None)
        if d_epoch:
            print(f"  {bench:<23} {d_fit_scaled:>8.1f} {d_epoch:>8.1f} {a_fit_scaled:>8.4f} {a_epoch:>8.4f}")
        else:
            print(f"  {bench:<23} {d_fit_scaled:>8.1f} {'N/A':>8} {a_fit_scaled:>8.4f} {'N/A':>8}")

else:
    raise ValueError(f"Unknown mode: {MODE}. Use 'simple' or 'full'.")

# %%
# Apply PRIORITIZE_EPOCH: use Epoch's ECI when available
eci_fitted = eci_scores.copy()  # Keep original fitted values for display
eci_final = {}

if PRIORITIZE_EPOCH:
    n_from_epoch = 0
    n_from_fitted = 0
    for model in eci_scores:
        if model in epoch_eci:
            eci_final[model] = epoch_eci[model]
            n_from_epoch += 1
        else:
            eci_final[model] = eci_scores[model]
            n_from_fitted += 1
    print(f"\nPRIORITIZE_EPOCH=True:")
    print(f"  Using Epoch's ECI: {n_from_epoch} models")
    print(f"  Using fitted ECI: {n_from_fitted} models")
else:
    eci_final = eci_scores.copy()
    print(f"\nPRIORITIZE_EPOCH=False: Using fitted ECI for all models")

# %%
# Print results
print("\n" + "="*70)
print(f"{'Model':<40} {'Final':>10} {'Fitted':>10} {'Epoch':>10} {'Source':<8}")
print("="*70)

print("\nUser models:")
for model in sorted(user_models):
    if model in eci_final:
        final = eci_final[model]
        fitted = eci_fitted.get(model)
        epoch_val = epoch_eci.get(model)
        source = "epoch" if (PRIORITIZE_EPOCH and epoch_val) else "fitted"
        fitted_str = f"{fitted:>10.1f}" if fitted else f"{'--':>10}"
        epoch_str = f"{epoch_val:>10.1f}" if epoch_val else f"{'--':>10}"
        print(f"  {model:<38} {final:>10.1f} {fitted_str} {epoch_str} {source:<8}")

# For full mode, also show sample of Epoch models
if MODE == "full":
    print("\nEpoch models (sample):")
    epoch_only = [m for m in eci_final.keys() if m not in user_models and m in epoch_eci]
    epoch_sorted = sorted(epoch_only, key=lambda m: eci_final[m], reverse=True)

    n = len(epoch_sorted)
    if n > 0:
        sample_idx = [0, 1, 2, n//4, n//2, 3*n//4, n-3, n-2, n-1]
        for i in sorted(set(sample_idx)):
            if 0 <= i < n:
                model = epoch_sorted[i]
                final = eci_final[model]
                epoch_val = epoch_eci[model]
                diff = final - epoch_val
                print(f"  {model:<38} {final:>10.1f} {epoch_val:>10.1f} {diff:>+8.1f}")

# %%
# Save results
OUTPUT_DIR.mkdir(exist_ok=True)

cm_df = pd.DataFrame([
    {
        "model": m,
        "eci_fitted": eci_final[m],  # This is the value to use (may be from Epoch if PRIORITIZE_EPOCH)
        "eci_our_fit": eci_fitted.get(m),  # Our fitted value
        "eci_epoch": epoch_eci.get(m),  # Epoch's published value
        "source": "epoch" if (PRIORITIZE_EPOCH and m in epoch_eci) else "fitted",
    }
    for m in sorted(eci_final.keys(), key=lambda x: -eci_final[x])
])
cm_df.to_csv(OUTPUT_DIR / "eci_model_capabilities.csv", index=False)

print(f"\nSaved to {OUTPUT_DIR / 'eci_model_capabilities.csv'}")

# %%
