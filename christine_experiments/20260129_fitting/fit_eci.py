# %%
"""Fit ECI from baseline data + Epoch benchmark scores."""

import json
import sys
sys.path.append("/Users/christineye/emergent-doordash")

import pandas as pd
import numpy as np
from pathlib import Path

from src.modelx import (
    load_baseline,
    load_epoch_benchmark_scores,
    load_epoch_params,
    fit_eci,
)

%load_ext autoreload
%autoreload 2

# %%
# Configuration
BASELINE_FOLDER = "/Users/christineye/emergent-doordash/christine_experiments/20251113/baseline"
OUTPUT_DIR = Path("/Users/christineye/emergent-doordash/christine_experiments/20260129_fitting")
EPOCH_ECI_FILE = "/Users/christineye/emergent-doordash/src/modelx/eci/eci_scores.json"

# Two anchor models for rescaling (must be in Epoch's data)
ANCHOR1 = ("claude-3-5-sonnet-20240620", 130.0)
ANCHOR2 = ("gpt-5-2025-08-07_medium", 150.0)

# Mapping from baseline eval names to ECI benchmark names
EVAL_TO_ECI = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "mmlu_5_shot_cot": "MMLU",
    "math_level_5": "MATH level 5",
    # "gpqa": "GPQA diamond",  # Excluded: causes fitting issues for small models
}

# %%
# Load Epoch's pre-computed ECI scores for comparison
with open(EPOCH_ECI_FILE) as f:
    epoch_eci = json.load(f)
print(f"Loaded {len(epoch_eci)} Epoch ECI scores for comparison")

# %%
# Load user baseline scores
print("\nLoading baseline results...")
user_rows = []

for eval_name, eci_benchmark in EVAL_TO_ECI.items():
    df = load_baseline(BASELINE_FOLDER, eval_name)
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
user_models = user_scores["model"].unique()
print(f"\nTotal user scores: {len(user_scores)}")
print(f"User models: {len(user_models)}")

# %%
# Load Epoch benchmark scores and combine
print("\nLoading Epoch benchmark scores...")
epoch_scores = load_epoch_benchmark_scores(only_eci_models=True)
print(f"Epoch scores: {len(epoch_scores)} ({epoch_scores['model'].nunique()} models)")

combined = pd.concat([epoch_scores, user_scores], ignore_index=True)
combined = combined.drop_duplicates(subset=["model", "benchmark"], keep="last")
print(f"Combined: {len(combined)} scores, {combined['model'].nunique()} models")

# %%
# Fit ECI
print("\nFitting ECI...")
result = fit_eci(
    combined,
    anchor_benchmark="Winogrande",
    reg_strength=0.1,
    min_benchmarks=3,
)
print(f"RMSE: {result['rmse']:.4f}")

# %%
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
print(f"\nTransform: ECI = {scale:.4f} * raw + {offset:.2f}")

rescaled = {m: scale * c + offset for m, c in result["Cm"].items()}

# %%
# Debug: Compare fitted benchmark params to Epoch's
print("\nBenchmark parameters (fitted vs Epoch):")
epoch_params = load_epoch_params()
print(f"{'Benchmark':<30} {'D_fit':>8} {'D_epoch':>8} {'α_fit':>8} {'α_epoch':>8}")
for bench in sorted(result["Db"].keys()):
    d_fit = result["Db"][bench]
    a_fit = result["ab"][bench]
    d_epoch = epoch_params["difficulty"].get(bench, None)
    a_epoch = epoch_params["slope"].get(bench, None)
    if d_epoch:
        print(f"  {bench:<28} {d_fit:>8.2f} {d_epoch:>8.2f} {a_fit:>8.4f} {a_epoch:>8.4f}")
    else:
        print(f"  {bench:<28} {d_fit:>8.2f} {'N/A':>8} {a_fit:>8.4f} {'N/A':>8}")

# %%
# Debug: Check raw values before rescaling for small models
print("\nDebug: Raw values for small models:")
small_models = ["Qwen2.5-0.5B-Instruct", "Qwen3-0.6B", "gemma-3-1b-it"]
for model in small_models:
    if model in result["Cm"]:
        raw = result["Cm"][model]
        # What benchmarks does this model have?
        model_scores = combined[combined["model"] == model]
        print(f"\n  {model}: raw={raw:.2f}")
        for _, row in model_scores.iterrows():
            bench = row["benchmark"]
            score = row["score"]
            d = result["Db"].get(bench, 0)
            a = result["ab"].get(bench, 1)
            # What capability would this single score imply?
            # score = sigmoid(a * (C - D)) => C = D + logit(score)/a
            score_clipped = np.clip(score, 0.01, 0.99)
            implied_c = d + np.log(score_clipped / (1 - score_clipped)) / a
            print(f"    {bench:<20} score={score:.3f} -> implied C={implied_c:.1f} (D={d:.1f}, α={a:.3f})")

# %%
# Debug: Predictions for small models
print("\nDebug: Predictions for small models:")
pred_df = result["predictions"]
for model in small_models:
    if model in result["Cm"]:
        model_preds = pred_df[pred_df["model"] == model]
        print(f"\n  {model} (raw Cm = {result['Cm'][model]:.2f}):")
        for _, row in model_preds.iterrows():
            print(f"    {row['benchmark']:<20} actual={row['score']:.3f} pred={row['predicted']:.3f} err={row['error']:+.3f}")

# %%
# Print results with comparison to Epoch's values
print("\n" + "="*80)
print(f"{'Model':<40} {'Raw':>8} {'Fitted':>10} {'Epoch':>10} {'Diff':>10}")
print("="*80)

# User models first
print("\nUser models:")
for model in sorted(user_models):
    if model in rescaled:
        fitted = rescaled[model]
        epoch_val = epoch_eci.get(model)
        if epoch_val:
            diff = fitted - epoch_val
            print(f"  {model:<38} {fitted:>10.1f} {epoch_val:>10.1f} {diff:>+10.1f}")
        else:
            print(f"  {model:<38} {fitted:>10.1f} {'N/A':>10} {'':>10}")

# Sample of Epoch models for validation
print("\nEpoch models (sample):")
epoch_only = [m for m in rescaled.keys() if m not in user_models and m in epoch_eci]
epoch_sorted = sorted(epoch_only, key=lambda m: rescaled[m], reverse=True)

# Show top, middle, bottom
n = len(epoch_sorted)
sample_idx = [0, 1, 2, n//4, n//2, 3*n//4, n-3, n-2, n-1]
for i in set(sample_idx):
    if 0 <= i < n:
        model = epoch_sorted[i]
        fitted = rescaled[model]
        epoch_val = epoch_eci[model]
        diff = fitted - epoch_val
        print(f"  {model:<38} {fitted:>10.1f} {epoch_val:>10.1f} {diff:>+10.1f}")

# %%
# Save results
OUTPUT_DIR.mkdir(exist_ok=True)

cm_df = pd.DataFrame([
    {"model": m, "Cm_fitted": rescaled[m], "Cm_epoch": epoch_eci.get(m)}
    for m in sorted(rescaled.keys(), key=lambda x: -rescaled[x])
])
cm_df.to_csv(OUTPUT_DIR / "eci_model_capabilities.csv", index=False)

print(f"\nSaved to {OUTPUT_DIR / 'eci_model_capabilities.csv'}")
