# Epoch Capabilities Index (ECI)

Summary of "A Rosetta Stone for AI Benchmarks" (Ho et al., 2025) and how to construct an ECI-style x-axis for model comparisons.

## Core Idea

The ECI stitches together multiple benchmarks into a single capability scale by fitting a statistical model that relates:
- **Model capabilities** (Cm) - a single number per model
- **Benchmark difficulties** (Db) - a single number per benchmark
- **Benchmark slopes** (αb) - how steeply scores change with capability

## The Statistical Model

The core equation (similar to Item Response Theory):

```
score(m, b) = σ(αb × (Cm - Db))
```

Where:
- `score(m, b)` = benchmark score (0 to 1)
- `σ` = sigmoid function: 1 / (1 + exp(-x))
- `Cm` = model capability
- `Db` = benchmark difficulty
- `αb` = benchmark slope (spread in task difficulties)

**Interpretation:**
- When `Cm - Db = 0`, score = 50%
- Higher Cm → higher scores
- Higher Db → lower scores (harder benchmark)
- Higher αb → sharper transition from low to high scores

## Fitting the Model

### Data Requirements

Need a matrix of (model, benchmark, score) tuples with sufficient overlap:
- Models should be evaluated on multiple benchmarks (they filter to ≥3)
- Benchmarks should have multiple models evaluated
- They use 179 models, 38 benchmarks, 1324 scores

### Optimization

```python
from scipy.optimize import least_squares

# Initialize
Cm = {model: 0 for model in models}
Db = {bench: 0 for bench in benchmarks}
αb = {bench: 1 for bench in benchmarks}

# Fit using least squares with L2 regularization (λ=0.1)
# Minimize: Σ (score_predicted - score_actual)² + λ × (Σ Cm² + Σ Db² + Σ αb²)
```

### Identifiability

The model has two degrees of freedom that need to be fixed:

1. **Multiplicative rescale**: {αb, Cm, Db} and {k×αb, Cm/k, Db/k} give same predictions
2. **Additive shift**: {Cm, Db} and {Cm+δ, Db+δ} give same predictions

**Solution**: Pick an anchor benchmark (e.g., WinoGrande) and fix:
- `α_anchor = 1`
- `D_anchor = 0`

## Interpreting Capability Scores

### Option 1: Relative to Model Pairs

Express capabilities as "GPT-4-to-GPT-5 jumps":
- GPT-4 ≈ 1.6, GPT-5-high ≈ 2.6 → gap ≈ 1.0
- A model at 2.1 is "half a GPT-4-to-5 jump" above GPT-4

### Option 2: Map to Time Horizons

Using METR's time horizon data (how long humans take on tasks AIs can do):

```
time_horizon = exp(3.69 × Cm - 4.58)
```

This gives R² ≈ 0.85 correlation with actual time horizon data.

### Capability Progression

From the paper:
- Frontier capabilities increase ~0.55 units/year
- This corresponds to time horizon doubling every ~5 months

## Implementation for Our Use Case

### Option A: Use Epoch's Data Directly

Their data is available at: https://github.com/epoch-research/benchmark-stitching

We could:
1. Download their fitted Cm values for models
2. Map our model names to theirs
3. Use Cm as x-axis instead of model size

### Option B: Fit Our Own Model

If we have multiple benchmarks with overlapping models:

```python
import numpy as np
from scipy.optimize import least_squares

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def fit_eci(scores_df, anchor_benchmark="WinoGrande"):
    """
    scores_df: DataFrame with columns [model, benchmark, score]
    Returns: dict of model capabilities, dict of benchmark difficulties
    """
    models = scores_df["model"].unique()
    benchmarks = scores_df["benchmark"].unique()

    # Parameter indices
    n_models = len(models)
    n_benchmarks = len(benchmarks)
    model_idx = {m: i for i, m in enumerate(models)}
    bench_idx = {b: i for i, b in enumerate(benchmarks)}

    def residuals(params):
        # params: [Cm..., Db..., αb...]
        Cm = params[:n_models]
        Db = params[n_models:n_models + n_benchmarks]
        αb = params[n_models + n_benchmarks:]

        resid = []
        for _, row in scores_df.iterrows():
            m_i = model_idx[row["model"]]
            b_i = bench_idx[row["benchmark"]]
            pred = sigmoid(αb[b_i] * (Cm[m_i] - Db[b_i]))
            resid.append(pred - row["score"])

        # L2 regularization
        reg = 0.1 * np.concatenate([Cm, Db, αb - 1])
        return np.concatenate([resid, reg])

    # Initialize
    x0 = np.concatenate([
        np.zeros(n_models),      # Cm
        np.zeros(n_benchmarks),  # Db
        np.ones(n_benchmarks),   # αb
    ])

    result = least_squares(residuals, x0, method='trf')

    Cm = dict(zip(models, result.x[:n_models]))
    Db = dict(zip(benchmarks, result.x[n_models:n_models + n_benchmarks]))

    # Normalize: shift so anchor benchmark has Db=0
    if anchor_benchmark in Db:
        shift = Db[anchor_benchmark]
        Cm = {m: c - shift for m, c in Cm.items()}
        Db = {b: d - shift for b, d in Db.items()}

    return Cm, Db
```

### Option C: Simplified Single-Benchmark Approach

If using only one benchmark, this reduces to:
```
Cm = σ⁻¹(score) = logit(score) = log(score / (1 - score))
```

This is essentially what the `hint_transform=logit` does in our plotting code.

## Integration with modelx

To add ECI as an x-axis option:

```python
# In src/modelx/eci.py

def eci_capability(model: str) -> float:
    """Get ECI capability score for a model.

    Could be:
    1. Lookup from Epoch's data
    2. Computed from our own benchmark data
    3. Estimated from model size (as fallback)
    """
    # Option 1: Use precomputed values
    ECI_SCORES = {
        "Qwen3-0.6B": 0.8,
        "Qwen3-8B": 1.5,
        # ... etc
    }
    return ECI_SCORES.get(model, size(model))  # fallback to size
```

## Key Findings from the Paper

1. **Progress rate**: ~0.55 capability units/year at the frontier
2. **Algorithmic efficiency**: ~6× compute reduction per year (wide uncertainty: 1-50×)
3. **Model specialization**: Some models optimized for different tasks (Claude → code, Gemini → multimodal)
4. **Acceleration detection**: Can detect 2× accelerations within 2-3 months

## Limitations

1. Assumes capabilities are 1-dimensional (not true in practice)
2. Sensitive to benchmark selection
3. Newer/harder benchmarks may have overestimated difficulty (sparse data at high capability)
4. Does not account for evaluation setup differences (prompts, scaffolds, etc.)

## Our Implementation Notes (eci.py)

### Anchoring Approach

Epoch supports two anchoring modes:
1. **Benchmark anchoring**: Fix one benchmark's D=0, α=1 (raw scale), then shift after fitting
2. **Model anchoring**: Fix two models' capabilities directly (e.g., Claude 3.5 Sonnet = 130, GPT-5 = 150)

We use **model anchoring** since it directly produces ECI values on Epoch's published scale.

### Key Implementation Details

From Epoch's `fit_statistical_model()`:
- Random initialization: `randn() * 0.1` for C and D
- Bounds: C, D in [-10, 10], α in [0.1, 10] (for raw scale fitting)
- Regularization: Single penalty term `sqrt(reg_strength * (sum(C²) + sum(D²) + sum(α²)) / n_params)`

### Critical Fix: Regularization with Model Anchoring

**Problem**: When using model anchoring with C values at ECI scale (~130-150), regularizing C² toward 0 causes severe underfitting. The regularization penalty (C² ~ 17000) dominates the loss.

**Solution**: Set `reg_strength=0` when using model anchoring on ECI scale. The anchors provide sufficient constraint to avoid overfitting.

**Results with fix**:
- RMSE: 0.075 (vs 0.13 with regularization)
- Model ECI values match Epoch's within ±0.5 points
- Winogrande predictions improved from errors of -0.3 to -0.4 down to -0.06 to -0.08

### Data Sources

1. **Epoch benchmark scores**: `benchmark_data/*.csv` (37 benchmarks, 282 models with ECI)
2. **User scores (auto)**: `src/modelx/model_scores.csv` - from baseline eval runs
3. **User scores (manual)**: `src/modelx/model_scores_manual.csv` - from system cards

Filtering: Only include Epoch's ECI models + user models to maintain scale consistency.

## References

- Paper: https://arxiv.org/abs/2512.00193
- Code: https://github.com/epoch-research/benchmark-stitching
- Live dashboard: Epoch Capabilities Index (continuously updated)
