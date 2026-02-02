# Joint Scaling Law Fitting

This document describes the procedure for fitting a joint scaling law that models how model accuracy depends on both **capability** (ECI) and **hint fraction**.

## Overview

We want to predict model accuracy on a task as a function of:
- **C**: Model capability, measured by ECI (Epoch Capabilities Index)
- **h**: Hint fraction, the proportion of the solution revealed to the model

## Step 1: Estimating Model Capability (ECI)

### Background

The Epoch Capabilities Index (ECI) is a unified capability score derived from performance across multiple benchmarks. The core idea is that benchmark performance follows a sigmoid in capability:

$$
\text{score}(m, b) = \sigma(\alpha_b \cdot (C_m - D_b))
$$

where:
- $C_m$ = capability of model $m$
- $D_b$ = difficulty of benchmark $b$
- $\alpha_b$ = slope (discrimination) of benchmark $b$
- $\sigma(x) = 1 / (1 + e^{-x})$ is the sigmoid function

### Our Approach: Using Epoch's Pre-fitted Parameters

Rather than re-fitting all benchmark parameters, we use Epoch's published difficulty ($D_b$) and slope ($\alpha_b$) values. For each model, we find the capability $C_m$ that best explains its benchmark scores:

$$
C_m = \arg\min_C \sum_b \left( \text{score}_{m,b} - \sigma(\alpha_b \cdot (C - D_b)) \right)^2
$$

This gives us a single capability number per model on the ECI scale (typically 80-150 range).

## Step 2: Joint Scaling Law Model

### Model Form

We fit a joint sigmoid model that captures how accuracy depends on both capability and hint:

$$
\varepsilon(C, h) = L + (1 - L) \cdot \sigma(\alpha C + \beta h + \gamma C h + \delta)
$$

where:
- $\varepsilon$ = predicted accuracy
- $C$ = model capability (ECI)
- $h$ = hint fraction (0 to 1)
- $L$ = lower asymptote (e.g., 0.2 for random baseline on 5-choice task)
- $\alpha$ = capability coefficient
- $\beta$ = hint coefficient
- $\gamma$ = interaction coefficient (capability × hint)
- $\delta$ = offset

### Interpretation

- **Without cross term** ($\gamma = 0$): Capability and hint contribute independently
- **With cross term** ($\gamma \neq 0$): The effect of hints depends on capability (or vice versa)
  - $\gamma > 0$: Hints help more capable models more
  - $\gamma < 0$: Hints help less capable models more

### Lower Asymptote

The lower asymptote $L$ represents the floor accuracy (e.g., random guessing). Setting $L = 0.2$ for a 5-choice task ensures predictions stay in $[0.2, 1.0]$.

## Step 3: Fitting Procedure

### Data

For each (model, hint) pair, we have an accuracy measurement. The data is structured as:
- Model → ECI (from Step 1)
- Hint fraction → $h \in \{0.0, 0.05, 0.1, ..., 0.95, 1.0\}$
- Accuracy → measured performance

### Optimization

We fit parameters $(\alpha, \beta, \gamma, \delta)$ using nonlinear least squares (scipy's `curve_fit`):

$$
\min_{\alpha, \beta, \gamma, \delta} \sum_{m, h} \left( \varepsilon_{\text{actual}}(m, h) - \varepsilon_{\text{pred}}(C_m, h) \right)^2
$$

### Train/Test Split

Models can be split into train and test sets:
- **Train models**: Used to fit the joint scaling law
- **Test models**: Held out to evaluate generalization

## Step 4: Evaluation Metrics

### RMS Error

Root mean squared error between predicted and actual accuracy:

$$
\text{RMS} = \sqrt{\frac{1}{N} \sum_{m, h} \left( \varepsilon_{\text{actual}}(m, h) - \varepsilon_{\text{pred}}(C_m, h) \right)^2}
$$

Computed separately for train, test, and all models.

### Midpoint Error

For each hint level $h$, we can fit an individual sigmoid in capability:

$$
\varepsilon_h(C) = L + (1 - L) \cdot \sigma(\alpha_h C + \beta_h)
$$

The **midpoint** is the capability where accuracy = 50%:

$$
C_{50}^{(h)} = -\beta_h / \alpha_h
$$

The joint fit also implies a midpoint at each hint level. Solving $\varepsilon(C, h) = 0.5$:

$$
C_{50}^{\text{joint}}(h) = \frac{-\beta h - \delta}{\alpha + \gamma h}
$$

**Midpoint error** is the absolute difference between joint and individual midpoints:

$$
\text{Midpoint Error}(h) = \left| C_{50}^{\text{joint}}(h) - C_{50}^{(h)} \right|
$$

This measures how well the joint model captures the "threshold capability" for each hint level.

## Step 5: Model Sweep Analysis

To understand how many models are needed for a good fit, we sweep over the number of training models:

1. Sort all models by ECI (lowest to highest)
2. For $n = 5, 6, ..., N$:
   - Use the $n$ lowest-ECI models as training set
   - Fit joint scaling law
   - Compute RMS (train, test, all)
   - Compute midpoint errors at selected hint values

This reveals:
- How quickly the fit stabilizes
- Whether low-capability models alone can predict high-capability behavior
- The minimum number of models needed for reliable extrapolation

## File Structure

- `fit_eci.py`: Estimates ECI for models using Epoch's parameters
- `plots.py`: Main analysis notebook with joint scaling law fitting and visualization
- `eci_model_capabilities.csv`: Output ECI values per model

## Key Functions

### In `src/modelx/fitting.py`:
- `fit_sigmoid()`: Fit single-variable sigmoid
- `fit_joint_sigmoid()`: Fit joint 2D sigmoid with capability and hint

### In `plots.py`:
- `fit_individual_sigmoids_by_hint()`: Fit separate sigmoids for each hint level
- `compute_midpoint_errors()`: Compare joint vs individual midpoints
- `compute_rms()`: Compute RMS error for a joint fit
- `run_model_sweep()`: Sweep over number of training models

## References

- [Epoch AI: The Epoch AI Capabilities Index](https://epoch.ai/data/eci)
