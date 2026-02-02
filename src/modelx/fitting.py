"""Sigmoid fitting utilities for model scaling analysis."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Callable


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))


def fit_sigmoid(
    x: np.ndarray,
    y: np.ndarray,
    use_log: bool = False,
    scale: bool = False,
    lower: float | None = None,
    upper: float | None = None,
) -> dict:
    """Unified sigmoid fitting function.

    Fits various sigmoid models based on parameters:
    - Basic: y = σ(m*x + b)
    - Scaled: y = h * σ(m*x + b)
    - Asymptote: y = L + (U-L) * σ(m*x + b)
    - Scaled asymptote: y = L + h * σ(m*x + b)

    Args:
        x: Input values
        y: Target values
        use_log: If True, use log(x) instead of x
        scale: If True, fit scaling parameter h
        lower: Lower asymptote L (if not None, constrains the fit)
        upper: Upper asymptote U (used when lower is set and scale=False)

    Returns:
        Dict with keys:
        - 'type': fit type string ('basic', 'scaled', 'asymptote', 'scaled_asymptote')
        - 'params': tuple of fitted parameters
        - 'predict': function(x) -> y that evaluates the fitted model
    """
    x_transformed = np.log(x) if use_log else x

    if lower is not None:
        if scale:
            # y = L + h * σ(m*x + b)
            h_max = (upper if upper is not None else 1.0) - lower
            def model(x_t, h, m, b):
                return lower + h * sigmoid(m * x_t + b)
            params, _ = curve_fit(
                model, x_transformed, y,
                p0=[min(np.max(y) - lower, h_max), 1, 0],
                bounds=([0, -np.inf, -np.inf], [h_max, np.inf, np.inf]),
                maxfev=10000,
            )
            h, m, b = params
            fit_type = "scaled_asymptote"
            fit_params = (lower, h, m, b)
            def predict(x_new):
                x_t = np.log(x_new) if use_log else x_new
                return lower + h * sigmoid(m * x_t + b)
        else:
            # y = L + (U-L) * σ(m*x + b)
            U = upper if upper is not None else 1.0
            def model(x_t, m, b):
                return lower + (U - lower) * sigmoid(m * x_t + b)
            # Initial guess: center sigmoid on data
            x_mean = np.mean(x_transformed)
            m0 = 1.0 / max(np.std(x_transformed), 1e-6)
            b0 = -m0 * x_mean
            params, _ = curve_fit(model, x_transformed, y, p0=[m0, b0], maxfev=10000)
            m, b = params
            fit_type = "asymptote"
            fit_params = (lower, U, m, b)
            def predict(x_new):
                x_t = np.log(x_new) if use_log else x_new
                return lower + (U - lower) * sigmoid(m * x_t + b)
    elif scale:
        # y = h * σ(m*x + b)
        def model(x_t, h, m, b):
            return h * sigmoid(m * x_t + b)
        params, _ = curve_fit(
            model, x_transformed, y,
            p0=[np.max(y), 1, 0],
            bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        h, m, b = params
        fit_type = "scaled"
        fit_params = (h, m, b)
        def predict(x_new):
            x_t = np.log(x_new) if use_log else x_new
            return h * sigmoid(m * x_t + b)
    else:
        # y = σ(m*x + b)
        def model(x_t, m, b):
            return sigmoid(m * x_t + b)
        # Initial guess: center sigmoid on data (m*x_mean + b = 0 at midpoint)
        x_mean = np.mean(x_transformed)
        m0 = 1.0 / max(np.std(x_transformed), 1e-6)  # Scale by data spread
        b0 = -m0 * x_mean
        params, _ = curve_fit(model, x_transformed, y, p0=[m0, b0], maxfev=10000)
        m, b = params
        fit_type = "basic"
        fit_params = (m, b)
        def predict(x_new):
            x_t = np.log(x_new) if use_log else x_new
            return sigmoid(m * x_t + b)

    return {
        "type": fit_type,
        "params": fit_params,
        "predict": predict,
        "use_log": use_log,
    }


def fit_joint_sigmoid(
    df: pd.DataFrame,
    x_col: str = "model_size",
    y_col: str = "accuracy",
    hint_col: str = "hint",
    include_cross: bool = True,
    hint_transform: Callable[[float], float] | None = None,
    exclude_hints: set[float] | None = None,
    fit_models: set[str] | None = None,
    use_log_x: bool = True,
    x_values: dict[str, float] | Callable[[str], float | None] | None = None,
    lower: float | None = None,
) -> dict:
    """Fit joint sigmoid model: σ(α*C + β*H + γ*C*H + δ) or σ(α*C + β*H + δ)

    where C = log(x) or x (depending on use_log_x), H = hint (or transformed hint).

    If lower is set, fits: L + (1-L) * σ(...)

    Args:
        df: DataFrame with model, accuracy, hint columns (and optionally x_col)
        x_col: Column for x values (used if x_values is None)
        y_col: Column for y values (accuracy)
        hint_col: Column for hint values
        include_cross: If True, include γ*C*H term
        hint_transform: Function to transform hint values
        exclude_hints: Set of hint values to exclude from fitting
        fit_models: Set of model names to include (None = all)
        use_log_x: If True, use log(x) for fitting; if False, use x directly
        x_values: Optional dict {model: x_value} or function(model) -> x_value
                  If provided, uses this instead of x_col from dataframe
        lower: If set, pin lower asymptote to this value (e.g., 0.2 for random baseline)

    Returns:
        Dict with keys:
        - 'params': array of fitted parameters [α, β, γ, δ] or [α, β, δ]
        - 'rms': RMS error
        - 'include_cross': whether cross term was included
        - 'predict': function(x, hint) -> y
        - 'use_log_x': whether log(x) was used
        - 'lower': lower asymptote if set
    """
    if hint_transform is None:
        hint_transform = lambda h: h
    if exclude_hints is None:
        exclude_hints = set()

    # Helper to get x value for a model
    def get_x(row):
        if x_values is None:
            return row[x_col]
        elif callable(x_values):
            return x_values(row["model"])
        else:
            return x_values.get(row["model"])

    C_all, H_all, y_all = [], [], []

    for _, row in df.iterrows():
        if row[hint_col] in exclude_hints:
            continue
        if fit_models is not None and row["model"] not in fit_models:
            continue
        if pd.isna(row[y_col]):
            continue

        x_val = get_x(row)
        if x_val is None:
            continue

        c_val = np.log(x_val) if use_log_x else x_val
        C_all.append(c_val)
        H_all.append(hint_transform(row[hint_col]))
        y_all.append(row[y_col])

    min_points = 4 if include_cross else 3
    if len(C_all) < min_points:
        raise ValueError(f"Not enough data points for joint fit: {len(C_all)}")

    C_arr = np.array(C_all)
    H_arr = np.array(H_all)
    y_arr = np.array(y_all)
    CH = np.array([C_arr, H_arr])

    # Compute data-dependent initial guesses
    # α should scale so sigmoid transitions across C range: α ≈ 4 / range(C)
    # δ should center the sigmoid: δ ≈ -α * mean(C)
    c_range = max(C_arr.max() - C_arr.min(), 1e-6)
    c_mean = C_arr.mean()
    h_range = max(H_arr.max() - H_arr.min(), 1e-6)
    alpha_init = 4.0 / c_range
    beta_init = 4.0 / h_range if h_range > 0.1 else 1.0
    delta_init = -alpha_init * c_mean

    L = lower if lower is not None else 0.0
    U = 1.0

    if include_cross:
        def model(CH, alpha, beta, gamma, delta):
            C, H = CH
            sig = 1 / (1 + np.exp(-(alpha * C + beta * H + gamma * C * H + delta)))
            return L + (U - L) * sig
        p0 = [alpha_init, beta_init, 0, delta_init]
        params, _ = curve_fit(model, CH, y_arr, p0=p0, maxfev=10000)
        y_pred = model(CH, *params)
        def predict(x, hint):
            C = np.log(x) if use_log_x else x
            H = hint_transform(hint)
            sig = 1 / (1 + np.exp(-(params[0] * C + params[1] * H + params[2] * C * H + params[3])))
            return L + (U - L) * sig
    else:
        def model(CH, alpha, beta, delta):
            C, H = CH
            sig = 1 / (1 + np.exp(-(alpha * C + beta * H + delta)))
            return L + (U - L) * sig
        p0 = [alpha_init, beta_init, delta_init]
        params, _ = curve_fit(model, CH, y_arr, p0=p0, maxfev=10000)
        y_pred = model(CH, *params)
        def predict(x, hint):
            C = np.log(x) if use_log_x else x
            H = hint_transform(hint)
            sig = 1 / (1 + np.exp(-(params[0] * C + params[1] * H + params[2])))
            return L + (U - L) * sig

    rms = np.sqrt(np.mean((y_arr - y_pred) ** 2))

    return {
        "params": params,
        "rms": rms,
        "include_cross": include_cross,
        "predict": predict,
        "use_log_x": use_log_x,
        "lower": lower,
    }


def format_equation(fit_result: dict) -> str:
    """Format fit result as a readable equation string.

    Args:
        fit_result: Dict returned by fit_sigmoid or fit_joint_sigmoid

    Returns:
        Formatted equation string
    """
    if "include_cross" in fit_result:
        # Joint model
        params = fit_result["params"]
        lower = fit_result.get("lower")
        if fit_result["include_cross"]:
            alpha, beta, gamma, delta = params
            sig = f"σ({alpha:.2f}C {beta:+.2f}H {gamma:+.2f}CH {delta:+.2f})"
        else:
            alpha, beta, delta = params
            sig = f"σ({alpha:.2f}C {beta:+.2f}H {delta:+.2f})"
        if lower is not None:
            return f"{lower:.2f} + {1-lower:.2f}·{sig}"
        return sig

    # Single sigmoid
    fit_type = fit_result["type"]
    params = fit_result["params"]
    x_str = "log(x)" if fit_result.get("use_log", False) else "x"

    if fit_type == "basic":
        m, b = params
        return f"σ({m:.2f}·{x_str} {b:+.2f})"
    elif fit_type == "scaled":
        h, m, b = params
        return f"{h:.3f}·σ({m:.2f}·{x_str} {b:+.2f})"
    elif fit_type == "asymptote":
        L, U, m, b = params
        return f"{L:.3f} + {U-L:.3f}·σ({m:.2f}·{x_str} {b:+.2f})"
    elif fit_type == "scaled_asymptote":
        L, h, m, b = params
        return f"{L:.3f} + {h:.3f}·σ({m:.2f}·{x_str} {b:+.2f})"
    else:
        raise ValueError(f"Unknown fit type: {fit_type}")
