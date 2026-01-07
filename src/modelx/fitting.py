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
            params, _ = curve_fit(model, x_transformed, y, p0=[1, 0], maxfev=10000)
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
        params, _ = curve_fit(model, x_transformed, y, p0=[1, 0], maxfev=10000)
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
) -> dict:
    """Fit joint sigmoid model: σ(α*C + β*H + γ*C*H + δ) or σ(α*C + β*H + δ)

    where C = log(model_size), H = hint (or transformed hint).

    Args:
        df: DataFrame with model, model_size, accuracy, hint columns
        x_col: Column for model size
        y_col: Column for y values (accuracy)
        hint_col: Column for hint values
        include_cross: If True, include γ*C*H term
        hint_transform: Function to transform hint values
        exclude_hints: Set of hint values to exclude from fitting
        fit_models: Set of model names to include (None = all)

    Returns:
        Dict with keys:
        - 'params': array of fitted parameters [α, β, γ, δ] or [α, β, δ]
        - 'rms': RMS error
        - 'include_cross': whether cross term was included
        - 'predict': function(x, hint) -> y
    """
    if hint_transform is None:
        hint_transform = lambda h: h
    if exclude_hints is None:
        exclude_hints = set()

    C_all, H_all, y_all = [], [], []

    for _, row in df.iterrows():
        if row[hint_col] in exclude_hints:
            continue
        if fit_models is not None and row["model"] not in fit_models:
            continue
        if pd.isna(row[y_col]):
            continue

        C_all.append(np.log(row[x_col]))
        H_all.append(hint_transform(row[hint_col]))
        y_all.append(row[y_col])

    min_points = 4 if include_cross else 3
    if len(C_all) < min_points:
        raise ValueError(f"Not enough data points for joint fit: {len(C_all)}")

    CH = np.array([C_all, H_all])
    y_arr = np.array(y_all)

    if include_cross:
        def model(CH, alpha, beta, gamma, delta):
            C, H = CH
            return 1 / (1 + np.exp(-(alpha * C + beta * H + gamma * C * H + delta)))
        params, _ = curve_fit(model, CH, y_arr, p0=[1, 1, 0, 0], maxfev=10000)
        y_pred = model(CH, *params)
        def predict(x, hint):
            C = np.log(x)
            H = hint_transform(hint)
            return 1 / (1 + np.exp(-(params[0] * C + params[1] * H + params[2] * C * H + params[3])))
    else:
        def model(CH, alpha, beta, delta):
            C, H = CH
            return 1 / (1 + np.exp(-(alpha * C + beta * H + delta)))
        params, _ = curve_fit(model, CH, y_arr, p0=[1, 1, 0], maxfev=10000)
        y_pred = model(CH, *params)
        def predict(x, hint):
            C = np.log(x)
            H = hint_transform(hint)
            return 1 / (1 + np.exp(-(params[0] * C + params[1] * H + params[2])))

    rms = np.sqrt(np.mean((y_arr - y_pred) ** 2))

    return {
        "params": params,
        "rms": rms,
        "include_cross": include_cross,
        "predict": predict,
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
        if fit_result["include_cross"]:
            alpha, beta, gamma, delta = params
            return f"σ({alpha:.2f}C {beta:+.2f}H {gamma:+.2f}CH {delta:+.2f})"
        else:
            alpha, beta, delta = params
            return f"σ({alpha:.2f}C {beta:+.2f}H {delta:+.2f})"

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
