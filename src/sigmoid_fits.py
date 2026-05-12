from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid_curve(
    x: np.ndarray,
    *,
    lower: float,
    slope: float,
    bias: float,
) -> np.ndarray:
    return lower + (1.0 - lower) * (1.0 / (1.0 + np.exp(-(slope * x + bias))))


def fit_plot_sigmoid(
    xs: list[float],
    ys: list[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(xs) < 4:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.allclose(y, y[0]):
        return None

    try:
        from scipy.optimize import curve_fit

        lower0 = float(np.clip(np.min(y) - 0.02, 0.0, 0.95))
        y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
        mid_idx = int(np.argmin(np.abs(y - y_mid)))
        x_mid = float(x[mid_idx])
        slope0 = 0.2
        bias0 = -slope0 * x_mid

        def fn(x_input: np.ndarray, lower: float, slope: float, bias: float) -> np.ndarray:
            return sigmoid_curve(
                np.asarray(x_input, dtype=float),
                lower=float(lower),
                slope=float(slope),
                bias=float(bias),
            )

        params, _ = curve_fit(
            fn,
            x,
            y,
            p0=[lower0, slope0, bias0],
            bounds=([0.0, 1e-6, -200.0], [0.99, 10.0, 200.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    x_fit = np.linspace(float(np.min(x)) - 2.0, float(np.max(x)) + 2.0, 200, dtype=float)
    y_fit = fn(x_fit, *params)
    return x_fit, y_fit


def clip_accuracy_for_logit(
    y: np.ndarray,
    *,
    lower: float | None,
) -> np.ndarray:
    lower_value = 0.0 if lower is None else float(lower)
    scaled = (np.asarray(y, dtype=float) - lower_value) / max(1.0 - lower_value, 1e-8)
    return np.clip(scaled, 1e-4, 1.0 - 1e-4)


def fit_bounded_sigmoid_1d(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    lower: float | None,
) -> dict[str, Any] | None:
    from scipy.optimize import curve_fit

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or np.allclose(y, y[0]):
        return None

    clipped = clip_accuracy_for_logit(y, lower=lower)
    target = np.log(clipped / (1.0 - clipped))
    slope0, bias0 = np.linalg.lstsq(
        np.column_stack([x, np.ones_like(x)]),
        target,
        rcond=None,
    )[0]
    if not np.isfinite(slope0):
        slope0 = 0.1
    if not np.isfinite(bias0):
        bias0 = -np.mean(x) * slope0

    lower_value = 0.0 if lower is None else float(lower)

    def fn(x_input: np.ndarray, slope: float, bias: float) -> np.ndarray:
        return sigmoid_curve(
            np.asarray(x_input, dtype=float),
            lower=lower_value,
            slope=float(slope),
            bias=float(bias),
        )

    try:
        params, _ = curve_fit(
            fn,
            x,
            y,
            p0=[float(slope0), float(bias0)],
            bounds=([-20.0, -500.0], [20.0, 500.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    slope, bias = [float(value) for value in params]
    if abs(slope) <= 1e-8:
        return None

    def predict(x_input: float) -> float:
        return float(fn(np.asarray([x_input], dtype=float), slope, bias)[0])

    return {
        "params": [slope, bias],
        "midpoint": float(-bias / slope),
        "predict": predict,
    }
