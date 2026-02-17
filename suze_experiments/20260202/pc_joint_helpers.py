"""PC-based joint scaling fitters for 20260202 experiments.

These utilities fit a joint sigmoid scaling law where the "capability" term is a
linear combination of principal component (PC) scores:

  S(model) = alpha · PC(model)

and the joint model is:

  y = L + (1-L) * sigmoid(S + beta*g(h) + gamma*S*g(h) + delta)

where g(h) is a hint transform (identity/logit/learned piecewise linear, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class PCJointFit:
    n_pcs: int
    alpha: np.ndarray  # (n_pcs,)
    beta: float
    gamma: float | None
    delta: float
    include_cross: bool
    lower: float | None
    rms: float

    def capability(self, pc_scores: np.ndarray) -> float:
        return float(np.dot(self.alpha, pc_scores[: self.n_pcs]))

    def predict(self, pc_scores: np.ndarray, hint: float, hint_transform: Callable[[float], float]) -> float:
        L = float(self.lower) if self.lower is not None else 0.0
        U_minus_L = 1.0 - L
        s = self.capability(pc_scores)
        h_t = float(hint_transform(float(hint)))
        if self.include_cross and self.gamma is not None:
            z = s + self.beta * h_t + float(self.gamma) * s * h_t + self.delta
        else:
            z = s + self.beta * h_t + self.delta
        return float(L + U_minus_L * (1.0 / (1.0 + math.exp(-z))))


def fit_joint_sigmoid_over_pcs(
    *,
    df: pd.DataFrame,
    pc_scores_map: dict[str, np.ndarray],
    n_pcs: int,
    fit_models: set[str] | None,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
    alpha_fixed: np.ndarray | None = None,
) -> PCJointFit:
    """Fit a joint sigmoid where capability is alpha·PC (alpha fitted unless provided).

    Args:
      df: DataFrame with columns: model, hint, accuracy
      pc_scores_map: {model -> np.ndarray of PC scores}
      n_pcs: number of PCs to use
      fit_models: subset of models to fit on (train set)
      include_cross: include gamma*S*g(h) term
      hint_transform: g(h)
      lower: lower asymptote L (None means L=0)
      alpha_fixed: optional fixed alpha vector (len n_pcs). If set, only fit beta/gamma/delta.
    """
    from scipy.optimize import curve_fit

    n_pcs = int(n_pcs)
    if n_pcs <= 0:
        raise ValueError(f"n_pcs must be >= 1, got {n_pcs}")

    fit_df = df[df["model"].isin(fit_models)].copy() if fit_models is not None else df.copy()
    if len(fit_df) == 0:
        raise ValueError("No rows to fit after filtering fit_models")

    # Build arrays
    P_list: list[np.ndarray] = []
    H_list: list[float] = []
    y_list: list[float] = []

    for _, r in fit_df.iterrows():
        m = str(r["model"])
        pcs = pc_scores_map.get(m)
        if pcs is None or len(pcs) < n_pcs:
            continue
        P_list.append(np.asarray(pcs[:n_pcs], dtype=float))
        H_list.append(float(hint_transform(float(r["hint"]))))
        y_list.append(float(r["accuracy"]))

    if len(P_list) < max(10, n_pcs + (3 if include_cross else 2)):
        raise ValueError(f"Not enough fit rows: {len(P_list)} for n_pcs={n_pcs}")

    P = np.stack(P_list, axis=0)  # (N, n_pcs)
    H = np.asarray(H_list, dtype=float)  # (N,)
    y = np.asarray(y_list, dtype=float)  # (N,)

    L = float(lower) if lower is not None else 0.0
    U_minus_L = 1.0 - L

    h_range = float(max(H.max() - H.min(), 1e-6))
    beta0 = 4.0 / h_range if h_range > 1e-3 else 1.0
    gamma0 = 0.0
    delta0 = 0.0

    if alpha_fixed is None:
        alpha0 = np.zeros(n_pcs, dtype=float)
        X = np.vstack([P.T, H.reshape(1, -1)])  # (n_pcs+1, N)

        def model(X_local, *theta):
            P_local = X_local[:n_pcs, :].T  # (N, n_pcs)
            H_local = X_local[n_pcs, :]  # (N,)
            if include_cross:
                alpha = np.asarray(theta[:n_pcs], dtype=float)
                beta = float(theta[n_pcs])
                gamma = float(theta[n_pcs + 1])
                delta = float(theta[n_pcs + 2])
                S = P_local @ alpha
                z = S + beta * H_local + gamma * S * H_local + delta
            else:
                alpha = np.asarray(theta[:n_pcs], dtype=float)
                beta = float(theta[n_pcs])
                delta = float(theta[n_pcs + 1])
                S = P_local @ alpha
                z = S + beta * H_local + delta
            return L + U_minus_L * _sigmoid(z)

        p0 = np.concatenate([alpha0, np.array([beta0, gamma0, delta0] if include_cross else [beta0, delta0])])
        params, _ = curve_fit(model, X, y, p0=p0, maxfev=20000)
        params = np.asarray(params, dtype=float)
        y_pred = model(X, *params)
        rms = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        alpha_hat = params[:n_pcs].astype(float)
        if include_cross:
            beta_hat, gamma_hat, delta_hat = [float(x) for x in params[n_pcs : n_pcs + 3]]
        else:
            beta_hat, delta_hat = [float(x) for x in params[n_pcs : n_pcs + 2]]
            gamma_hat = None
    else:
        alpha_hat = np.asarray(alpha_fixed, dtype=float).reshape(-1)
        if alpha_hat.shape[0] != n_pcs:
            raise ValueError(f"alpha_fixed must have length n_pcs={n_pcs}, got {alpha_hat.shape[0]}")

        S = P @ alpha_hat  # (N,)
        X = np.vstack([S.reshape(1, -1), H.reshape(1, -1)])  # (2, N)

        def model(X_local, *theta):
            S_local = X_local[0, :]
            H_local = X_local[1, :]
            if include_cross:
                beta, gamma, delta = [float(t) for t in theta]
                z = S_local + beta * H_local + gamma * S_local * H_local + delta
            else:
                beta, delta = [float(t) for t in theta]
                z = S_local + beta * H_local + delta
            return L + U_minus_L * _sigmoid(z)

        p0 = np.array([beta0, gamma0, delta0], dtype=float) if include_cross else np.array([beta0, delta0], dtype=float)
        params, _ = curve_fit(model, X, y, p0=p0, maxfev=20000)
        params = np.asarray(params, dtype=float)
        y_pred = model(X, *params)
        rms = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        if include_cross:
            beta_hat, gamma_hat, delta_hat = [float(x) for x in params]
        else:
            beta_hat, delta_hat = [float(x) for x in params]
            gamma_hat = None

    return PCJointFit(
        n_pcs=n_pcs,
        alpha=alpha_hat,
        beta=float(beta_hat),
        gamma=(float(gamma_hat) if gamma_hat is not None else None),
        delta=float(delta_hat),
        include_cross=bool(include_cross),
        lower=lower,
        rms=float(rms),
    )


def capability_map_from_alpha(
    *,
    pc_scores_map: dict[str, np.ndarray],
    alpha: np.ndarray,
    n_pcs: int,
) -> dict[str, float]:
    n_pcs = int(n_pcs)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha.shape[0] < n_pcs:
        raise ValueError(f"alpha has length {alpha.shape[0]} < n_pcs={n_pcs}")
    out: dict[str, float] = {}
    for m, pcs in pc_scores_map.items():
        pcs = np.asarray(pcs, dtype=float)
        if pcs.shape[0] < n_pcs:
            continue
        out[str(m)] = float(np.dot(alpha[:n_pcs], pcs[:n_pcs]))
    return out

