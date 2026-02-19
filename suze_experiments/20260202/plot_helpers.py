"""Plotting helpers for 20260202 experiments.

This module is intended to be imported from `experiments.py` and contains the
reusable plotting + fitting logic (as opposed to experiment-specific configs).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------ shared color palette ------------------------------
# Ordered list of colors for comparison plots (ECI first, then PC methods).
# Long enough that no two methods ever share a color; never use modulo indexing.
_COMPARISON_COLORS: list[str] = [
    "steelblue",     # ECI joint
    "darkorange",    # PC1
    "forestgreen",   # PC1+PC2
    "crimson",       # PC1+PC2+PC3
    "mediumpurple",  # PC1+PC2+PC3+PC4
    "saddlebrown",
    "teal",
    "gold",
]
_ECI_JOINT_COLOR = _COMPARISON_COLORS[0]
_PC_METHOD_COLORS = _COMPARISON_COLORS[1:]   # index 0 = PC1, 1 = PC1+PC2, …

# ------------------------------ hint transforms ------------------------------

def hint_identity(h: float) -> float:
    return h


def hint_logit(h: float) -> float:
    return math.log(h / (1.0 - h))


HINT_TRANSFORMS: dict[str, Callable[[float], float]] = {
    "identity": hint_identity,
    "logit": hint_logit,
}


@dataclass(frozen=True)
class PiecewiseLinearHintTransform:
    """Monotone piecewise-linear mapping from raw hint fraction to usefulness."""

    knot_x: np.ndarray  # shape (K,)
    knot_y: np.ndarray  # shape (K,)

    def __post_init__(self) -> None:
        x = np.asarray(self.knot_x, dtype=float)
        y = np.asarray(self.knot_y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("knot_x and knot_y must be 1D arrays of same length")
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 knots")
        if np.any(np.diff(x) <= 0):
            raise ValueError("knot_x must be strictly increasing")
        if np.min(x) < 0.0 or np.max(x) > 1.0:
            raise ValueError("knot_x must be within [0, 1]")
        if np.any(np.diff(y) < 0):
            raise ValueError("knot_y must be monotone non-decreasing")

    def __call__(self, h: float) -> float:
        x = np.asarray(self.knot_x, dtype=float)
        y = np.asarray(self.knot_y, dtype=float)
        h_clip = float(np.clip(h, x[0], x[-1]))
        return float(np.interp(h_clip, x, y))

    def vectorized(self, h: np.ndarray) -> np.ndarray:
        x = np.asarray(self.knot_x, dtype=float)
        y = np.asarray(self.knot_y, dtype=float)
        h_clip = np.clip(h.astype(float), x[0], x[-1])
        return np.interp(h_clip, x, y)


def plot_hint_transform_mapping(
    *,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    output_dir: Path,
    title: str,
) -> Path:
    """Plot raw hint fraction vs learned usefulness mapping."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(knot_x, knot_y, "-o", linewidth=2, markersize=5)
    ax.set_xlabel("raw hint fraction")
    ax.set_ylabel("learned hint usefulness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    y_min = float(np.min(knot_y))
    y_max = float(np.max(knot_y))
    pad = 0.05 * max(1e-6, y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)
    plt.tight_layout()

    out_path = output_dir / "hint_transform_mapping.png"
    save_figure(fig, out_path)
    return out_path


# ------------------------------ io helpers ------------------------------

def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save a matplotlib figure and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------ data loading ------------------------------

def load_eci_map(eci_file: Path) -> dict[str, float]:
    """Load per-model ECI values into a {model -> eci} dict."""
    eci_df = pd.read_csv(eci_file)
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def print_eci_table(
    *,
    all_models: list[str],
    eci_map: dict[str, float],
    train_models: set[str],
) -> None:
    print(f"Fitted ECIs for {len(all_models)} models:")
    for model in sorted(all_models, key=lambda m: eci_map.get(m, 0)):
        eci = eci_map.get(model)
        split = "TRAIN" if model in train_models else "TEST"
        if eci is not None and not (isinstance(eci, float) and np.isnan(eci)):
            print(f"  [{split:5s}] {model:35s} {eci:6.1f}")
        else:
            print(f"  [{split:5s}] {model:35s} MISSING")


def load_and_prepare_results_df(
    *,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
    all_models: list[str],
    hint_fractions: list[float],
    eci_map: dict[str, float],
    train_models: set[str],
) -> pd.DataFrame:
    # Import inside the function so that importing this module does not require
    # extra sys.path mutations.
    from src.modelx import load_results

    print("\nLoading results...")
    df = load_results(
        base_folder=str(base_folder),
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )

    df = df[df["model"].isin(all_models) & df["hint"].isin(hint_fractions)]
    print(f"Loaded {len(df)} rows for {df['model'].nunique()} models")

    df = df.copy()
    df["eci"] = df["model"].map(eci_map)

    missing = df[df["eci"].isna()]["model"].unique()
    if len(missing) > 0:
        print(f"WARNING: Missing ECI for {len(missing)} models: {missing.tolist()}")
        df = df.dropna(subset=["eci"])

    df["split"] = df["model"].apply(lambda m: "train" if m in train_models else "test")
    return df


# ------------------------------ fitting helpers ------------------------------

def format_joint_latex(result: dict, hint_transform: Callable[[float], float] = lambda h: h) -> str:
    """Format joint fit result as LaTeX equation."""
    params = result["params"]
    lower = result.get("lower")

    if result["include_cross"]:
        alpha, beta, gamma, delta = params
        sig = rf"\sigma({alpha:.3f} \cdot C + {beta:.3f} \cdot h + {gamma:.3f} \cdot Ch + {delta:.2f})"
    else:
        alpha, beta, delta = params
        sig = rf"\sigma({alpha:.3f} \cdot C + {beta:.3f} \cdot h + {delta:.2f})"

    if lower is not None:
        return rf"${lower:.2f} + {1-lower:.2f} \cdot {sig}$"
    return rf"${sig}$"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _validate_knots(knots: list[float], *, require_endpoints: bool) -> np.ndarray:
    knot_x = np.asarray(knots, dtype=float)
    if knot_x.ndim != 1 or knot_x.shape[0] < 2:
        raise ValueError("hint_knots must be a list of at least 2 floats")
    if np.any(np.diff(knot_x) <= 0):
        raise ValueError("hint_knots must be strictly increasing")
    if float(np.min(knot_x)) < 0.0 or float(np.max(knot_x)) > 1.0:
        raise ValueError("hint_knots must lie within [0, 1]")
    if require_endpoints:
        if not np.isclose(knot_x[0], 0.0) or not np.isclose(knot_x[-1], 1.0):
            raise ValueError("For fixed-endpoints learned transform, hint_knots must start at 0.0 and end at 1.0")
    return knot_x


def _decode_monotone_knot_y_fixed_endpoints(t: np.ndarray) -> np.ndarray:
    """Given unconstrained t (len K-1), return y knots (len K) with y0=0, y_last=1 monotone."""
    u = np.exp(t)  # positive increments
    s = float(np.sum(u))
    cum = np.cumsum(u) / s  # len K-1, last is 1
    y = np.concatenate([np.array([0.0]), cum])
    return y


def _decode_monotone_knot_y_free_endpoints(y0: float, t: np.ndarray) -> np.ndarray:
    """Given y0 and unconstrained t (len K-1), return y knots (len K) monotone with free endpoints.

    We fix total increase to 1.0 for identifiability (so g(1)=g(0)+1), while allowing a free shift y0.
    """
    u = np.exp(t)
    s = float(np.sum(u))
    cum = np.cumsum(u) / s  # last is 1
    y = y0 + np.concatenate([np.array([0.0]), cum])
    return y


def fit_joint_sigmoid_with_learned_hint_transform(
    *,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    fit_models: set[str],
    hint_knots: list[float],
    include_cross: bool,
    lower: float | None,
    mode: str,
) -> dict[str, object]:
    """Fit joint sigmoid with a learned monotone piecewise-linear hint transform.

    Args:
        df: DataFrame with columns model, hint, accuracy
        eci_map: mapping model->eci
        fit_models: set of model names to fit on (train set)
        hint_knots: x-knots in [0,1] (strictly increasing)
        include_cross: include C*H term
        lower: lower asymptote L
        mode: 'fixed_endpoints' or 'free_endpoints'
    """
    from scipy.optimize import minimize
    from src.modelx.fitting import fit_joint_sigmoid

    if mode not in {"fixed_endpoints", "free_endpoints"}:
        raise ValueError(f"Unknown mode: {mode}")

    require_endpoints = (mode == "fixed_endpoints")
    knot_x = _validate_knots(hint_knots, require_endpoints=require_endpoints)
    K = int(knot_x.shape[0])

    train_df = df[df["model"].isin(fit_models)].copy()
    if len(train_df) == 0:
        raise ValueError("No training rows after filtering to fit_models")

    # Build training arrays
    C = train_df["model"].map(eci_map).astype(float).to_numpy()
    H_raw = train_df["hint"].astype(float).to_numpy()
    y = train_df["accuracy"].astype(float).to_numpy()

    L = float(lower) if lower is not None else 0.0
    U_minus_L = 1.0 - L

    # Initialize joint params using existing fitter with identity transform.
    init = fit_joint_sigmoid(
        df,
        y_col="accuracy",
        hint_col="hint",
        include_cross=include_cross,
        hint_transform=hint_identity,
        use_log_x=False,
        x_values=eci_map,
        fit_models=fit_models,
        lower=lower,
    )
    init_params = np.asarray(init["params"], dtype=float)

    # Parameter vector:
    # - joint params (3 or 4)
    # - transform params:
    #   - fixed_endpoints: t of length K-1
    #   - free_endpoints: y0 (scalar) + t of length K-1
    if include_cross:
        p_joint = 4
    else:
        p_joint = 3

    t0 = np.zeros(K - 1, dtype=float)  # exp(t)=1 -> linear mapping
    if mode == "fixed_endpoints":
        theta0 = np.concatenate([init_params[:p_joint], t0])
    else:
        y0 = 0.0
        theta0 = np.concatenate([init_params[:p_joint], np.array([y0]), t0])

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint = theta[:p_joint]
        if mode == "fixed_endpoints":
            t = theta[p_joint:]
            knot_y = _decode_monotone_knot_y_fixed_endpoints(t)
        else:
            y0_local = float(theta[p_joint])
            t = theta[p_joint + 1 :]
            knot_y = _decode_monotone_knot_y_free_endpoints(y0_local, t)
        return joint, knot_y

    def mse(theta: np.ndarray) -> float:
        joint, knot_y = unpack(theta)
        g = np.interp(np.clip(H_raw, knot_x[0], knot_x[-1]), knot_x, knot_y)
        if include_cross:
            alpha, beta, gamma, delta = joint
            z = alpha * C + beta * g + gamma * C * g + delta
        else:
            alpha, beta, delta = joint
            z = alpha * C + beta * g + delta
        yhat = L + U_minus_L * _sigmoid(z)
        err = yhat - y
        return float(np.mean(err * err))

    res = minimize(
        mse,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )

    joint_opt, knot_y_opt = unpack(res.x)
    transform = PiecewiseLinearHintTransform(knot_x=knot_x, knot_y=knot_y_opt)

    def predict(eci: float, hint: float) -> float:
        g = transform(hint)
        if include_cross:
            alpha, beta, gamma, delta = joint_opt
            z = alpha * float(eci) + beta * g + gamma * float(eci) * g + delta
        else:
            alpha, beta, delta = joint_opt
            z = alpha * float(eci) + beta * g + delta
        return float(L + U_minus_L * (1.0 / (1.0 + math.exp(-z))))

    # Compute RMS on train objective data
    rms = float(math.sqrt(mse(res.x)))

    return {
        "params": joint_opt,
        "rms": rms,
        "include_cross": include_cross,
        "predict": predict,
        "use_log_x": False,
        "lower": lower,
        "hint_transform_type": f"learned_piecewise_linear_{mode}",
        "hint_knots": knot_x,
        "learned_knot_y": knot_y_opt,
        "optimizer_success": bool(res.success),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        "optimizer_nit": int(getattr(res, "nit", -1)),
        "optimizer_fun": float(getattr(res, "fun", float("nan"))),
    }


def fit_individual_sigmoids_by_hint(
    df: pd.DataFrame,
    eci_map: dict[str, float],
    *,
    fit_models: set[str] | None = None,
    lower: float | None = None,
) -> dict[float, dict]:
    """Fit σ(αC + β) for each hint value using `src.modelx.fitting.fit_sigmoid`."""
    results: dict[float, dict] = {}
    for hint in sorted(df["hint"].unique()):
        hint_df = df[df["hint"] == hint]
        if fit_models is not None:
            hint_df = hint_df[hint_df["model"].isin(fit_models)]
        if len(hint_df) < 3:
            continue

        C = hint_df["model"].map(eci_map).values
        y = hint_df["accuracy"].values
        if np.allclose(y, y[0]):
            # Degenerate: constant target => non-identifiable sigmoid.
            continue

        from src.modelx.fitting import fit_sigmoid, format_equation

        result = fit_sigmoid(C, y, use_log=False, lower=lower)
        if lower is not None:
            # asymptote params: (L, U, m, b)
            _L, _U, m, b = result["params"]
            midpoint = -b / m
        else:
            # basic params: (m, b)
            m, b = result["params"]
            midpoint = -b / m

        results[hint] = {
            "params": result["params"],
            "midpoint": midpoint,
            "predict": result["predict"],
            "equation": format_equation(result),
        }

    return results


def fit_individual_sigmoids_by_model(
    df: pd.DataFrame,
    hint_transform: Callable[[float], float] = lambda h: h,
    *,
    fit_models: set[str] | None = None,
    lower: float | None = None,
) -> dict[str, dict]:
    """Fit σ(βh + γ) for each model using `src.modelx.fitting.fit_sigmoid`."""
    models_to_fit = fit_models if fit_models is not None else set(df["model"].unique())
    results: dict[str, dict] = {}
    for model in sorted(models_to_fit):
        model_df = df[df["model"] == model]
        if len(model_df) < 3:
            continue

        H = np.array([hint_transform(h) for h in model_df["hint"].values])
        y = model_df["accuracy"].values
        if np.allclose(y, y[0]):
            continue

        from src.modelx.fitting import fit_sigmoid, format_equation

        result = fit_sigmoid(H, y, use_log=False, lower=lower)
        results[model] = {
            "params": result["params"],
            "predict": result["predict"],
            "equation": format_equation(result),
        }

    return results


def compute_midpoint_errors(
    joint_result: dict,
    individual_fits: dict[float, dict],
    hints: list[float],
    hint_transform: Callable[[float], float] = lambda h: h,
) -> dict[float, float]:
    """Compute |C_midpoint(joint, h) - C_midpoint(individual, h)| per hint."""
    errors: dict[float, float] = {}
    params = joint_result["params"]

    for hint in hints:
        if hint not in individual_fits:
            continue

        individual_midpoint = individual_fits[hint]["midpoint"]
        h_t = hint_transform(hint)

        if joint_result["include_cross"]:
            alpha, beta, gamma, delta = params
            denom = alpha + gamma * h_t
            if abs(denom) <= 1e-6:
                continue
            joint_midpoint = (-beta * h_t - delta) / denom
        else:
            alpha, beta, delta = params
            if abs(alpha) <= 1e-6:
                continue
            joint_midpoint = (-beta * h_t - delta) / alpha

        errors[hint] = abs(joint_midpoint - individual_midpoint)

    return errors


def compute_rms(
    joint_result: dict,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    models: set[str] | None = None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if len(eval_df) == 0:
        return float("nan")
    y_pred = np.array(
        [
            joint_result["predict"](eci_map[m], h)
            for m, h in zip(eval_df["model"], eval_df["hint"])
        ]
    )
    y_actual = eval_df["accuracy"].values
    return float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))


def compute_rms_individual(
    individual_by_hint: dict[float, dict],
    df: pd.DataFrame,
    eci_map: dict[str, float],
    models: set[str] | None = None,
) -> float:
    """Compute RMSE for per-hint individual sigmoid fits against a given set of models."""
    eval_df = df if models is None else df[df["model"].isin(models)]
    if len(eval_df) == 0:
        return float("nan")
    preds, actuals = [], []
    for _, row in eval_df.iterrows():
        hint = row["hint"]
        if hint not in individual_by_hint:
            continue
        preds.append(individual_by_hint[hint]["predict"](eci_map[row["model"]]))
        actuals.append(row["accuracy"])
    if not preds:
        return float("nan")
    return float(np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2)))


def run_model_sweep(
    df: pd.DataFrame,
    eci_map: dict[str, float],
    hint_fractions: list[float],
    hint_transform: Callable[[float], float],
    include_cross: bool,
    lower_asymptote: float | None,
    *,
    eval_hints: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep number of train models (sorted by ECI) and compute RMS + midpoint errors."""
    if eval_hints is None:
        eval_hints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    all_models = sorted(df["model"].unique(), key=lambda m: eci_map.get(m, 0))
    individual_by_hint_all = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=None, lower=lower_asymptote)

    from src.modelx.fitting import fit_joint_sigmoid

    rows: list[dict[str, float]] = []
    for n in range(5, len(all_models) + 1):
        train_models = set(all_models[:n])
        test_models = set(all_models[n:])

        joint_result = fit_joint_sigmoid(
            df,
            y_col="accuracy",
            hint_col="hint",
            include_cross=include_cross,
            hint_transform=hint_transform,
            use_log_x=False,
            x_values=eci_map,
            fit_models=train_models,
            lower=lower_asymptote,
        )

        rms_train = compute_rms(joint_result, df, eci_map, train_models)
        rms_test = compute_rms(joint_result, df, eci_map, test_models) if test_models else float("nan")
        rms_all = compute_rms(joint_result, df, eci_map, None)
        rms_h0_all = compute_rms(joint_result, df[df["hint"] == 0.0], eci_map, None)
        rms_h0_test = compute_rms(joint_result, df[df["hint"] == 0.0], eci_map, test_models) if test_models else float("nan")

        individual_by_hint_train = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=train_models, lower=lower_asymptote)
        rms_indiv_train = compute_rms_individual(individual_by_hint_train, df, eci_map, train_models)
        rms_indiv_test = compute_rms_individual(individual_by_hint_train, df, eci_map, test_models) if test_models else float("nan")
        rms_indiv_all = compute_rms_individual(individual_by_hint_train, df, eci_map, None)

        # Midpoint errors vs ground truth (individual fit on all data).
        # Joint midpoint error: |joint_midpoint - indiv_all_midpoint|
        midpoint_errors_joint = compute_midpoint_errors(joint_result, individual_by_hint_all, eval_hints, hint_transform)
        # Individual-train midpoint error: |indiv_train_midpoint - indiv_all_midpoint|
        midpoint_errors_indiv = {}
        for h in eval_hints:
            if h in individual_by_hint_train and h in individual_by_hint_all:
                midpoint_errors_indiv[h] = abs(individual_by_hint_train[h]["midpoint"] - individual_by_hint_all[h]["midpoint"])
        row: dict[str, float] = {
            "n_models": float(n),
            "rms_train": rms_train, "rms_test": rms_test, "rms_all": rms_all,
            "rms_h0_all": rms_h0_all, "rms_h0_test": rms_h0_test,
            "rms_indiv_train": rms_indiv_train, "rms_indiv_test": rms_indiv_test, "rms_indiv_all": rms_indiv_all,
            "delta_rms_train": rms_train - rms_indiv_train,
            "delta_rms_test": rms_test - rms_indiv_test,
            "delta_rms_all": rms_all - rms_indiv_all,
        }
        for h in eval_hints:
            row[f"midpoint_h_{h:.1f}"] = float(midpoint_errors_joint.get(h, float("nan")))
            me_joint = midpoint_errors_joint.get(h, float("nan"))
            me_indiv = midpoint_errors_indiv.get(h, float("nan"))
            row[f"delta_midpoint_h_{h:.1f}"] = me_joint - me_indiv
        rows.append(row)

    out = pd.DataFrame(rows)
    out["n_models"] = out["n_models"].astype(int)
    return out


# ------------------------------ plotting ------------------------------

def plot_accuracy_vs_eci_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict,
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

    eci_range = np.linspace(df["eci"].min() - 5, df["eci"].max() + 5, 100)
    for hint in hints:
        hint_df = df[df["hint"] == hint].sort_values("eci")
        train_df = hint_df[hint_df["split"] == "train"]
        ax.scatter(
            train_df["eci"],
            train_df["accuracy"],
            color=colors[hint],
            label=f"h={hint:.2f}",
            alpha=0.8,
            s=60,
            marker="o",
        )

        test_df = hint_df[hint_df["split"] == "test"]
        if len(test_df) > 0:
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color=colors[hint],
                alpha=0.8,
                s=60,
                marker="s",
                edgecolors="black",
            )

        y_fit = [joint_result["predict"](c, hint) for c in eci_range]
        ax.plot(eci_range, y_fit, "-", color=colors[hint], alpha=0.5, linewidth=2)

    ax.set_xlabel("eci", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_title(f"{label}\n{joint_latex}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "accuracy_vs_eci_by_hint.png"
    save_figure(fig, out_path)
    return out_path


def plot_individual_fits_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict,
    individual_by_hint: dict[float, dict],
    individual_by_hint_train: dict[float, dict],
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}
    eci_range = np.linspace(df["eci"].min() - 5, df["eci"].max() + 5, 100)

    n_rows = 3
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, hint in enumerate(hints):
        ax = axes[i]
        hint_df = df[df["hint"] == hint].sort_values("eci")
        train_df = hint_df[hint_df["split"] == "train"]
        ax.scatter(train_df["eci"], train_df["accuracy"], color=colors[hint], alpha=0.8, s=40)

        test_df = hint_df[hint_df["split"] == "test"]
        if len(test_df) > 0:
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color=colors[hint],
                alpha=0.8,
                s=40,
                marker="s",
                edgecolors="black",
            )

        y_joint = [joint_result["predict"](c, hint) for c in eci_range]
        ax.plot(eci_range, y_joint, "--", color="gray", linewidth=2, label="joint (train)")

        if hint in individual_by_hint_train:
            y_indiv_train = [individual_by_hint_train[hint]["predict"](c) for c in eci_range]
            ax.plot(eci_range, y_indiv_train, "-", color="orange", linewidth=2, label="indiv (train)")

        if hint in individual_by_hint:
            y_indiv_all = [individual_by_hint[hint]["predict"](c) for c in eci_range]
            ax.plot(eci_range, y_indiv_all, "-", color=colors[hint], linewidth=2, label="indiv (all)")
            midpoint = individual_by_hint[hint]["midpoint"]
            ax.axvline(midpoint, color=colors[hint], linestyle=":", alpha=0.5)

        ax.set_title(f"h = {hint:.2f}", fontsize=11)
        ax.set_xlabel("eci")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(len(hints), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{label} - Individual fits per hint\nJoint: {joint_latex}", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / "individual_fits_by_hint.png"
    save_figure(fig, out_path)
    return out_path


def plot_accuracy_vs_hint_by_model(
    *,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    joint_result: dict,
    individual_by_model: dict[str, dict],
    label: str,
    joint_latex: str,
    output_dir: Path,
) -> Path:
    models_sorted = sorted(df["model"].unique(), key=lambda m: eci_map.get(m, 0))
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    hint_range = np.linspace(0, 1, 100)
    model_cmap = plt.cm.coolwarm
    model_colors = {m: model_cmap(i / max(n_models - 1, 1)) for i, m in enumerate(models_sorted)}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        eci = eci_map.get(model, 0.0)

        ax.scatter(model_df["hint"], model_df["accuracy"], color=model_colors[model], alpha=0.8, s=40)

        y_joint = [joint_result["predict"](eci, h) for h in hint_range]
        ax.plot(hint_range, y_joint, "--", color="gray", linewidth=2, label="joint fit")

        if model in individual_by_model:
            y_indiv = [individual_by_model[model]["predict"](h) for h in hint_range]
            ax.plot(hint_range, y_indiv, "-", color=model_colors[model], linewidth=2, label="individual fit")

        ax.set_title(f"{model}\neci={eci:.1f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{label} - Accuracy vs Hint per model\nJoint: {joint_latex}", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / "accuracy_vs_hint_by_model.png"
    save_figure(fig, out_path)
    return out_path


def plot_model_sweep(
    *,
    sweep_df: pd.DataFrame,
    label: str,
    eval_hints: list[float],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(sweep_df["n_models"], sweep_df["rms_train"], "o-", label="train", color="blue")
    ax.plot(sweep_df["n_models"], sweep_df["rms_test"], "s-", label="test", color="red")
    ax.plot(sweep_df["n_models"], sweep_df["rms_all"], "^-", label="all", color="green")
    ax.set_xlabel("number of train models")
    ax.set_ylabel("rms")
    ax.set_title("RMS vs number of train models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sweep_df["n_models"], sweep_df["delta_rms_train"], "o-", label="train", color="blue")
    ax.plot(sweep_df["n_models"], sweep_df["delta_rms_test"], "s-", label="test", color="red")
    ax.plot(sweep_df["n_models"], sweep_df["delta_rms_all"], "^-", label="all", color="green")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("number of train models")
    ax.set_ylabel("delta RMS (joint - individual)")
    ax.set_title("delta RMS vs number of train models\n(negative = joint wins)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(eval_hints) - 1, 1)) for i, h in enumerate(eval_hints)}
    for h in eval_hints:
        col = f"midpoint_h_{h:.1f}"
        if col in sweep_df.columns:
            ax.plot(sweep_df["n_models"], sweep_df[col], "o-", label=f"h={h:.1f}", color=colors[h])
    ax.set_xlabel("number of train models")
    ax.set_ylabel("midpoint error (eci units)")
    ax.set_title("midpoint error per hint vs number of train models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for h in eval_hints:
        col = f"delta_midpoint_h_{h:.1f}"
        if col in sweep_df.columns:
            ax.plot(sweep_df["n_models"], sweep_df[col], "o-", label=f"h={h:.1f}", color=colors[h])
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("number of train models")
    ax.set_ylabel("delta midpoint error (eci units)")
    ax.set_title("delta midpoint error vs number of train models\n(negative = joint wins)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} (fitting joint scaling)", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / "model_sweep.png"
    save_figure(fig, out_path)
    return out_path


# ------------------------------ orchestration ------------------------------

def run_joint_scaling_plots(
    *,
    base_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    label: str,
    all_models: list[str],
    num_holdout_models: int,
    hint_fractions: list[float],
    eval_hints_for_sweep: list[float],
    include_cross: bool,
    lower_asymptote: float | None,
    hint_transform: str | Callable[[float], float],
    hint_knots: list[float] | None = None,
    output_dir: Path,
) -> dict[str, object]:
    """Run the joint scaling analysis and write artifacts to `output_dir`.

    Split logic:
    - Fit is performed on a train set computed by holding out the top
      `num_holdout_models` models by ECI (highest ECIs are held out).
    - The holdout set is used as the "test" set for summary metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write plots directly into the run directory (no nested `plots/`).
    plots_dir = output_dir

    learned_mode: str | None = None
    learned_knot_x: np.ndarray | None = None
    learned_knot_y: np.ndarray | None = None
    learned_optimizer_meta: dict[str, object] | None = None

    if isinstance(hint_transform, str):
        hint_transform_name = hint_transform
        if hint_transform_name in {"learned_piecewise_linear_fixed_endpoints", "learned_piecewise_linear_free_endpoints"}:
            if hint_knots is None:
                raise ValueError(f"hint_knots must be provided for hint_transform={hint_transform_name!r}")
            learned_mode = "fixed_endpoints" if hint_transform_name == "learned_piecewise_linear_fixed_endpoints" else "free_endpoints"
            # We'll fill hint_transform_fn after we compute the train set and run the learned fit.
            hint_transform_fn = hint_identity
        else:
            if hint_transform_name not in HINT_TRANSFORMS:
                raise ValueError(
                    f"Unknown hint_transform={hint_transform_name!r}. "
                    f"Choose one of: {sorted(HINT_TRANSFORMS.keys()) + ['learned_piecewise_linear_fixed_endpoints', 'learned_piecewise_linear_free_endpoints']}"
                )
            hint_transform_fn = HINT_TRANSFORMS[hint_transform_name]
    else:
        hint_transform_fn = hint_transform
        hint_transform_name = getattr(hint_transform_fn, "__name__", "custom")

    eci_map = load_eci_map(eci_file)

    if num_holdout_models < 0:
        raise ValueError(f"num_holdout_models must be >= 0, got {num_holdout_models}")
    if num_holdout_models > len(all_models):
        raise ValueError(
            f"num_holdout_models ({num_holdout_models}) cannot exceed number of models ({len(all_models)})"
        )

    models_sorted_by_eci = sorted(all_models, key=lambda m: eci_map.get(m, float("-inf")))
    holdout_set = set(models_sorted_by_eci[-num_holdout_models:]) if num_holdout_models > 0 else set()
    train_set = set(models_sorted_by_eci[:-num_holdout_models]) if num_holdout_models > 0 else set(all_models)

    print_eci_table(all_models=all_models, eci_map=eci_map, train_models=train_set)

    df = load_and_prepare_results_df(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
        eci_map=eci_map,
        train_models=train_set,
    )

    from src.modelx.fitting import format_equation

    if learned_mode is None:
        from src.modelx.fitting import fit_joint_sigmoid

        joint_result = fit_joint_sigmoid(
            df,
            y_col="accuracy",
            hint_col="hint",
            include_cross=include_cross,
            hint_transform=hint_transform_fn,
            use_log_x=False,
            x_values=eci_map,
            fit_models=train_set,
            lower=lower_asymptote,
        )
    else:
        learned_fit = fit_joint_sigmoid_with_learned_hint_transform(
            df=df,
            eci_map=eci_map,
            fit_models=train_set,
            hint_knots=hint_knots if hint_knots is not None else [],
            include_cross=include_cross,
            lower=lower_asymptote,
            mode=learned_mode,
        )
        joint_result = learned_fit
        learned_knot_x = np.asarray(learned_fit["hint_knots"], dtype=float)
        learned_knot_y = np.asarray(learned_fit["learned_knot_y"], dtype=float)
        hint_transform_fn = PiecewiseLinearHintTransform(knot_x=learned_knot_x, knot_y=learned_knot_y)
        learned_optimizer_meta = {
            "optimizer_success": bool(learned_fit.get("optimizer_success")),
            "optimizer_status": int(learned_fit.get("optimizer_status", -1)),
            "optimizer_message": str(learned_fit.get("optimizer_message", "")),
            "optimizer_nit": int(learned_fit.get("optimizer_nit", -1)),
            "optimizer_fun": float(learned_fit.get("optimizer_fun", float("nan"))),
        }
        plot_hint_transform_mapping(
            knot_x=learned_knot_x,
            knot_y=learned_knot_y,
            output_dir=plots_dir,
            title=f"{label} - learned hint mapping ({learned_mode})",
        )

    joint_latex = format_joint_latex(joint_result, hint_transform_fn)
    joint_equation = format_equation(joint_result)

    # Holdout/test set is the top-ECI models.
    test_set = holdout_set

    rms_train = compute_rms(joint_result, df, eci_map, train_set)
    rms_test = compute_rms(joint_result, df, eci_map, test_set) if test_set else float("nan")
    rms_all = compute_rms(joint_result, df, eci_map, None)

    individual_by_hint_all = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=None, lower=lower_asymptote)
    individual_by_hint_train = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=train_set, lower=lower_asymptote)

    rms_indiv_train = compute_rms_individual(individual_by_hint_train, df, eci_map, train_set)
    rms_indiv_test = compute_rms_individual(individual_by_hint_train, df, eci_map, test_set) if test_set else float("nan")
    rms_indiv_all = compute_rms_individual(individual_by_hint_train, df, eci_map, None)
    delta_rms_train = rms_train - rms_indiv_train
    delta_rms_test = rms_test - rms_indiv_test
    delta_rms_all = rms_all - rms_indiv_all
    # Keep per-model curves in raw hint space so the x-axis remains "hint fraction".
    individual_by_model = fit_individual_sigmoids_by_model(df, hint_identity, fit_models=None, lower=lower_asymptote)

    midpoint_errors_all = compute_midpoint_errors(joint_result, individual_by_hint_all, hint_fractions, hint_transform_fn)
    midpoint_errors_train = compute_midpoint_errors(joint_result, individual_by_hint_train, hint_fractions, hint_transform_fn)
    mean_midpoint_error_all = float(np.mean(list(midpoint_errors_all.values()))) if midpoint_errors_all else float("nan")
    mean_midpoint_error_train = float(np.mean(list(midpoint_errors_train.values()))) if midpoint_errors_train else float("nan")

    if test_set:
        individual_by_hint_test = fit_individual_sigmoids_by_hint(df, eci_map, fit_models=test_set, lower=lower_asymptote)
        midpoint_errors_test = compute_midpoint_errors(joint_result, individual_by_hint_test, hint_fractions, hint_transform_fn)
        mean_midpoint_error_test = float(np.mean(list(midpoint_errors_test.values()))) if midpoint_errors_test else float("nan")
    else:
        mean_midpoint_error_test = float("nan")

    plot_accuracy_vs_eci_by_hint(
        df=df,
        joint_result=joint_result,
        label=label,
        joint_latex=joint_latex,
        output_dir=plots_dir,
    )
    plot_individual_fits_by_hint(
        df=df,
        joint_result=joint_result,
        individual_by_hint=individual_by_hint_all,
        individual_by_hint_train=individual_by_hint_train,
        label=label,
        joint_latex=joint_latex,
        output_dir=plots_dir,
    )
    plot_accuracy_vs_hint_by_model(
        df=df,
        eci_map=eci_map,
        joint_result=joint_result,
        individual_by_model=individual_by_model,
        label=label,
        joint_latex=joint_latex,
        output_dir=plots_dir,
    )

    sweep_df = run_model_sweep(
        df,
        eci_map,
        hint_fractions,
        hint_transform_fn,
        include_cross,
        lower_asymptote,
        eval_hints=eval_hints_for_sweep,
    )
    plot_model_sweep(
        sweep_df=sweep_df,
        label=label,
        eval_hints=eval_hints_for_sweep,
        output_dir=plots_dir,
    )
    sweep_df.to_csv(output_dir / "model_sweep.csv", index=False)

    config_resolved: dict[str, object] = {
        "output_dir": str(output_dir),
        "base_folder": str(base_folder),
        "eci_file": str(eci_file),
        "eval_name": eval_name,
        "solver": solver,
        "condition": condition,
        "label": label,
        "all_models": all_models,
        "num_holdout_models": int(num_holdout_models),
        "train_models": sorted(train_set, key=lambda m: eci_map.get(m, float("-inf"))),
        "holdout_models": sorted(test_set, key=lambda m: eci_map.get(m, float("-inf"))),
        "hint_fractions": hint_fractions,
        "eval_hints_for_sweep": eval_hints_for_sweep,
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "hint_transform": hint_transform_name,
        "hint_knots": hint_knots,
        "learned_knot_y": learned_knot_y.tolist() if learned_knot_y is not None else None,
        "learned_optimizer": learned_optimizer_meta,
    }

    metrics: dict[str, object] = {
        "joint_equation": joint_equation,
        "joint_latex": joint_latex,
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "rms_train": rms_train,
        "rms_test": rms_test,
        "rms_all": rms_all,
        "rms_indiv_train": rms_indiv_train,
        "rms_indiv_test": rms_indiv_test,
        "rms_indiv_all": rms_indiv_all,
        "delta_rms_train": delta_rms_train,
        "delta_rms_test": delta_rms_test,
        "delta_rms_all": delta_rms_all,
        "mean_midpoint_error_all": mean_midpoint_error_all,
        "mean_midpoint_error_train": mean_midpoint_error_train,
        "mean_midpoint_error_test": mean_midpoint_error_test,
        "n_train_models": int(len(train_set)),
        "n_test_models": int(len(test_set)),
        "joint_params": [float(x) for x in joint_result["params"]],
        "config": config_resolved,
    }

    write_json(output_dir / "config_resolved.json", config_resolved)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


# =============================================================================
# PC-capability variant (capability = alpha·PC)
# =============================================================================


def load_and_prepare_results_df_with_capability(
    *,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
    all_models: list[str],
    hint_fractions: list[float],
    capability_map: dict[str, float],
    train_models: set[str],
    eci_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Load results and attach scalar capability (and optionally ECI) columns."""
    from src.modelx import load_results

    print("\nLoading results...")
    df = load_results(
        base_folder=str(base_folder),
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )
    df = df[df["model"].isin(all_models) & df["hint"].isin(hint_fractions)]
    print(f"Loaded {len(df)} rows for {df['model'].nunique()} models")

    df = df.copy()
    df["capability"] = df["model"].map(capability_map)
    missing = df[df["capability"].isna()]["model"].unique()
    if len(missing) > 0:
        print(f"WARNING: Missing capability for {len(missing)} models: {missing.tolist()}")
        df = df.dropna(subset=["capability"])

    if eci_map is not None:
        df["eci"] = df["model"].map(eci_map)

    df["split"] = df["model"].apply(lambda m: "train" if m in train_models else "test")
    return df


def _predict_from_capability(
    *,
    capability: float,
    hint: float,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
) -> float:
    L = float(lower) if lower is not None else 0.0
    U_minus_L = 1.0 - L
    h_t = float(hint_transform(float(hint)))
    x = float(capability)
    if include_cross and gamma is not None:
        z = x + float(beta) * h_t + float(gamma) * x * h_t + float(delta)
    else:
        z = x + float(beta) * h_t + float(delta)
    return float(L + U_minus_L * (1.0 / (1.0 + math.exp(-z))))


def compute_rms_capability_fit(
    *,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
    df: pd.DataFrame,
    models: set[str] | None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if len(eval_df) == 0:
        return float("nan")
    y_pred = np.array(
        [
            _predict_from_capability(
                capability=float(cap),
                hint=float(h),
                beta=beta,
                gamma=gamma,
                delta=delta,
                include_cross=include_cross,
                hint_transform=hint_transform,
                lower=lower,
            )
            for cap, h in zip(eval_df["capability"], eval_df["hint"])
        ],
        dtype=float,
    )
    y_actual = eval_df["accuracy"].astype(float).to_numpy()
    return float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))


def fit_individual_sigmoids_by_hint_capability(
    df: pd.DataFrame,
    *,
    fit_models: set[str] | None = None,
    lower: float | None = None,
) -> dict[float, dict]:
    """Fit σ(m*capability + b) per hint."""
    from src.modelx.fitting import fit_sigmoid, format_equation

    results: dict[float, dict] = {}
    for hint in sorted(df["hint"].unique()):
        hint_df = df[df["hint"] == hint]
        if fit_models is not None:
            hint_df = hint_df[hint_df["model"].isin(fit_models)]
        if len(hint_df) < 3:
            continue

        x = hint_df["capability"].astype(float).to_numpy()
        y = hint_df["accuracy"].astype(float).to_numpy()
        if np.allclose(y, y[0]):
            continue

        res = fit_sigmoid(x, y, use_log=False, lower=lower)
        if lower is not None:
            _L, _U, m, b = res["params"]
        else:
            m, b = res["params"]
        midpoint = float(-b / m) if abs(float(m)) > 1e-12 else float("nan")

        results[float(hint)] = {
            "params": res["params"],
            "midpoint": midpoint,
            "predict": res["predict"],
            "equation": format_equation(res),
        }
    return results


def fit_individual_sigmoids_by_model_hint_space(
    df: pd.DataFrame,
    *,
    fit_models: set[str] | None = None,
    lower: float | None = None,
) -> dict[str, dict]:
    """Fit σ(m*h + b) per model (x-axis is raw hint fraction)."""
    from src.modelx.fitting import fit_sigmoid, format_equation

    models_to_fit = sorted(fit_models) if fit_models is not None else sorted(df["model"].unique().tolist())
    out: dict[str, dict] = {}
    for model in models_to_fit:
        model_df = df[df["model"] == model].sort_values("hint")
        if len(model_df) < 3:
            continue

        x = model_df["hint"].astype(float).to_numpy()
        y = model_df["accuracy"].astype(float).to_numpy()
        if np.allclose(y, y[0]):
            continue

        res = fit_sigmoid(x, y, use_log=False, lower=lower)
        out[str(model)] = {
            "params": res["params"],
            "predict": res["predict"],
            "equation": format_equation(res),
        }
    return out


def compute_midpoint_errors_capability(
    *,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    individual_fits: dict[float, dict],
    hints: list[float],
    hint_transform: Callable[[float], float] = lambda h: h,
) -> dict[float, float]:
    """Compute |cap_midpoint(joint,h) - cap_midpoint(individual,h)| per hint."""
    errors: dict[float, float] = {}
    for hint in hints:
        hint = float(hint)
        if hint not in individual_fits:
            continue
        indiv_mid = float(individual_fits[hint]["midpoint"])
        h_t = float(hint_transform(hint))

        if include_cross and gamma is not None:
            denom = 1.0 + float(gamma) * h_t
            if abs(denom) <= 1e-8:
                continue
            joint_mid = (-float(beta) * h_t - float(delta)) / denom
        else:
            joint_mid = -float(beta) * h_t - float(delta)

        errors[hint] = float(abs(joint_mid - indiv_mid))
    return errors


def _draw_accuracy_vs_capability_by_hint(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
    label: str,
    title_equation: str,
    x_label: str = "capability",
    legend: bool = True,
) -> None:
    """Draw accuracy-vs-capability scatter+fit into an existing axes object."""
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

    x_min = float(df["capability"].min())
    x_max = float(df["capability"].max())
    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 120)

    for hint in hints:
        hint_df = df[df["hint"] == hint].sort_values("capability")
        train_df = hint_df[hint_df["split"] == "train"]
        ax.scatter(train_df["capability"], train_df["accuracy"], color=colors[hint], label=f"h={hint:.2f}", alpha=0.8, s=60, marker="o")

        test_df = hint_df[hint_df["split"] == "test"]
        if len(test_df) > 0:
            ax.scatter(test_df["capability"], test_df["accuracy"], color=colors[hint], alpha=0.8, s=60, marker="s", edgecolors="black")

        y_fit = [
            _predict_from_capability(
                capability=float(x),
                hint=float(hint),
                beta=beta,
                gamma=gamma,
                delta=delta,
                include_cross=include_cross,
                hint_transform=hint_transform,
                lower=lower,
            )
            for x in x_range
        ]
        ax.plot(x_range, y_fit, "-", color=colors[hint], alpha=0.5, linewidth=2)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("accuracy", fontsize=10)
    ax.set_title(f"{label}\n{title_equation}", fontsize=10)
    if legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_accuracy_vs_capability_by_hint(
    *,
    df: pd.DataFrame,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
    label: str,
    title_equation: str,
    output_dir: Path,
    x_label: str = "capability",
    out_name: str = "accuracy_vs_pc_capability_by_hint.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_accuracy_vs_capability_by_hint(
        ax,
        df=df,
        beta=beta,
        gamma=gamma,
        delta=delta,
        include_cross=include_cross,
        hint_transform=hint_transform,
        lower=lower,
        label=label,
        title_equation=title_equation,
        x_label=x_label,
        legend=True,
    )
    plt.tight_layout()
    out_path = output_dir / out_name
    save_figure(fig, out_path)
    return out_path


def plot_all_pc_accuracy_vs_capability(
    *,
    pc_panels: list[dict],
    label: str,
    output_dir: Path,
    out_name: str = "comparison_accuracy_vs_capability_by_hint.png",
) -> Path:
    """Grid of accuracy-vs-capability plots, one panel per PC method.

    Each entry in `pc_panels` must have:
      df             – DataFrame with columns capability, hint, accuracy, split
      beta, gamma, delta, include_cross – fit parameters
      hint_transform – callable
      lower          – lower asymptote
      method_label   – display name (e.g. "PC1", "PC1+PC2")
      title_equation – equation string for the subplot title
    """
    n = len(pc_panels)
    if n == 0:
        return output_dir / out_name
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13 * n_cols, 7 * n_rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, panel in enumerate(pc_panels):
        _draw_accuracy_vs_capability_by_hint(
            axes_flat[i],
            df=panel["df"],
            beta=float(panel["beta"]),
            gamma=(float(panel["gamma"]) if panel.get("gamma") is not None else None),
            delta=float(panel["delta"]),
            include_cross=bool(panel["include_cross"]),
            hint_transform=panel["hint_transform"],
            lower=panel["lower"],
            label=panel["method_label"],
            title_equation=panel["title_equation"],
            x_label="pc_capability",
            legend=(i == 0),
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{label}: accuracy vs PC capability by hint fraction", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / out_name
    save_figure(fig, out_path)
    return out_path


def plot_individual_fits_by_hint_capability(
    *,
    df: pd.DataFrame,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
    individual_by_hint_all: dict[float, dict],
    individual_by_hint_train: dict[float, dict],
    label: str,
    title_equation: str,
    output_dir: Path,
    x_label: str = "capability",
    out_name: str = "individual_fits_by_hint.png",
) -> Path:
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

    x_min = float(df["capability"].min())
    x_max = float(df["capability"].max())
    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 120)

    n_rows, n_cols = 3, 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, hint in enumerate(hints):
        ax = axes[i]
        hint_df = df[df["hint"] == hint].sort_values("capability")
        train_df = hint_df[hint_df["split"] == "train"]
        ax.scatter(train_df["capability"], train_df["accuracy"], color=colors[hint], alpha=0.8, s=40)

        test_df = hint_df[hint_df["split"] == "test"]
        if len(test_df) > 0:
            ax.scatter(test_df["capability"], test_df["accuracy"], color=colors[hint], alpha=0.8, s=40, marker="s", edgecolors="black")

        y_joint = [
            _predict_from_capability(
                capability=float(x),
                hint=float(hint),
                beta=beta,
                gamma=gamma,
                delta=delta,
                include_cross=include_cross,
                hint_transform=hint_transform,
                lower=lower,
            )
            for x in x_range
        ]
        ax.plot(x_range, y_joint, "--", color="gray", linewidth=2, label="joint (train)")

        if hint in individual_by_hint_train:
            y_indiv_train = [float(individual_by_hint_train[hint]["predict"](x)) for x in x_range]
            ax.plot(x_range, y_indiv_train, "-", color="orange", linewidth=2, label="indiv (train)")

        if hint in individual_by_hint_all:
            y_indiv_all = [float(individual_by_hint_all[hint]["predict"](x)) for x in x_range]
            ax.plot(x_range, y_indiv_all, "-", color=colors[hint], linewidth=2, label="indiv (all)")
            ax.axvline(float(individual_by_hint_all[hint]["midpoint"]), color=colors[hint], linestyle=":", alpha=0.5)

        ax.set_title(f"h = {hint:.2f}", fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(len(hints), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{label} - Individual fits per hint\nJoint: {title_equation}", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / out_name
    save_figure(fig, out_path)
    return out_path


def plot_accuracy_vs_hint_by_model_capability(
    *,
    df: pd.DataFrame,
    capability_map: dict[str, float],
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    lower: float | None,
    individual_by_model: dict[str, dict],
    label: str,
    title_equation: str,
    output_dir: Path,
    out_name: str = "accuracy_vs_hint_by_model.png",
) -> Path:
    models_sorted = sorted(df["model"].unique(), key=lambda m: capability_map.get(m, 0.0))
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    hint_range = np.linspace(0, 1, 120)
    model_cmap = plt.cm.coolwarm
    model_colors = {m: model_cmap(i / max(n_models - 1, 1)) for i, m in enumerate(models_sorted)}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        cap = float(capability_map.get(model, float("nan")))

        ax.scatter(model_df["hint"], model_df["accuracy"], color=model_colors[model], alpha=0.8, s=40)

        y_joint = [
            _predict_from_capability(
                capability=cap,
                hint=float(h),
                beta=beta,
                gamma=gamma,
                delta=delta,
                include_cross=include_cross,
                hint_transform=hint_transform,
                lower=lower,
            )
            for h in hint_range
        ]
        ax.plot(hint_range, y_joint, "--", color="gray", linewidth=2, label="joint fit")

        if model in individual_by_model:
            y_indiv = [float(individual_by_model[model]["predict"](h)) for h in hint_range]
            ax.plot(hint_range, y_indiv, "-", color=model_colors[model], linewidth=2, label="individual fit")

        ax.set_title(f"{model}\ncap={cap:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{label} - Accuracy vs Hint per model\nJoint: {title_equation}", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / out_name
    save_figure(fig, out_path)
    return out_path


def run_model_sweep_capability(
    *,
    df: pd.DataFrame,
    pc_scores_map: dict[str, np.ndarray],
    models_sorted: list[str],
    hint_fractions: list[float],
    hint_transform: Callable[[float], float],
    include_cross: bool,
    lower_asymptote: float | None,
    n_pcs: int,
    alpha_fixed: np.ndarray | None,
    eval_hints: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep number of train models (sorted by capability) and compute RMS + midpoint errors."""
    if eval_hints is None:
        eval_hints = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    from pc_joint_helpers import fit_joint_sigmoid_over_pcs

    individual_by_hint_all = fit_individual_sigmoids_by_hint_capability(df, fit_models=None, lower=lower_asymptote)

    rows: list[dict[str, float]] = []
    for n in range(5, len(models_sorted) + 1):
        train_models = set(models_sorted[:n])
        test_models = set(models_sorted[n:])

        fit = fit_joint_sigmoid_over_pcs(
            df=df,
            pc_scores_map=pc_scores_map,
            n_pcs=int(n_pcs),
            fit_models=train_models,
            include_cross=include_cross,
            hint_transform=hint_transform,
            lower=lower_asymptote,
            alpha_fixed=alpha_fixed,
        )

        # Attach the capability for this alpha to compute per-hint individual fits and midpoint errors.
        cap_map = {m: float(np.dot(fit.alpha[: int(n_pcs)], pc_scores_map[m][: int(n_pcs)])) for m in models_sorted}
        df_tmp = df.copy()
        df_tmp["capability"] = df_tmp["model"].map(cap_map).astype(float)

        rms_train = compute_rms_capability_fit(
            beta=fit.beta,
            gamma=fit.gamma,
            delta=fit.delta,
            include_cross=fit.include_cross,
            hint_transform=hint_transform,
            lower=lower_asymptote,
            df=df_tmp,
            models=train_models,
        )
        rms_test = compute_rms_capability_fit(
            beta=fit.beta,
            gamma=fit.gamma,
            delta=fit.delta,
            include_cross=fit.include_cross,
            hint_transform=hint_transform,
            lower=lower_asymptote,
            df=df_tmp,
            models=test_models,
        ) if test_models else float("nan")
        rms_all = compute_rms_capability_fit(
            beta=fit.beta,
            gamma=fit.gamma,
            delta=fit.delta,
            include_cross=fit.include_cross,
            hint_transform=hint_transform,
            lower=lower_asymptote,
            df=df_tmp,
            models=None,
        )
        rms_h0_all = compute_rms_capability_fit(
            beta=fit.beta,
            gamma=fit.gamma,
            delta=fit.delta,
            include_cross=fit.include_cross,
            hint_transform=hint_transform,
            lower=lower_asymptote,
            df=df_tmp[df_tmp["hint"] == 0.0],
            models=None,
        )
        rms_h0_test = compute_rms_capability_fit(
            beta=fit.beta,
            gamma=fit.gamma,
            delta=fit.delta,
            include_cross=fit.include_cross,
            hint_transform=hint_transform,
            lower=lower_asymptote,
            df=df_tmp[df_tmp["hint"] == 0.0],
            models=test_models,
        ) if test_models else float("nan")

        midpoint_errors = compute_midpoint_errors_capability(
            beta=fit.beta,
            gamma=fit.gamma,
            delta=fit.delta,
            include_cross=fit.include_cross,
            individual_fits=individual_by_hint_all,
            hints=eval_hints,
            hint_transform=hint_transform,
        )

        row: dict[str, float] = {"n_models": float(n), "rms_train": rms_train, "rms_test": rms_test, "rms_all": rms_all, "rms_h0_all": rms_h0_all, "rms_h0_test": rms_h0_test}
        for h in eval_hints:
            row[f"midpoint_h_{h:.1f}"] = float(midpoint_errors.get(float(h), float("nan")))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["n_models"] = out["n_models"].astype(int)
    return out


def plot_model_sweep_capability(
    *,
    sweep_df: pd.DataFrame,
    label: str,
    eval_hints: list[float],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(sweep_df["n_models"], sweep_df["rms_train"], "o-", label="train", color="blue")
    ax.plot(sweep_df["n_models"], sweep_df["rms_test"], "s-", label="test", color="red")
    ax.plot(sweep_df["n_models"], sweep_df["rms_all"], "^-", label="all", color="green")
    ax.set_xlabel("number of train models")
    ax.set_ylabel("rms")
    ax.set_title("RMS vs number of train models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(eval_hints) - 1, 1)) for i, h in enumerate(eval_hints)}
    for h in eval_hints:
        col = f"midpoint_h_{h:.1f}"
        if col in sweep_df.columns:
            ax.plot(sweep_df["n_models"], sweep_df[col], "o-", label=f"h={h:.1f}", color=colors[h])
    ax.set_xlabel("number of train models")
    ax.set_ylabel("midpoint error (capability units)")
    ax.set_title("midpoint error vs number of train models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} (fitting joint scaling)", fontsize=12)
    plt.tight_layout()

    out_path = output_dir / "model_sweep.png"
    save_figure(fig, out_path)
    return out_path


def run_joint_scaling_plots_pc(
    *,
    base_folder: Path,
    baseline_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
    label: str,
    all_models: list[str],
    num_holdout_models: int,
    hint_fractions: list[float],
    eval_hints_for_sweep: list[float],
    include_cross: bool,
    lower_asymptote: float | None,
    hint_transform: str | Callable[[float], float],
    n_pcs: int,
    output_dir: Path,
    eci_file: Path | None = None,
    alpha_fixed: np.ndarray | None = None,
    pca_n_components: int | None = None,
) -> dict[str, object]:
    """Run the joint scaling analysis using capability=alpha·PC and write artifacts to `output_dir`."""
    from pca_helpers import compute_pc_scores, plot_component_weights_heatmap, plot_explained_variance
    from pc_joint_helpers import fit_joint_sigmoid_over_pcs

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir

    # Resolve hint transform (reuse the existing ones from this module)
    if isinstance(hint_transform, str):
        hint_transform_name = hint_transform
        if hint_transform_name not in HINT_TRANSFORMS:
            raise ValueError(f"Unknown hint_transform={hint_transform_name!r}. Choose one of {sorted(HINT_TRANSFORMS.keys())}")
        hint_transform_fn = HINT_TRANSFORMS[hint_transform_name]
    else:
        hint_transform_fn = hint_transform
        hint_transform_name = getattr(hint_transform_fn, "__name__", "custom")

    # PCA + PC scores
    n_components = int(pca_n_components) if pca_n_components is not None else max(int(n_pcs), 5)
    _pivot, pca, pc_scores_map = compute_pc_scores(baseline_folder=baseline_folder, n_components=n_components)
    plot_component_weights_heatmap(pca=pca, outfile=plots_dir / "pca_component_weights.png")
    plot_explained_variance(pca=pca, outfile=plots_dir / "pca_explained_variance.png")

    # Optional ECI map for split consistency / proxy metrics
    eci_map: dict[str, float] | None
    if eci_file is not None:
        eci_map = load_eci_map(eci_file)
    else:
        eci_map = None

    # Train/test split: by ECI if available, else by preliminary capability ordering
    if num_holdout_models < 0 or num_holdout_models > len(all_models):
        raise ValueError(f"num_holdout_models must be in [0, {len(all_models)}], got {num_holdout_models}")

    if eci_map is not None:
        models_sorted = sorted(all_models, key=lambda m: eci_map.get(m, float("-inf")))
    else:
        # Temporary ordering by PC-1 score (monotone in many cases); updated after fit.
        models_sorted = sorted(all_models, key=lambda m: float(pc_scores_map.get(m, np.zeros(1))[0]) if m in pc_scores_map else float("-inf"))

    holdout_set = set(models_sorted[-num_holdout_models:]) if num_holdout_models > 0 else set()
    train_set = set(models_sorted[:-num_holdout_models]) if num_holdout_models > 0 else set(all_models)

    # Fit alpha (unless fixed), beta/gamma/delta on train set
    df_raw = load_and_prepare_results_df_with_capability(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
        capability_map={m: 0.0 for m in all_models},  # placeholder, overwritten below
        train_models=train_set,
        eci_map=eci_map,
    )
    # Overwrite placeholder capability using fitted alpha
    fit = fit_joint_sigmoid_over_pcs(
        df=df_raw,
        pc_scores_map=pc_scores_map,
        n_pcs=int(n_pcs),
        fit_models=train_set,
        include_cross=include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        alpha_fixed=alpha_fixed,
    )

    capability_map = {m: float(np.dot(fit.alpha[: int(n_pcs)], pc_scores_map[m][: int(n_pcs)])) for m in all_models if m in pc_scores_map}

    df = load_and_prepare_results_df_with_capability(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
        capability_map=capability_map,
        train_models=train_set,
        eci_map=eci_map,
    )

    # RMS
    rms_train = compute_rms_capability_fit(
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        df=df,
        models=train_set,
    )
    rms_test = compute_rms_capability_fit(
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        df=df,
        models=holdout_set,
    ) if holdout_set else float("nan")
    rms_all = compute_rms_capability_fit(
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        df=df,
        models=None,
    )

    # Individual fits and midpoint errors (capability units)
    individual_by_hint_all = fit_individual_sigmoids_by_hint_capability(df, fit_models=None, lower=lower_asymptote)
    individual_by_hint_train = fit_individual_sigmoids_by_hint_capability(df, fit_models=train_set, lower=lower_asymptote)
    individual_by_model = fit_individual_sigmoids_by_model_hint_space(df, fit_models=None, lower=lower_asymptote)

    midpoint_errors_all = compute_midpoint_errors_capability(
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        individual_fits=individual_by_hint_all,
        hints=hint_fractions,
        hint_transform=hint_transform_fn,
    )
    midpoint_errors_train = compute_midpoint_errors_capability(
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        individual_fits=individual_by_hint_train,
        hints=hint_fractions,
        hint_transform=hint_transform_fn,
    )
    mean_midpoint_error_all = float(np.mean(list(midpoint_errors_all.values()))) if midpoint_errors_all else float("nan")
    mean_midpoint_error_train = float(np.mean(list(midpoint_errors_train.values()))) if midpoint_errors_train else float("nan")

    # Title equation string (human-readable, used in plot titles)
    alpha_str = ", ".join([f"{a:.3f}" for a in fit.alpha[: int(n_pcs)].tolist()])
    if include_cross and fit.gamma is not None:
        eq = f"σ(α·PC + β·h + γ·(α·PC)h + δ), α=[{alpha_str}], β={fit.beta:.3f}, γ={fit.gamma:.3f}, δ={fit.delta:.3f}"
    else:
        eq = f"σ(α·PC + β·h + δ), α=[{alpha_str}], β={fit.beta:.3f}, δ={fit.delta:.3f}"
    if lower_asymptote is not None:
        eq = f"{lower_asymptote:.2f} + {1-lower_asymptote:.2f}·" + eq

    plot_accuracy_vs_capability_by_hint(
        df=df,
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        label=label,
        title_equation=eq,
        output_dir=plots_dir,
        x_label="pc_capability",
        out_name="accuracy_vs_pc_capability_by_hint.png",
    )
    plot_individual_fits_by_hint_capability(
        df=df,
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        individual_by_hint_all=individual_by_hint_all,
        individual_by_hint_train=individual_by_hint_train,
        label=label,
        title_equation=eq,
        output_dir=plots_dir,
        x_label="pc_capability",
    )
    plot_accuracy_vs_hint_by_model_capability(
        df=df,
        capability_map=capability_map,
        beta=fit.beta,
        gamma=fit.gamma,
        delta=fit.delta,
        include_cross=fit.include_cross,
        hint_transform=hint_transform_fn,
        lower=lower_asymptote,
        individual_by_model=individual_by_model,
        label=label,
        title_equation=eq,
        output_dir=plots_dir,
    )

    # Model sweep (capability ordering)
    models_sorted_by_cap = sorted(df["model"].unique().tolist(), key=lambda m: capability_map.get(m, float("-inf")))
    sweep_df = run_model_sweep_capability(
        df=df,
        pc_scores_map=pc_scores_map,
        models_sorted=models_sorted_by_cap,
        hint_fractions=hint_fractions,
        hint_transform=hint_transform_fn,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
        n_pcs=int(n_pcs),
        alpha_fixed=(np.asarray(alpha_fixed, dtype=float) if alpha_fixed is not None else None),
        eval_hints=eval_hints_for_sweep,
    )
    plot_model_sweep_capability(sweep_df=sweep_df, label=label, eval_hints=eval_hints_for_sweep, output_dir=plots_dir)

    config_resolved: dict[str, object] = {
        "output_dir": str(output_dir),
        "base_folder": str(base_folder),
        "baseline_folder": str(baseline_folder),
        "eci_file": str(eci_file) if eci_file is not None else None,
        "eval_name": eval_name,
        "solver": solver,
        "condition": condition,
        "label": label,
        "all_models": all_models,
        "num_holdout_models": int(num_holdout_models),
        "train_models": sorted(train_set, key=lambda m: (eci_map.get(m, float("-inf")) if eci_map is not None else capability_map.get(m, float("-inf")))),
        "holdout_models": sorted(holdout_set, key=lambda m: (eci_map.get(m, float("-inf")) if eci_map is not None else capability_map.get(m, float("-inf")))),
        "hint_fractions": hint_fractions,
        "eval_hints_for_sweep": eval_hints_for_sweep,
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "hint_transform": hint_transform_name,
        "n_pcs": int(n_pcs),
        "alpha_fixed": (alpha_fixed.tolist() if alpha_fixed is not None else None),
    }

    metrics: dict[str, object] = {
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "rms_train": float(rms_train),
        "rms_test": float(rms_test),
        "rms_all": float(rms_all),
        "mse_train": float(rms_train) ** 2,
        "mse_test": float(rms_test) ** 2 if not math.isnan(float(rms_test)) else float("nan"),
        "mse_all": float(rms_all) ** 2,
        "mean_midpoint_error_all": mean_midpoint_error_all,
        "mean_midpoint_error_train": mean_midpoint_error_train,
        "n_train_models": int(len(train_set)),
        "n_test_models": int(len(holdout_set)),
        "alpha": [float(x) for x in fit.alpha[: int(n_pcs)].tolist()],
        "beta": float(fit.beta),
        "gamma": float(fit.gamma) if fit.gamma is not None else None,
        "delta": float(fit.delta),
        "equation_text": eq,
        "explained_variance_ratio": [float(x) for x in np.asarray(pca.explained_variance_ratio, dtype=float).tolist()],
        "config": config_resolved,
    }

    write_json(output_dir / "config_resolved.json", config_resolved)
    write_json(output_dir / "metrics.json", metrics)
    sweep_df.to_csv(output_dir / "model_sweep.csv", index=False)
    return metrics


def _fit_affine_capability_to_eci(
    *,
    capability_map: dict[str, float],
    eci_map: dict[str, float],
    models: set[str],
) -> tuple[float, float]:
    """Fit eci ≈ a*capability + b via least squares on the provided model set."""
    xs: list[float] = []
    ys: list[float] = []
    for m in models:
        if m in capability_map and m in eci_map:
            x = capability_map[m]
            y = eci_map[m]
            if x is None or y is None:
                continue
            if isinstance(x, float) and np.isnan(x):
                continue
            if isinstance(y, float) and np.isnan(y):
                continue
            xs.append(float(x))
            ys.append(float(y))
    if len(xs) < 2:
        raise ValueError(f"Need at least 2 models to calibrate capability->ECI, got {len(xs)}")
    X = np.stack([np.asarray(xs, dtype=float), np.ones(len(xs), dtype=float)], axis=1)  # (N,2)
    y = np.asarray(ys, dtype=float)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    return float(a), float(b)


def _joint_midpoint_capability(
    *,
    hint: float,
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
) -> float | None:
    """Midpoint in capability units (x where z=0)."""
    h_t = float(hint_transform(float(hint)))
    if include_cross and gamma is not None:
        denom = 1.0 + float(gamma) * h_t
        if abs(denom) <= 1e-8:
            return None
        return float((-float(beta) * h_t - float(delta)) / denom)
    return float(-float(beta) * h_t - float(delta))


def _mean_abs_midpoint_error_proxy_eci_units(
    *,
    hints: list[float],
    eci_midpoints: dict[float, float],
    beta: float,
    gamma: float | None,
    delta: float,
    include_cross: bool,
    hint_transform: Callable[[float], float],
    a: float,
    b: float,
) -> float:
    errs: list[float] = []
    for h in hints:
        h = float(h)
        if h not in eci_midpoints:
            continue
        cap_mid = _joint_midpoint_capability(
            hint=h,
            beta=beta,
            gamma=gamma,
            delta=delta,
            include_cross=include_cross,
            hint_transform=hint_transform,
        )
        if cap_mid is None:
            continue
        eci_pred = float(a * cap_mid + b)
        errs.append(abs(eci_pred - float(eci_midpoints[h])))
    return float(np.mean(errs)) if errs else float("nan")


def plot_comparison_fits_by_hint(
    *,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    indiv_all_by_hint: dict[float, dict],
    eci_joint_params: list[float],
    eci_joint_include_cross: bool,
    eci_joint_lower: float | None,
    hint_transform: Callable[[float], float],
    pc_methods: list[dict],
    eval_hints: list[float],
    label: str,
    output_dir: Path,
    out_name: str = "comparison_fits_by_hint.png",
) -> Path:
    """Per-hint grid comparing individual-all, ECI joint, and one line per PC method.

    All curves are plotted on the ECI x-axis.  For ECI joint the curve is a
    smooth sigmoid.  For each PC method the prediction is evaluated at each
    model's actual (ECI, capability) pair and the points are connected sorted
    by ECI, giving an approximate curve that need not be a perfect sigmoid.

    Each entry in `pc_methods` must contain:
      label, capability_map, beta, gamma, delta, include_cross, lower
    """
    hints = sorted(eval_hints)
    n_hints = len(hints)
    n_cols = 7
    n_rows = (n_hints + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.2 * n_rows))
    axes_flat = np.array(axes).flatten()

    eci_arr = np.array(sorted(eci_map.values()), dtype=float)
    eci_range = np.linspace(eci_arr.min() - 5, eci_arr.max() + 5, 120)

    # Fixed colours per method (shared palette, no recycling)
    indiv_color = "black"
    eci_joint_color = _ECI_JOINT_COLOR

    # Data scatter colour by hint (same viridis ramp as the rest of the codebase)
    all_hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    scatter_colors = {h: cmap(i / max(len(all_hints) - 1, 1)) for i, h in enumerate(all_hints)}

    L_eci = float(eci_joint_lower) if eci_joint_lower is not None else 0.0
    params = [float(p) for p in eci_joint_params]

    for idx, hint in enumerate(hints):
        ax = axes_flat[idx]
        hint_df = df[df["hint"] == hint].sort_values("eci")

        # Data scatter
        ax.scatter(
            hint_df["eci"], hint_df["accuracy"],
            color=scatter_colors.get(float(hint), "gray"),
            alpha=0.6, s=25, zorder=3,
        )

        # Individual all (ground truth sigmoid in ECI space)
        if float(hint) in indiv_all_by_hint:
            fit = indiv_all_by_hint[float(hint)]
            y_indiv = [float(fit["predict"](c)) for c in eci_range]
            ax.plot(eci_range, y_indiv, "-", color=indiv_color, linewidth=2,
                    label="indiv (all)", zorder=5)
            ax.axvline(float(fit["midpoint"]), color=indiv_color,
                       linestyle=":", alpha=0.35, linewidth=1)

        # ECI joint (smooth sigmoid)
        h_t = float(hint_transform(float(hint)))
        if eci_joint_include_cross:
            α_e, β_e, γ_e, δ_e = params
            y_eci = [
                L_eci + (1.0 - L_eci) / (1.0 + math.exp(-(α_e * c + β_e * h_t + γ_e * c * h_t + δ_e)))
                for c in eci_range
            ]
        else:
            α_e, β_e, δ_e = params
            y_eci = [
                L_eci + (1.0 - L_eci) / (1.0 + math.exp(-(α_e * c + β_e * h_t + δ_e)))
                for c in eci_range
            ]
        ax.plot(eci_range, y_eci, "--", color=eci_joint_color, linewidth=2,
                label="ECI joint", zorder=4)

        # PC methods – evaluate at each model's actual (ECI, capability) pair
        for j, pc_m in enumerate(pc_methods):
            cap_map = pc_m["capability_map"]
            pts: list[tuple[float, float]] = []
            for m, eci_v in eci_map.items():
                if m in cap_map:
                    pred = _predict_from_capability(
                        capability=float(cap_map[m]),
                        hint=float(hint),
                        beta=float(pc_m["beta"]),
                        gamma=(float(pc_m["gamma"]) if pc_m.get("gamma") is not None else None),
                        delta=float(pc_m["delta"]),
                        include_cross=bool(pc_m["include_cross"]),
                        hint_transform=hint_transform,
                        lower=pc_m.get("lower"),
                    )
                    pts.append((float(eci_v), pred))
            pts.sort(key=lambda t: t[0])
            if pts:
                xs, ys = zip(*pts)
                color = _PC_METHOD_COLORS[j]
                ax.plot(xs, ys, "-", color=color, linewidth=1.5,
                        label=pc_m["label"], alpha=0.85, zorder=4)

        ax.set_title(f"h = {hint:.2f}", fontsize=9)
        ax.set_xlabel("ECI", fontsize=7)
        ax.set_ylabel("accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for j in range(n_hints, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{label}: comparison fits by hint", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / out_name
    save_figure(fig, out_path)
    return out_path


def _pc_label(n: int) -> str:
    """Human-readable label for a method using n principal components."""
    return "+".join(f"PC{i + 1}" for i in range(n))


def plot_comparison_model_sweep(
    *,
    method_sweeps: list[dict],
    label: str,
    eval_hints: list[float],
    output_dir: Path,
    n_models_range: tuple[int, int] | None = None,
) -> None:
    """RMS and midpoint-error comparison vs number of train models.

    Each entry in `method_sweeps` must have:
      name        – display name for the method
      sweep_df    – DataFrame with columns n_models, rms_all, midpoint_h_X.X ...
      a_eci_scale – multiply midpoint columns by this to convert to ECI units
                    (1.0 for ECI since it is already in ECI units)

    Args:
      n_models_range: optional (min, max) inclusive range of n_models to plot.
                      Rows outside this range are dropped from the x-axis.
                      None means show all rows.
    """
    method_colors = _COMPARISON_COLORS

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if n_models_range is None:
            return df
        lo, hi = int(n_models_range[0]), int(n_models_range[1])
        return df[(df["n_models"] >= lo) & (df["n_models"] <= hi)]

    # ---- Figure 1: RMS 2x2 grid ----
    # Rows: all models / test models only
    # Cols: all hints / hint=0 only
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (axes[0, 0], "rms_all",     "all models, all hints"),
        (axes[0, 1], "rms_h0_all",  "all models, hint = 0 only"),
        (axes[1, 0], "rms_test",    "test models only, all hints"),
        (axes[1, 1], "rms_h0_test", "test models only, hint = 0 only"),
    ]
    for ax, col, subtitle in panels:
        for i, m in enumerate(method_sweeps):
            df = _filter(m["sweep_df"])
            if col in df.columns:
                ax.plot(df["n_models"], df[col], "o-", label=m["name"], color=method_colors[i])
        ax.set_xlabel("number of train models")
        ax.set_ylabel("RMS")
        ax.set_title(subtitle)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label}: RMS vs number of train models", fontsize=13)
    plt.tight_layout()
    save_figure(fig, output_dir / "comparison_model_sweep_rms.png")

    # ---- Figure 2: midpoint error vs n_train_models ----
    # ECI midpoints are in ECI units; PC midpoints are in native capability units.
    # We scale PC midpoints by `a_eci_scale` (slope of affine cap→ECI calibration)
    # so all lines share ECI units.
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(method_sweeps):
        df = _filter(m["sweep_df"])
        a = float(m.get("a_eci_scale", 1.0))
        mid_cols = [f"midpoint_h_{h:.1f}" for h in eval_hints if f"midpoint_h_{h:.1f}" in df.columns]
        if not mid_cols:
            continue
        mean_mid = df[mid_cols].mean(axis=1) * a
        ax.plot(df["n_models"], mean_mid, "o-", label=m["name"], color=method_colors[i])
    ax.set_xlabel("number of train models")
    ax.set_ylabel("mean midpoint error (ECI units)")
    ax.set_title(f"{label}: midpoint error vs number of train models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir / "comparison_model_sweep_midpoint.png")


def compare_capability_approaches(
    *,
    base_folder: Path,
    baseline_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    label: str,
    all_models: list[str],
    num_holdout_models: int,
    hint_fractions: list[float],
    eval_hints_for_sweep: list[float],
    include_cross: bool,
    lower_asymptote: float | None,
    hint_transform: str | Callable[[float], float],
    output_dir: Path,
    pc_ns: list[int] = [2, 3],
    sweep_n_models_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Run ECI vs PC2 vs PC3 and write a comparison report + per-method artifacts.

    Args:
      sweep_n_models_range: optional (min, max) inclusive range of n_train_models
                            shown on the x-axis of the model-sweep plots.
                            None means show all values from 5 to len(all_models).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run ECI baseline method (writes its own artifacts) ----
    eci_out = output_dir / "eci"
    eci_metrics = run_joint_scaling_plots(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        label=f"{label} (ECI)",
        all_models=all_models,
        num_holdout_models=num_holdout_models,
        hint_fractions=hint_fractions,
        eval_hints_for_sweep=eval_hints_for_sweep,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
        hint_transform=hint_transform,
        output_dir=eci_out,
    )

    # ---- Ground truth midpoints in ECI units (current pipeline definition) ----
    if isinstance(hint_transform, str):
        hint_transform_fn = HINT_TRANSFORMS[hint_transform]
    else:
        hint_transform_fn = hint_transform

    eci_map = load_eci_map(eci_file)
    models_sorted_by_eci = sorted(all_models, key=lambda m: eci_map.get(m, float("-inf")))
    holdout_set = set(models_sorted_by_eci[-num_holdout_models:]) if num_holdout_models > 0 else set()
    train_set = set(models_sorted_by_eci[:-num_holdout_models]) if num_holdout_models > 0 else set(all_models)

    df_eci = load_and_prepare_results_df(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
        eci_map=eci_map,
        train_models=train_set,
    )
    indiv_eci_all = fit_individual_sigmoids_by_hint(df_eci, eci_map, fit_models=None, lower=lower_asymptote)
    indiv_eci_train = fit_individual_sigmoids_by_hint(df_eci, eci_map, fit_models=train_set, lower=lower_asymptote)
    indiv_eci_test = fit_individual_sigmoids_by_hint(df_eci, eci_map, fit_models=holdout_set, lower=lower_asymptote) if holdout_set else {}

    eci_mid_all = {float(h): float(v["midpoint"]) for h, v in indiv_eci_all.items()}
    eci_mid_train = {float(h): float(v["midpoint"]) for h, v in indiv_eci_train.items()}
    eci_mid_test = {float(h): float(v["midpoint"]) for h, v in indiv_eci_test.items()}

    # ---- Run PC methods ----
    pc_results: list[dict[str, object]] = []
    pc_panels: list[dict] = []
    for n in pc_ns:
        pc_out = output_dir / f"pc{int(n)}"
        pc_label = _pc_label(int(n))
        pc_metrics = run_joint_scaling_plots_pc(
            base_folder=base_folder,
            baseline_folder=baseline_folder,
            eci_file=eci_file,
            eval_name=eval_name,
            solver=solver,
            condition=condition,
            label=f"{label} ({pc_label})",
            all_models=all_models,
            num_holdout_models=num_holdout_models,
            hint_fractions=hint_fractions,
            eval_hints_for_sweep=eval_hints_for_sweep,
            include_cross=include_cross,
            lower_asymptote=lower_asymptote,
            hint_transform=hint_transform,
            n_pcs=int(n),
            output_dir=pc_out,
        )

        # Proxy midpoint error in ECI units via affine calibration (capability -> ECI)
        alpha = np.asarray(pc_metrics["alpha"], dtype=float)
        # Recompute capability map from saved alpha (consistent with pca_helpers + saved run).
        from pca_helpers import compute_pc_scores

        _pivot, _pca, pc_scores_map = compute_pc_scores(baseline_folder=baseline_folder, n_components=max(int(n), 5))
        capability_map = {m: float(np.dot(alpha[: int(n)], pc_scores_map[m][: int(n)])) for m in all_models if m in pc_scores_map}

        # Build df with PC capability column for the combined grid plot.
        df_pc = df_eci.drop(columns=["eci"], errors="ignore").copy()
        df_pc["capability"] = df_pc["model"].map(capability_map)
        df_pc = df_pc.dropna(subset=["capability"])
        alpha_str = ", ".join(f"{a_i:.3f}" for a_i in alpha[: int(n)].tolist())
        if bool(pc_metrics["include_cross"]) and pc_metrics.get("gamma") is not None:
            eq_short = f"σ(α·PC+β·h+γ·S·h+δ), α=[{alpha_str}]"
        else:
            eq_short = f"σ(α·PC+β·h+δ), α=[{alpha_str}]"
        pc_panels.append({
            "df": df_pc,
            "capability_map": capability_map,
            "beta": float(pc_metrics["beta"]),
            "gamma": float(pc_metrics["gamma"]) if pc_metrics.get("gamma") is not None else None,
            "delta": float(pc_metrics["delta"]),
            "include_cross": bool(pc_metrics["include_cross"]),
            "hint_transform": hint_transform_fn,
            "lower": lower_asymptote,
            "method_label": pc_label,
            "title_equation": eq_short,
        })

        a, b = _fit_affine_capability_to_eci(capability_map=capability_map, eci_map=eci_map, models=train_set)

        proxy_all = _mean_abs_midpoint_error_proxy_eci_units(
            hints=hint_fractions,
            eci_midpoints=eci_mid_all,
            beta=float(pc_metrics["beta"]),
            gamma=(float(pc_metrics["gamma"]) if pc_metrics.get("gamma") is not None else None),
            delta=float(pc_metrics["delta"]),
            include_cross=bool(pc_metrics["include_cross"]),
            hint_transform=hint_transform_fn,
            a=a,
            b=b,
        )
        proxy_train = _mean_abs_midpoint_error_proxy_eci_units(
            hints=hint_fractions,
            eci_midpoints=eci_mid_train,
            beta=float(pc_metrics["beta"]),
            gamma=(float(pc_metrics["gamma"]) if pc_metrics.get("gamma") is not None else None),
            delta=float(pc_metrics["delta"]),
            include_cross=bool(pc_metrics["include_cross"]),
            hint_transform=hint_transform_fn,
            a=a,
            b=b,
        )
        proxy_test = _mean_abs_midpoint_error_proxy_eci_units(
            hints=hint_fractions,
            eci_midpoints=eci_mid_test,
            beta=float(pc_metrics["beta"]),
            gamma=(float(pc_metrics["gamma"]) if pc_metrics.get("gamma") is not None else None),
            delta=float(pc_metrics["delta"]),
            include_cross=bool(pc_metrics["include_cross"]),
            hint_transform=hint_transform_fn,
            a=a,
            b=b,
        )

        pc_results.append(
            {
                "method": pc_label,
                "n_pcs": int(n),
                "output_dir": str(pc_out),
                "rms_train": float(pc_metrics["rms_train"]),
                "rms_test": float(pc_metrics["rms_test"]),
                "rms_all": float(pc_metrics["rms_all"]),
                "mse_train": float(pc_metrics["mse_train"]),
                "mse_test": float(pc_metrics["mse_test"]),
                "mse_all": float(pc_metrics["mse_all"]),
                "mean_midpoint_error_all_native": float(pc_metrics["mean_midpoint_error_all"]),
                "mean_midpoint_error_train_native": float(pc_metrics["mean_midpoint_error_train"]),
                "mean_midpoint_error_proxy_eci_units_all": float(proxy_all),
                "mean_midpoint_error_proxy_eci_units_train": float(proxy_train),
                "mean_midpoint_error_proxy_eci_units_test": float(proxy_test),
                "capability_to_eci_affine": {"a": float(a), "b": float(b)},
            }
        )

    # ---- Combined accuracy-vs-capability grid (all PC methods in one image) ----
    if pc_panels:
        plot_all_pc_accuracy_vs_capability(
            pc_panels=pc_panels,
            label=label,
            output_dir=output_dir,
        )

    # ---- Comparison fits by hint (individual-all + ECI joint + each PC) ----
    if pc_panels:
        plot_comparison_fits_by_hint(
            df=df_eci,
            eci_map=eci_map,
            indiv_all_by_hint=indiv_eci_all,
            eci_joint_params=list(eci_metrics["joint_params"]),
            eci_joint_include_cross=bool(eci_metrics["include_cross"]),
            eci_joint_lower=eci_metrics.get("lower_asymptote"),
            hint_transform=hint_transform_fn,
            pc_methods=[
                {
                    "label": p["method_label"],
                    "capability_map": p["capability_map"],
                    "beta": p["beta"],
                    "gamma": p["gamma"],
                    "delta": p["delta"],
                    "include_cross": p["include_cross"],
                    "lower": p["lower"],
                }
                for p in pc_panels
            ],
            eval_hints=sorted(df_eci["hint"].unique().tolist()),
            label=label,
            output_dir=output_dir,
        )

    # ---- Collect sweep DataFrames for comparison model-sweep plots ----
    # ECI sweep CSV was saved inside run_joint_scaling_plots
    eci_sweep_csv = eci_out / "model_sweep.csv"
    eci_sweep_df = pd.read_csv(eci_sweep_csv) if eci_sweep_csv.exists() else None

    pc_sweep_entries: list[dict] = []
    for pc_res in pc_results:
        pc_sweep_csv = Path(pc_res["output_dir"]) / "model_sweep.csv"
        if pc_sweep_csv.exists():
            pc_sweep_df = pd.read_csv(pc_sweep_csv)
            a = float(pc_res["capability_to_eci_affine"]["a"])
            pc_sweep_entries.append(
                {"name": pc_res["method"], "sweep_df": pc_sweep_df, "a_eci_scale": a}
            )

    method_sweeps: list[dict] = []
    if eci_sweep_df is not None:
        method_sweeps.append({"name": "ECI", "sweep_df": eci_sweep_df, "a_eci_scale": 1.0})
    method_sweeps.extend(pc_sweep_entries)

    if method_sweeps:
        plot_comparison_model_sweep(
            method_sweeps=method_sweeps,
            label=label,
            eval_hints=eval_hints_for_sweep,
            output_dir=output_dir,
            n_models_range=sweep_n_models_range,
        )

    # ---- Build comparison table ----
    eci_row = {
        "method": "ECI",
        "output_dir": str(eci_out),
        "rms_train": float(eci_metrics["rms_train"]),
        "rms_test": float(eci_metrics["rms_test"]),
        "rms_all": float(eci_metrics["rms_all"]),
        "mse_train": float(eci_metrics["rms_train"]) ** 2,
        "mse_test": float(eci_metrics["rms_test"]) ** 2 if not math.isnan(float(eci_metrics["rms_test"])) else float("nan"),
        "mse_all": float(eci_metrics["rms_all"]) ** 2,
        "mean_midpoint_error_all_native": float(eci_metrics["mean_midpoint_error_all"]),
        "mean_midpoint_error_train_native": float(eci_metrics["mean_midpoint_error_train"]),
        # For ECI, proxy == native (already in ECI units).
        "mean_midpoint_error_proxy_eci_units_all": float(eci_metrics["mean_midpoint_error_all"]),
        "mean_midpoint_error_proxy_eci_units_train": float(eci_metrics["mean_midpoint_error_train"]),
        "mean_midpoint_error_proxy_eci_units_test": float(eci_metrics["mean_midpoint_error_test"]),
    }
    rows = [eci_row, *pc_results]
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_dir / "comparison.csv", index=False)
    write_json(output_dir / "comparison.json", rows)

    # ---- Summary plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = comp_df["method"].tolist()

    ax = axes[0]
    ax.bar(methods, comp_df["mse_all"].tolist())
    ax.set_title("MSE (all points)")
    ax.set_ylabel("MSE")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(methods, comp_df["mean_midpoint_error_proxy_eci_units_all"].tolist())
    ax.set_title("Midpoint error proxy (ECI units, all)")
    ax.set_ylabel("mean |Δ midpoint|")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{label}: ECI vs PC methods", fontsize=12)
    plt.tight_layout()
    save_figure(fig, output_dir / "comparison_summary.png")

    result: dict[str, object] = {
        "output_dir": str(output_dir),
        "eci_dir": str(eci_out),
        "pc_dirs": {r["method"]: r["output_dir"] for r in pc_results},
        "comparison_csv": str(output_dir / "comparison.csv"),
        "comparison_json": str(output_dir / "comparison.json"),
        "comparison_summary_plot": str(output_dir / "comparison_summary.png"),
    }
    write_json(output_dir / "config_resolved.json", {
        "base_folder": str(base_folder),
        "baseline_folder": str(baseline_folder),
        "eci_file": str(eci_file),
        "eval_name": eval_name,
        "solver": solver,
        "condition": condition,
        "label": label,
        "all_models": all_models,
        "num_holdout_models": int(num_holdout_models),
        "hint_fractions": hint_fractions,
        "eval_hints_for_sweep": eval_hints_for_sweep,
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "hint_transform": (hint_transform if isinstance(hint_transform, str) else getattr(hint_transform, "__name__", "custom")),
        "pc_ns": [int(n) for n in pc_ns],
        "output_dir": str(output_dir),
    })
    write_json(output_dir / "metrics.json", result)
    return result

