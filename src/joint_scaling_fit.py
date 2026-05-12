from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.sigmoid_fits import clip_accuracy_for_logit, fit_bounded_sigmoid_1d, sigmoid_curve


def joint_sigmoid(z: np.ndarray | float, *, lower: float | None) -> np.ndarray | float:
    lower_value = 0.0 if lower is None else float(lower)
    sigmoid = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    return lower_value + (1.0 - lower_value) * sigmoid


def fit_linear_init(
    *,
    capability: np.ndarray,
    hint: np.ndarray,
    accuracy: np.ndarray,
    include_cross: bool,
    lower: float | None,
) -> np.ndarray:
    clipped = clip_accuracy_for_logit(accuracy, lower=lower)
    target = np.log(clipped / (1.0 - clipped))
    if include_cross:
        design = np.column_stack([capability, hint, capability * hint, np.ones_like(capability)])
    else:
        design = np.column_stack([capability, hint, np.ones_like(capability)])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(params, dtype=float)


def fit_joint_sigmoid_model(
    *,
    df: pd.DataFrame,
    fit_models: set[str],
    x_field: str,
    include_cross: bool,
    lower: float | None,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    train_df = df[df["model"].isin(fit_models)].copy()
    if train_df.empty:
        raise ValueError("No training rows available for the joint fit.")

    capability = train_df[x_field].to_numpy(dtype=float)
    hint = train_df["hint_fraction"].to_numpy(dtype=float)
    accuracy = train_df["accuracy"].to_numpy(dtype=float)
    init = fit_linear_init(
        capability=capability,
        hint=hint,
        accuracy=accuracy,
        include_cross=include_cross,
        lower=lower,
    )

    def predict_from_params(params: np.ndarray, x_capability: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
        if include_cross:
            alpha, beta, gamma, delta = params
            z = alpha * x_capability + beta * x_hint + gamma * x_capability * x_hint + delta
        else:
            alpha, beta, delta = params
            z = alpha * x_capability + beta * x_hint + delta
        return np.asarray(joint_sigmoid(z, lower=lower), dtype=float)

    def objective(params: np.ndarray) -> float:
        predictions = predict_from_params(params, capability, hint)
        residuals = predictions - accuracy
        return float(np.mean(residuals * residuals))

    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    params = np.asarray(result.x, dtype=float)

    def predict(single_capability: float, single_hint: float) -> float:
        prediction = predict_from_params(
            params,
            np.asarray([single_capability], dtype=float),
            np.asarray([single_hint], dtype=float),
        )[0]
        return float(prediction)

    return {
        "params": params,
        "include_cross": bool(include_cross),
        "lower": lower,
        "x_field": str(x_field),
        "predict": predict,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "optimizer_nit": int(getattr(result, "nit", -1)),
    }


def fit_individual_sigmoids_by_hint(
    *,
    df: pd.DataFrame,
    x_field: str,
    fit_models: set[str] | None,
    lower: float | None,
) -> dict[float, dict[str, Any]]:
    results: dict[float, dict[str, Any]] = {}
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    for hint_fraction in hint_fractions:
        hint_df = df[df["hint_fraction"] == hint_fraction]
        if fit_models is not None:
            hint_df = hint_df[hint_df["model"].isin(fit_models)]
        fit = fit_bounded_sigmoid_1d(
            xs=hint_df[x_field].to_numpy(dtype=float),
            ys=hint_df["accuracy"].to_numpy(dtype=float),
            lower=lower,
        )
        if fit is not None:
            results[float(hint_fraction)] = fit
    return results


def fit_individual_sigmoids_by_model(
    *,
    df: pd.DataFrame,
    fit_models: set[str] | None,
    lower: float | None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    model_names = sorted(set(df["model"].tolist()) if fit_models is None else fit_models)
    for model in model_names:
        model_df = df[df["model"] == model]
        fit = fit_bounded_sigmoid_1d(
            xs=model_df["hint_fraction"].to_numpy(dtype=float),
            ys=model_df["accuracy"].to_numpy(dtype=float),
            lower=lower,
        )
        if fit is not None:
            results[str(model)] = fit
    return results


def compute_rms_joint(
    *,
    joint_result: dict[str, Any],
    df: pd.DataFrame,
    x_field: str,
    models: set[str] | None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if eval_df.empty:
        return float("nan")
    predictions = np.asarray(
        [
            joint_result["predict"](float(row[x_field]), float(row["hint_fraction"]))
            for _, row in eval_df.iterrows()
        ],
        dtype=float,
    )
    actual = eval_df["accuracy"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((actual - predictions) ** 2)))


def compute_rms_individual_by_hint(
    *,
    individual_by_hint: dict[float, dict[str, Any]],
    df: pd.DataFrame,
    x_field: str,
    models: set[str] | None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if eval_df.empty:
        return float("nan")
    predictions: list[float] = []
    actual: list[float] = []
    for _, row in eval_df.iterrows():
        hint_fraction = float(row["hint_fraction"])
        fit = individual_by_hint.get(hint_fraction)
        if fit is None:
            continue
        predictions.append(float(fit["predict"](float(row[x_field]))))
        actual.append(float(row["accuracy"]))
    if not predictions:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predictions)) ** 2)))


def compute_midpoint_errors(
    *,
    joint_result: dict[str, Any],
    individual_fits: dict[float, dict[str, Any]],
    hint_fractions: list[float],
) -> dict[float, float]:
    params = np.asarray(joint_result["params"], dtype=float)
    errors: dict[float, float] = {}
    for hint_fraction in hint_fractions:
        fit = individual_fits.get(float(hint_fraction))
        if fit is None:
            continue
        if joint_result["include_cross"]:
            alpha, beta, gamma, delta = params
            denom = alpha + gamma * float(hint_fraction)
            if abs(denom) <= 1e-8:
                continue
            midpoint = (-beta * float(hint_fraction) - delta) / denom
        else:
            alpha, beta, delta = params
            if abs(alpha) <= 1e-8:
                continue
            midpoint = (-beta * float(hint_fraction) - delta) / alpha
        errors[float(hint_fraction)] = abs(float(midpoint) - float(fit["midpoint"]))
    return errors


def format_joint_equation(joint_result: dict[str, Any]) -> str:
    params = np.asarray(joint_result["params"], dtype=float)
    lower = joint_result.get("lower")
    if joint_result["include_cross"]:
        alpha, beta, gamma, delta = params
        expr = f"σ({alpha:.3f}·C + {beta:.3f}·h + {gamma:.3f}·C·h + {delta:.2f})"
    else:
        alpha, beta, delta = params
        expr = f"σ({alpha:.3f}·C + {beta:.3f}·h + {delta:.2f})"
    if lower is not None:
        lower_value = float(lower)
        return f"{lower_value:.2f} + {1.0 - lower_value:.2f}·{expr}"
    return expr


def build_joint_scaling_df(
    *,
    base_rows: list[dict[str, Any]],
    x_map: dict[str, float],
    x_field: str,
    train_models: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        model = str(row["model"])
        if model not in x_map:
            continue
        rows.append(
            {
                "model": model,
                "hint_fraction": float(row["hint_fraction"]),
                "accuracy": float(row["accuracy"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                x_field: float(x_map[model]),
                "split": "train" if model in train_models else "test",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No rows available for joint scaling after merging {x_field} values."
        )
    return df


def run_joint_model_sweep(
    *,
    df: pd.DataFrame,
    x_field: str,
    models_sorted_by_x: list[str],
    include_cross: bool,
    lower_asymptote: float | None,
    n_train_min: int | None = None,
    n_train_max: int | None = None,
) -> pd.DataFrame:
    sweep_hint_fraction = 0.0
    individual_by_hint_all = fit_individual_sigmoids_by_hint(
        df=df,
        x_field=x_field,
        fit_models=None,
        lower=lower_asymptote,
    )
    df_hint0 = df[df["hint_fraction"] == sweep_hint_fraction].copy()

    min_n_train = 3 if n_train_min is None else int(n_train_min)
    max_n_train = len(models_sorted_by_x) if n_train_max is None else int(n_train_max)
    if min_n_train < 3:
        raise ValueError(f"n_train_min must be >= 3, got {min_n_train}")
    if max_n_train > len(models_sorted_by_x):
        raise ValueError(
            f"n_train_max ({max_n_train}) cannot exceed number of models "
            f"({len(models_sorted_by_x)})"
        )
    if min_n_train > max_n_train:
        raise ValueError(
            f"n_train_min ({min_n_train}) cannot exceed n_train_max ({max_n_train})"
        )

    rows: list[dict[str, float]] = []
    for n_models in range(min_n_train, max_n_train + 1):
        train_models = set(models_sorted_by_x[:n_models])
        test_models = set(models_sorted_by_x[n_models:])

        joint_result = fit_joint_sigmoid_model(
            df=df,
            fit_models=train_models,
            x_field=x_field,
            include_cross=include_cross,
            lower=lower_asymptote,
        )
        individual_by_hint_train = fit_individual_sigmoids_by_hint(
            df=df,
            x_field=x_field,
            fit_models=train_models,
            lower=lower_asymptote,
        )

        midpoint_errors_joint = compute_midpoint_errors(
            joint_result=joint_result,
            individual_fits=individual_by_hint_all,
            hint_fractions=[sweep_hint_fraction],
        )
        midpoint_errors_individual: dict[float, float] = {}
        all_fit = individual_by_hint_all.get(sweep_hint_fraction)
        train_fit = individual_by_hint_train.get(sweep_hint_fraction)
        if all_fit is not None and train_fit is not None:
            midpoint_errors_individual[sweep_hint_fraction] = abs(
                float(train_fit["midpoint"]) - float(all_fit["midpoint"])
            )

        row: dict[str, float] = {
            "n_models": float(n_models),
            "rms_h0_test": compute_rms_joint(
                joint_result=joint_result,
                df=df_hint0,
                x_field=x_field,
                models=test_models,
            ) if test_models else float("nan"),
            "rms_indiv_h0_test": compute_rms_individual_by_hint(
                individual_by_hint=individual_by_hint_train,
                df=df_hint0,
                x_field=x_field,
                models=test_models,
            ) if test_models else float("nan"),
            "rms_indiv_allfit_h0_test": compute_rms_individual_by_hint(
                individual_by_hint=individual_by_hint_all,
                df=df_hint0,
                x_field=x_field,
                models=test_models,
            ) if test_models else float("nan"),
        }
        row["delta_rms_h0_test"] = row["rms_h0_test"] - row["rms_indiv_h0_test"]
        key = f"midpoint_joint_h_{sweep_hint_fraction:.1f}"
        indiv_key = f"midpoint_indiv_h_{sweep_hint_fraction:.1f}"
        delta_key = f"delta_midpoint_h_{sweep_hint_fraction:.1f}"
        row[key] = float(midpoint_errors_joint.get(sweep_hint_fraction, float("nan")))
        row[indiv_key] = float(midpoint_errors_individual.get(sweep_hint_fraction, float("nan")))
        row[delta_key] = float(row[key] - row[indiv_key])
        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    sweep_df["n_models"] = sweep_df["n_models"].astype(int)
    return sweep_df


def build_h0_sweep_panels(
    *,
    df: pd.DataFrame,
    x_field: str,
    models_sorted_by_x: list[str],
    include_cross: bool,
    lower_asymptote: float | None,
    n_train_min: int | None = None,
    n_train_max: int | None = None,
) -> list[dict[str, Any]]:
    sweep_hint_fraction = 0.0
    df_hint0 = df[df["hint_fraction"] == sweep_hint_fraction].copy()
    if df_hint0.empty:
        raise ValueError("No rows found for hint_fraction = 0.0 when building sweep fit panels.")

    individual_by_hint_all = fit_individual_sigmoids_by_hint(
        df=df,
        x_field=x_field,
        fit_models=None,
        lower=lower_asymptote,
    )
    all_fit_h0 = individual_by_hint_all.get(sweep_hint_fraction)
    x_range = np.linspace(float(df_hint0[x_field].min()) - 5.0, float(df_hint0[x_field].max()) + 5.0, 120)
    panels: list[dict[str, Any]] = []

    min_n_train = 3 if n_train_min is None else int(n_train_min)
    max_n_train = len(models_sorted_by_x) if n_train_max is None else int(n_train_max)
    if min_n_train < 3:
        raise ValueError(f"n_train_min must be >= 3, got {min_n_train}")
    if max_n_train > len(models_sorted_by_x):
        raise ValueError(
            f"n_train_max ({max_n_train}) cannot exceed number of models "
            f"({len(models_sorted_by_x)})"
        )
    if min_n_train > max_n_train:
        raise ValueError(
            f"n_train_min ({min_n_train}) cannot exceed n_train_max ({max_n_train})"
        )

    for n_models in range(min_n_train, max_n_train + 1):
        train_models_for_panel = set(models_sorted_by_x[:n_models])
        test_models_for_panel = set(models_sorted_by_x[n_models:])

        df_panel = df_hint0.copy()
        df_panel["split"] = df_panel["model"].map(
            lambda model: "train" if model in train_models_for_panel else "test"
        )
        train_df = (
            df_panel[df_panel["split"] == "train"]
            .sort_values(x_field)
            .rename(columns={x_field: "x"})
        )
        test_df = (
            df_panel[df_panel["split"] == "test"]
            .sort_values(x_field)
            .rename(columns={x_field: "x"})
        )

        joint_result_for_panel = fit_joint_sigmoid_model(
            df=df,
            fit_models=train_models_for_panel,
            x_field=x_field,
            include_cross=include_cross,
            lower=lower_asymptote,
        )
        individual_by_hint_train_for_panel = fit_individual_sigmoids_by_hint(
            df=df,
            x_field=x_field,
            fit_models=train_models_for_panel,
            lower=lower_asymptote,
        )
        train_fit_h0 = individual_by_hint_train_for_panel.get(sweep_hint_fraction)

        panels.append(
            {
                "train_df": train_df,
                "test_df": test_df,
                "x_range": x_range,
                "predict_joint": lambda x_value, predict=joint_result_for_panel["predict"]: predict(
                    float(x_value), sweep_hint_fraction
                ),
                "predict_train": (
                    None
                    if train_fit_h0 is None
                    else lambda x_value, predict=train_fit_h0["predict"]: predict(float(x_value))
                ),
                "predict_all": (
                    None
                    if all_fit_h0 is None
                    else lambda x_value, predict=all_fit_h0["predict"]: predict(float(x_value))
                ),
                "midpoint_all": (
                    None if all_fit_h0 is None else float(all_fit_h0["midpoint"])
                ),
                "n_train": n_models,
                "n_test": len(test_models_for_panel),
            }
        )
    return panels
