from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.pca import (
    PCAResult,
    build_hinted_feature_rows,
    format_component_equation,
    project_scores_with_pca,
    run_pca,
)
from src.joint_scaling_fit import build_joint_scaling_df, fit_joint_sigmoid_model
from src.scaling_data import eci_benchmark_label, load_eci_map
from src.sigmoid_fits import clip_accuracy_for_logit


HINTED_ACC_LOGIT_FRACTIONS = tuple(round(step / 10.0, 1) for step in range(1, 11))


def hinted_accuracy_logit_method_name(hint_fraction: float) -> str:
    hint_fraction_int = int(round(float(hint_fraction) * 10.0))
    if not 1 <= hint_fraction_int <= 10:
        raise ValueError(
            "hinted accuracy logit x-axis only supports hint fractions 0.1 through 1.0, "
            f"got {hint_fraction}"
        )
    return f"hinted_acc_h{hint_fraction_int:02d}_logit"


HINTED_ACC_LOGIT_METHODS = tuple(
    hinted_accuracy_logit_method_name(hint_fraction)
    for hint_fraction in HINTED_ACC_LOGIT_FRACTIONS
)


SUPPORTED_X_AXIS_METHODS = (
    "eci",
    "eci_pc1",
    "hinted_pc1",
    "hinted_pc12_linear",
    *HINTED_ACC_LOGIT_METHODS,
)


@dataclass
class XAxisSpec:
    name: str
    label: str
    benchmark_label: str | None
    equation: str | None
    model_to_x: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _mean_accuracy_by_model(
    *,
    base_rows: list[dict[str, Any]],
    models: set[str],
) -> dict[str, float]:
    values_by_model: dict[str, list[float]] = {}
    for row in base_rows:
        model = str(row["model"])
        if model not in models:
            continue
        values_by_model.setdefault(model, []).append(float(row["accuracy"]))
    return {
        model: float(np.mean(values))
        for model, values in values_by_model.items()
        if values
    }


def orient_pc_x_axis_to_accuracy(
    *,
    x_axis: XAxisSpec,
    base_rows: list[dict[str, Any]],
    train_models: list[str],
) -> XAxisSpec:
    if "pc" not in x_axis.name:
        return x_axis

    mean_accuracy = _mean_accuracy_by_model(
        base_rows=base_rows,
        models={str(model) for model in train_models},
    )
    shared_models = [
        str(model)
        for model in train_models
        if str(model) in x_axis.model_to_x and str(model) in mean_accuracy
    ]
    if len(shared_models) < 2:
        x_axis.metadata["accuracy_orientation_flipped"] = False
        x_axis.metadata["accuracy_orientation_reason"] = "fewer_than_two_shared_train_models"
        return x_axis

    xs = np.asarray([x_axis.model_to_x[model] for model in shared_models], dtype=float)
    ys = np.asarray([mean_accuracy[model] for model in shared_models], dtype=float)
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        x_axis.metadata["accuracy_orientation_flipped"] = False
        x_axis.metadata["accuracy_orientation_reason"] = "constant_x_or_accuracy"
        return x_axis

    correlation = float(np.corrcoef(xs, ys)[0, 1])
    if not np.isfinite(correlation):
        x_axis.metadata["accuracy_orientation_flipped"] = False
        x_axis.metadata["accuracy_orientation_reason"] = "nonfinite_correlation"
        return x_axis

    x_axis.metadata["accuracy_orientation_target"] = "mean_accuracy_across_hint_fractions"
    x_axis.metadata["accuracy_orientation_train_models"] = list(shared_models)
    x_axis.metadata["accuracy_orientation_correlation_before"] = correlation
    if correlation >= 0.0:
        x_axis.metadata["accuracy_orientation_flipped"] = False
        return x_axis

    x_axis.model_to_x = {
        model: -float(value)
        for model, value in x_axis.model_to_x.items()
    }
    x_axis.metadata["accuracy_orientation_flipped"] = True
    x_axis.metadata["accuracy_orientation_correlation_after"] = -correlation

    if x_axis.name == "hinted_pc12_linear":
        weight_pc1 = -float(x_axis.metadata.get("linear_weight_pc1", 0.0))
        weight_pc2 = -float(x_axis.metadata.get("linear_weight_pc2", 0.0))
        x_axis.metadata["linear_weight_pc1"] = weight_pc1
        x_axis.metadata["linear_weight_pc2"] = weight_pc2
        x_axis.equation = f"C = {weight_pc1:+.3f}·PC1 {weight_pc2:+.3f}·PC2"
    else:
        pca_result = get_pca_result(x_axis)
        component_idx = int(x_axis.metadata.get("component_idx", 0))
        if pca_result is not None and 0 <= component_idx < pca_result.components.shape[0]:
            pca_result.components[component_idx] = -pca_result.components[component_idx]
            pca_result.scores[:, component_idx] = -pca_result.scores[:, component_idx]
            x_axis.equation = format_component_equation(
                pca_result,
                component_idx=component_idx,
            )

    return x_axis


def _format_hint_fraction_label(hint_fractions: list[float]) -> str:
    return ", ".join(f"{float(hint_fraction):.1f}" for hint_fraction in hint_fractions)


def parse_hinted_accuracy_logit_method(method: str) -> float | None:
    if not method.startswith("hinted_acc_h") or not method.endswith("_logit"):
        return None
    encoded_fraction = method[len("hinted_acc_h") : -len("_logit")]
    if len(encoded_fraction) != 2 or not encoded_fraction.isdigit():
        return None
    hint_fraction_int = int(encoded_fraction)
    if hint_fraction_int < 1 or hint_fraction_int > 10:
        return None
    return float(hint_fraction_int) / 10.0


def build_eci_x_axis(
    *,
    eci_path: Path,
) -> XAxisSpec:
    return XAxisSpec(
        name="eci",
        label="ECI",
        benchmark_label=eci_benchmark_label(eci_path),
        equation=None,
        model_to_x=load_eci_map(eci_path),
        metadata={
            "eci_path": str(eci_path),
        },
    )


def _canonicalize_series_models(
    series: pd.Series,
    canonicalize_model_name: Callable[[str], str] | None,
) -> pd.Series:
    if canonicalize_model_name is None:
        return series.map(lambda value: str(value))
    return series.map(lambda value: canonicalize_model_name(str(value)))


def build_eci_pc_x_axis(
    *,
    scores_df: pd.DataFrame,
    benchmark_order: list[str],
    selected_models: list[str],
    fit_models: list[str],
    canonicalize_model_name: Callable[[str], str] | None = None,
    component_idx: int = 0,
) -> XAxisSpec:
    df = scores_df[scores_df["benchmark"].isin(benchmark_order)].copy()
    df["model"] = _canonicalize_series_models(df["model"], canonicalize_model_name)
    pivot = df.pivot(index="model", columns="benchmark", values="score")
    pivot = pivot.reindex(columns=benchmark_order)

    selected_models_ordered = [str(model) for model in selected_models]
    fit_models_ordered = [str(model) for model in fit_models]

    missing_selected = sorted(set(selected_models_ordered) - set(pivot.index.tolist()))
    if missing_selected:
        raise ValueError(f"Missing baseline benchmark rows for selected models: {missing_selected}")

    selected_pivot = pivot.loc[selected_models_ordered]
    selected_pivot = selected_pivot.dropna(axis=0, how="any")
    missing_after_drop = sorted(set(selected_models_ordered) - set(selected_pivot.index.tolist()))
    if missing_after_drop:
        raise ValueError(
            f"Selected models missing complete baseline benchmark coverage: {missing_after_drop}"
        )

    missing_fit = sorted(set(fit_models_ordered) - set(selected_pivot.index.tolist()))
    if missing_fit:
        raise ValueError(f"Fit models missing complete baseline benchmark coverage: {missing_fit}")

    fit_pivot = selected_pivot.loc[fit_models_ordered]
    pca_result = run_pca(
        model_names=[str(model_name) for model_name in fit_pivot.index.tolist()],
        feature_names=[str(benchmark_name) for benchmark_name in benchmark_order],
        matrix=fit_pivot.to_numpy(dtype=float),
        metadata={
            "benchmarks": list(benchmark_order),
            "fit_model_names": list(fit_models_ordered),
            "project_model_names": list(selected_models_ordered),
        },
    )
    projected_scores = project_scores_with_pca(
        pca_result,
        feature_names=[str(benchmark_name) for benchmark_name in benchmark_order],
        matrix=selected_pivot.loc[selected_models_ordered].to_numpy(dtype=float),
    )
    component_number = component_idx + 1
    return XAxisSpec(
        name=f"eci_pc{component_number}",
        label=f"ECI PC{component_number}",
        benchmark_label=", ".join(str(benchmark_name) for benchmark_name in benchmark_order),
        equation=format_component_equation(
            pca_result,
            component_idx=component_idx,
        ),
        model_to_x={
            model_name: float(projected_scores[idx, component_idx])
            for idx, model_name in enumerate(selected_models_ordered)
        },
        metadata={
            "pca_result": pca_result,
            "component_idx": int(component_idx),
            "benchmarks": list(benchmark_order),
            "fit_model_names": list(fit_models_ordered),
            "project_model_names": list(selected_models_ordered),
        },
    )


def _fit_hinted_train_pca_and_project(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float] | None,
    selected_models: list[str],
    fit_models: list[str],
    canonicalize_model_name: Callable[[str], str] | None,
) -> tuple[PCAResult, np.ndarray, list[float], list[str]]:
    raw_rows_by_model, selected_hint_fractions, shared_fractioners = build_hinted_feature_rows(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fractions=hint_fractions,
    )
    rows_by_model: dict[str, dict[str, float]] = {}
    for model_name, feature_row in raw_rows_by_model.items():
        canonical_model_name = (
            canonicalize_model_name(model_name)
            if canonicalize_model_name is not None
            else str(model_name)
        )
        rows_by_model[canonical_model_name] = {
            str(feature_name): float(value)
            for feature_name, value in feature_row.items()
        }

    selected_models_ordered = [str(model) for model in selected_models]
    fit_models_ordered = [str(model) for model in fit_models]
    missing_selected = sorted(set(selected_models_ordered) - set(rows_by_model.keys()))
    if missing_selected:
        raise ValueError(f"Missing hinted rows for selected models: {missing_selected}")
    missing_fit = sorted(set(fit_models_ordered) - set(rows_by_model.keys()))
    if missing_fit:
        raise ValueError(f"Missing hinted rows for fit models: {missing_fit}")

    shared_features = sorted(
        set.intersection(*(set(rows_by_model[model].keys()) for model in fit_models_ordered)),
        key=lambda name: (
            name.split("@", 1)[0],
            float(name.split("@", 1)[1]) if "@" in name else 0.0,
        ),
    )
    if not shared_features:
        raise ValueError("No shared hinted features found across fit models.")

    missing_project_features = [
        model_name
        for model_name in selected_models_ordered
        if any(feature_name not in rows_by_model[model_name] for feature_name in shared_features)
    ]
    if missing_project_features:
        raise ValueError(
            f"Selected models missing train-shared hinted features: {missing_project_features}"
        )

    fit_matrix = np.asarray(
        [
            [rows_by_model[model_name][feature_name] for feature_name in shared_features]
            for model_name in fit_models_ordered
        ],
        dtype=float,
    )
    pca_result = run_pca(
        model_names=list(fit_models_ordered),
        feature_names=list(shared_features),
        matrix=fit_matrix,
        metadata={
            "benchmark": benchmark,
            "hint_type": hint_type,
            "requested_fractioner": fractioner,
            "shared_fractioners": list(shared_fractioners),
            "hint_fractions": list(selected_hint_fractions),
            "fit_model_names": list(fit_models_ordered),
            "project_model_names": list(selected_models_ordered),
        },
    )
    project_matrix = np.asarray(
        [
            [rows_by_model[model_name][feature_name] for feature_name in shared_features]
            for model_name in selected_models_ordered
        ],
        dtype=float,
    )
    projected_scores = project_scores_with_pca(
        pca_result,
        feature_names=list(shared_features),
        matrix=project_matrix,
    )
    return pca_result, projected_scores, selected_hint_fractions, shared_fractioners


def build_hinted_pc_x_axis(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float] | None = None,
    selected_models: list[str],
    fit_models: list[str],
    component_idx: int = 0,
    canonicalize_model_name: Callable[[str], str] | None = None,
) -> XAxisSpec:
    pca_result, projected_scores, selected_hint_fractions, shared_fractioners = _fit_hinted_train_pca_and_project(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fractions=hint_fractions,
        selected_models=selected_models,
        fit_models=fit_models,
        canonicalize_model_name=canonicalize_model_name,
    )
    selected_hint_fractions = [
        float(value) for value in selected_hint_fractions
    ]
    component_number = component_idx + 1
    return XAxisSpec(
        name=f"hinted_pc{component_number}",
        label=f"Hinted PC{component_number}",
        benchmark_label=_format_hint_fraction_label(selected_hint_fractions),
        equation=format_component_equation(
            pca_result,
            component_idx=component_idx,
        ),
        model_to_x={
            str(model_name): float(projected_scores[idx, component_idx])
            for idx, model_name in enumerate(selected_models)
        },
        metadata={
            "pca_result": pca_result,
            "component_idx": int(component_idx),
            "benchmark": benchmark,
            "hint_type": hint_type,
            "fractioner": fractioner,
            "hint_fractions": list(selected_hint_fractions),
            "shared_fractioners": list(shared_fractioners),
            "fit_model_names": list(fit_models),
            "project_model_names": list(selected_models),
        },
    )


def build_hinted_pc12_linear_x_axis(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float] | None,
    selected_models: list[str],
    fit_models: list[str],
    base_rows: list[dict[str, Any]],
    include_cross: bool,
    lower_asymptote: float | None,
    canonicalize_model_name: Callable[[str], str] | None = None,
) -> XAxisSpec:
    from scipy.optimize import minimize

    pca_result, projected_scores, selected_hint_fractions, shared_fractioners = _fit_hinted_train_pca_and_project(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fractions=hint_fractions,
        selected_models=selected_models,
        fit_models=fit_models,
        canonicalize_model_name=canonicalize_model_name,
    )
    if projected_scores.shape[1] < 2:
        raise ValueError("hinted_pc12_linear requires at least two hinted PCA components.")

    pc1_map = {
        str(model_name): float(projected_scores[idx, 0])
        for idx, model_name in enumerate(selected_models)
    }
    pc2_map = {
        str(model_name): float(projected_scores[idx, 1])
        for idx, model_name in enumerate(selected_models)
    }
    fit_model_set = set(str(model_name) for model_name in fit_models)

    def normalize_weights(weights: np.ndarray) -> np.ndarray | None:
        weights_array = np.asarray(weights, dtype=float)
        norm = float(np.linalg.norm(weights_array))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return weights_array / norm

    def build_linear_x_map(weights: np.ndarray) -> dict[str, float]:
        normalized_weights = normalize_weights(weights)
        if normalized_weights is None:
            raise ValueError(f"Cannot build PC1/PC2 x-axis from weights={weights}")
        weight_pc1 = float(normalized_weights[0])
        weight_pc2 = float(normalized_weights[1])
        return {
            str(model_name): weight_pc1 * pc1_map[str(model_name)] + weight_pc2 * pc2_map[str(model_name)]
            for model_name in selected_models
        }

    def objective(weights: np.ndarray) -> float:
        normalized_weights = normalize_weights(weights)
        if normalized_weights is None:
            return float("inf")
        x_map = build_linear_x_map(normalized_weights)
        df = build_joint_scaling_df(
            base_rows=base_rows,
            x_map=x_map,
            x_field="x_value",
            train_models=fit_model_set,
        )
        joint_result = fit_joint_sigmoid_model(
            df=df,
            fit_models=fit_model_set,
            x_field="x_value",
            include_cross=include_cross,
            lower=lower_asymptote,
        )
        return float(joint_result["optimizer_fun"])

    initial_weights = [
        np.asarray([+1.0, 0.0], dtype=float),
        np.asarray([-1.0, 0.0], dtype=float),
        np.asarray([0.0, +1.0], dtype=float),
        np.asarray([0.0, -1.0], dtype=float),
        np.asarray([+1.0, +1.0], dtype=float),
        np.asarray([+1.0, -1.0], dtype=float),
        np.asarray([-1.0, +1.0], dtype=float),
        np.asarray([-1.0, -1.0], dtype=float),
    ]
    constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.dot(weights, weights) - 1.0),
    }
    optimize_results = [
        minimize(
            objective,
            normalize_weights(weights),
            method="SLSQP",
            constraints=[constraint],
            options={"maxiter": 200},
        )
        for weights in initial_weights
    ]
    linear_result = min(optimize_results, key=lambda result: float(result.fun))
    normalized_weights = normalize_weights(np.asarray(linear_result.x, dtype=float))
    if normalized_weights is None:
        raise ValueError("PC1/PC2 linear optimizer returned invalid zero weights.")
    weight_pc1 = float(normalized_weights[0])
    weight_pc2 = float(normalized_weights[1])
    model_to_x = build_linear_x_map(normalized_weights)
    return XAxisSpec(
        name="hinted_pc12_linear",
        label="Hinted PC1/PC2 Linear",
        benchmark_label=_format_hint_fraction_label(selected_hint_fractions),
        equation=f"C = {weight_pc1:+.3f}·PC1 {weight_pc2:+.3f}·PC2",
        model_to_x=model_to_x,
        metadata={
            "pca_result": pca_result,
            "benchmark": benchmark,
            "hint_type": hint_type,
            "fractioner": fractioner,
            "hint_fractions": list(selected_hint_fractions),
            "shared_fractioners": list(shared_fractioners),
            "fit_model_names": list(fit_models),
            "project_model_names": list(selected_models),
            "linear_weight_pc1": weight_pc1,
            "linear_weight_pc2": weight_pc2,
            "linear_optimizer_success": bool(linear_result.success),
            "linear_optimizer_fun": float(linear_result.fun),
            "linear_optimizer_status": int(linear_result.status),
            "linear_optimizer_message": str(linear_result.message),
            "pc1_map": dict(pc1_map),
            "pc2_map": dict(pc2_map),
        },
    )


def build_hinted_accuracy_logit_x_axis(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fraction: float,
    selected_models: list[str],
    fit_models: list[str],
    base_rows: list[dict[str, Any]],
    method_name: str | None = None,
) -> XAxisSpec:
    resolved_method_name = (
        hinted_accuracy_logit_method_name(hint_fraction)
        if method_name is None
        else str(method_name)
    )
    hint_fraction_value = float(hint_fraction)
    hint_fraction_label = f"{hint_fraction_value:.1f}"
    selected_models_ordered = [str(model) for model in selected_models]
    fit_models_ordered = [str(model) for model in fit_models]
    rows_at_fraction = [
        row
        for row in base_rows
        if abs(float(row["hint_fraction"]) - hint_fraction_value) <= 1e-8
    ]
    raw_accuracy_map = {
        str(row["model"]): float(row["accuracy"])
        for row in rows_at_fraction
    }
    missing_selected = sorted(set(selected_models_ordered) - set(raw_accuracy_map.keys()))
    if missing_selected:
        raise ValueError(
            f"Missing hinted accuracy rows at hint_fraction={hint_fraction_label} "
            f"for selected models: {missing_selected}"
        )
    missing_fit = sorted(set(fit_models_ordered) - set(raw_accuracy_map.keys()))
    if missing_fit:
        raise ValueError(
            f"Missing hinted accuracy rows at hint_fraction={hint_fraction_label} "
            f"for fit models: {missing_fit}"
        )

    fit_raw = np.asarray([raw_accuracy_map[model_name] for model_name in fit_models_ordered], dtype=float)
    fit_clipped = clip_accuracy_for_logit(fit_raw, lower=None)
    fit_logit = np.log(fit_clipped / (1.0 - fit_clipped))
    fit_mean = float(np.mean(fit_logit))
    fit_std = float(np.std(fit_logit))
    safe_fit_std = 1.0 if fit_std <= 0.0 else fit_std

    selected_raw = np.asarray(
        [raw_accuracy_map[model_name] for model_name in selected_models_ordered],
        dtype=float,
    )
    selected_clipped = clip_accuracy_for_logit(selected_raw, lower=None)
    selected_logit = np.log(selected_clipped / (1.0 - selected_clipped))
    selected_z = (selected_logit - fit_mean) / safe_fit_std

    return XAxisSpec(
        name=resolved_method_name,
        label=f"z(logit(Accuracy @ h={hint_fraction_label}))",
        benchmark_label=benchmark,
        equation=f"x = z_train(logit(clip(acc @ h={hint_fraction_label})))",
        model_to_x={
            model_name: float(selected_z[idx])
            for idx, model_name in enumerate(selected_models_ordered)
        },
        metadata={
            "benchmark": benchmark,
            "hint_type": hint_type,
            "fractioner": fractioner,
            "hint_fraction": hint_fraction_value,
            "method_family": "hinted_acc_logit_fixed_fraction",
            "fit_model_names": list(fit_models_ordered),
            "project_model_names": list(selected_models_ordered),
            "fit_logit_mean": fit_mean,
            "fit_logit_std": fit_std,
            "fit_logit_std_safe": float(safe_fit_std),
            "raw_accuracy_map": dict(raw_accuracy_map),
            "fit_logit_values": {
                model_name: float(fit_logit[idx])
                for idx, model_name in enumerate(fit_models_ordered)
            },
            "project_logit_values": {
                model_name: float(selected_logit[idx])
                for idx, model_name in enumerate(selected_models_ordered)
            },
        },
    )


def get_pca_result(x_axis: XAxisSpec) -> PCAResult | None:
    pca_result = x_axis.metadata.get("pca_result")
    if isinstance(pca_result, PCAResult):
        return pca_result
    return None


def build_x_axes_from_methods(
    *,
    methods: list[str],
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float] | None,
    selected_models: list[str],
    fit_models: list[str],
    base_rows: list[dict[str, Any]] | None,
    include_cross: bool,
    lower_asymptote: float | None,
    eci_path: Path | None,
    scores_df: pd.DataFrame | None,
    benchmark_order: list[str] | None,
    canonicalize_model_name: Callable[[str], str] | None,
) -> list[XAxisSpec]:
    x_axes: list[XAxisSpec] = []
    for method in methods:
        if method == "eci":
            if eci_path is None:
                raise ValueError("--eci-file is required when x-axis method includes 'eci'.")
            x_axes.append(
                build_eci_x_axis(
                    eci_path=eci_path,
                )
            )
            continue

        if method == "eci_pc1":
            if scores_df is None or benchmark_order is None:
                raise ValueError(
                    "ECI PC1 requires scores_df and benchmark_order."
                )
            x_axes.append(
                build_eci_pc_x_axis(
                    scores_df=scores_df,
                    benchmark_order=benchmark_order,
                    selected_models=selected_models,
                    fit_models=fit_models,
                    canonicalize_model_name=canonicalize_model_name,
                    component_idx=0,
                )
            )
            continue

        if method == "hinted_pc1":
            x_axes.append(
                build_hinted_pc_x_axis(
                    benchmark=benchmark,
                    hint_type=hint_type,
                    fractioner=fractioner,
                    hint_fractions=hint_fractions,
                    selected_models=selected_models,
                    fit_models=fit_models,
                    component_idx=0,
                    canonicalize_model_name=canonicalize_model_name,
                )
            )
            continue

        if method == "hinted_pc12_linear":
            if base_rows is None:
                raise ValueError("hinted_pc12_linear requires base_rows.")
            x_axes.append(
                build_hinted_pc12_linear_x_axis(
                    benchmark=benchmark,
                    hint_type=hint_type,
                    fractioner=fractioner,
                    hint_fractions=hint_fractions,
                    selected_models=selected_models,
                    fit_models=fit_models,
                    base_rows=base_rows,
                    include_cross=include_cross,
                    lower_asymptote=lower_asymptote,
                    canonicalize_model_name=canonicalize_model_name,
                )
            )
            continue

        hinted_acc_logit_fraction = parse_hinted_accuracy_logit_method(method)
        if hinted_acc_logit_fraction is not None:
            if base_rows is None:
                raise ValueError(f"{method} requires base_rows.")
            x_axes.append(
                build_hinted_accuracy_logit_x_axis(
                    benchmark=benchmark,
                    hint_type=hint_type,
                    fractioner=fractioner,
                    hint_fraction=hinted_acc_logit_fraction,
                    selected_models=selected_models,
                    fit_models=fit_models,
                    base_rows=base_rows,
                    method_name=method,
                )
            )
            continue

        raise ValueError(
            f"Unsupported x-axis method: {method}. "
            f"Expected one of {SUPPORTED_X_AXIS_METHODS}."
        )

    return x_axes
