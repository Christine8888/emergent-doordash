from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.joint_scaling_fit import (
    build_h0_sweep_panels,
    build_joint_scaling_df,
    compute_midpoint_errors,
    compute_rms_individual_by_hint,
    compute_rms_joint,
    fit_individual_sigmoids_by_hint,
    fit_individual_sigmoids_by_model,
    fit_joint_sigmoid_model,
    format_joint_equation,
    run_joint_model_sweep,
)
from src.joint_scaling_plots import (
    plot_h0_fits_by_model_sweep,
    plot_joint_accuracy_vs_hint_by_model,
    plot_joint_accuracy_vs_x_by_hint,
    plot_joint_individual_fits_by_hint,
    plot_joint_model_sweep,
)
from src.x_axes import XAxisSpec


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_float = float(value)
        return None if np.isnan(value_float) else value_float
    return value


def run_joint_scaling_for_x_axis(
    *,
    base_rows: list[dict[str, Any]],
    x_axis: XAxisSpec,
    models: list[str],
    train_models: list[str] | None,
    holdout_models: list[str] | None,
    output_dir: Path,
    label: str,
    include_cross: bool,
    lower_asymptote: float | None,
    num_holdout_models: int,
) -> dict[str, Any]:
    if num_holdout_models < 0:
        raise ValueError(f"num_holdout_models must be >= 0, got {num_holdout_models}")
    if num_holdout_models > len(models):
        raise ValueError(
            f"num_holdout_models ({num_holdout_models}) cannot exceed number of models ({len(models)})"
        )

    missing_models = sorted(set(models) - set(x_axis.model_to_x.keys()))
    if missing_models:
        raise ValueError(
            f"Joint scaling x-axis {x_axis.name} missing model values: {missing_models}"
        )

    x_field = "x_value"
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_models is None:
        models_sorted_by_x = sorted(models, key=lambda model: float(x_axis.model_to_x[model]))
        holdout_model_set = set(models_sorted_by_x[-num_holdout_models:]) if num_holdout_models > 0 else set()
        train_model_order = (
            list(models_sorted_by_x[:-num_holdout_models])
            if num_holdout_models > 0
            else list(models_sorted_by_x)
        )
        holdout_model_order = list(models_sorted_by_x[len(train_model_order):])
    else:
        train_model_order = [str(model) for model in train_models]
        holdout_model_order = [] if holdout_models is None else [str(model) for model in holdout_models]
        holdout_model_set = set(holdout_model_order)
        models_sorted_by_x = sorted(models, key=lambda model: float(x_axis.model_to_x[model]))

    train_model_set = set(train_model_order)
    df = build_joint_scaling_df(
        base_rows=base_rows,
        x_map=x_axis.model_to_x,
        x_field=x_field,
        train_models=train_model_set,
    )
    filename_suffix = f"__n_test_{len(holdout_model_set)}"

    df = df.copy()
    df["split"] = df["model"].map(lambda model: "train" if model in train_model_set else "test")

    joint_result = fit_joint_sigmoid_model(
        df=df,
        fit_models=train_model_set,
        x_field=x_field,
        include_cross=include_cross,
        lower=lower_asymptote,
    )
    joint_equation = format_joint_equation(joint_result)

    individual_by_hint_all = fit_individual_sigmoids_by_hint(
        df=df,
        x_field=x_field,
        fit_models=None,
        lower=lower_asymptote,
    )
    individual_by_hint_train = fit_individual_sigmoids_by_hint(
        df=df,
        x_field=x_field,
        fit_models=train_model_set,
        lower=lower_asymptote,
    )
    individual_by_model = fit_individual_sigmoids_by_model(
        df=df,
        fit_models=None,
        lower=lower_asymptote,
    )

    plot_paths = {
        "accuracy_vs_x_by_hint": str(
            plot_joint_accuracy_vs_x_by_hint(
                df=df,
                x_field=x_field,
                x_label=x_axis.label,
                joint_predict_fn=joint_result["predict"],
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"accuracy_vs_{x_axis.name}_by_hint{filename_suffix}",
            )
        ),
        "individual_fits_by_hint": str(
            plot_joint_individual_fits_by_hint(
                df=df,
                x_field=x_field,
                x_label=x_axis.label,
                joint_predict_fn=joint_result["predict"],
                individual_by_hint_all=individual_by_hint_all,
                individual_by_hint_train=individual_by_hint_train,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"individual_fits_by_hint__{x_axis.name}{filename_suffix}",
            )
        ),
        "accuracy_vs_hint_by_model": str(
            plot_joint_accuracy_vs_hint_by_model(
                df=df,
                model_to_x=x_axis.model_to_x,
                x_label=x_axis.label,
                joint_predict_fn=joint_result["predict"],
                individual_by_model=individual_by_model,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"accuracy_vs_hint_by_model__{x_axis.name}{filename_suffix}",
            )
        ),
    }

    panels = build_h0_sweep_panels(
        df=df,
        x_field=x_field,
        models_sorted_by_x=models_sorted_by_x,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
    )
    plot_paths["h0_fits_by_model_sweep"] = str(
        plot_h0_fits_by_model_sweep(
            panels=panels,
            x_label=x_axis.label,
            label=label,
            output_dir=output_dir,
            filename_stem=f"h0_fits_by_model_sweep__{x_axis.name}",
        )
    )

    sweep_df = run_joint_model_sweep(
        df=df,
        x_field=x_field,
        models_sorted_by_x=models_sorted_by_x,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
    )
    plot_paths["model_sweep"] = str(
        plot_joint_model_sweep(
            sweep_df=sweep_df,
            x_label=x_axis.label,
            label=label,
            output_dir=output_dir,
            filename_stem=f"model_sweep__{x_axis.name}",
        )
    )

    metrics = {
        "x_axis_name": x_axis.name,
        "x_axis_label": x_axis.label,
        "x_axis_benchmark_label": x_axis.benchmark_label,
        "x_axis_equation": x_axis.equation,
        "joint_equation": joint_equation,
        "joint_params": [float(value) for value in np.asarray(joint_result["params"], dtype=float)],
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "optimizer_success": bool(joint_result["optimizer_success"]),
        "optimizer_status": int(joint_result["optimizer_status"]),
        "optimizer_message": str(joint_result["optimizer_message"]),
        "n_train_models": int(len(train_model_set)),
        "n_test_models": int(len(holdout_model_set)),
        "rms_train": compute_rms_joint(
            joint_result=joint_result,
            df=df,
            x_field=x_field,
            models=train_model_set,
        ),
        "rms_test": compute_rms_joint(
            joint_result=joint_result,
            df=df,
            x_field=x_field,
            models=holdout_model_set,
        ) if holdout_model_set else float("nan"),
        "rms_all": compute_rms_joint(
            joint_result=joint_result,
            df=df,
            x_field=x_field,
            models=None,
        ),
        "rms_indiv_train": compute_rms_individual_by_hint(
            individual_by_hint=individual_by_hint_train,
            df=df,
            x_field=x_field,
            models=train_model_set,
        ),
        "rms_indiv_test": compute_rms_individual_by_hint(
            individual_by_hint=individual_by_hint_train,
            df=df,
            x_field=x_field,
            models=holdout_model_set,
        ) if holdout_model_set else float("nan"),
        "rms_indiv_all": compute_rms_individual_by_hint(
            individual_by_hint=individual_by_hint_train,
            df=df,
            x_field=x_field,
            models=None,
        ),
        "train_models": list(train_model_order),
        "holdout_models": list(holdout_model_order),
        "plot_paths": plot_paths,
    }
    metrics["delta_rms_train"] = float(metrics["rms_train"]) - float(metrics["rms_indiv_train"])
    metrics["delta_rms_test"] = float(metrics["rms_test"]) - float(metrics["rms_indiv_test"])
    metrics["delta_rms_all"] = float(metrics["rms_all"]) - float(metrics["rms_indiv_all"])
    midpoint_errors_all = compute_midpoint_errors(
        joint_result=joint_result,
        individual_fits=individual_by_hint_all,
        hint_fractions=sorted(df["hint_fraction"].unique().tolist()),
    )
    metrics["mean_midpoint_error_all"] = (
        float(np.mean(list(midpoint_errors_all.values()))) if midpoint_errors_all else float("nan")
    )
    metrics["model_sweep_rows"] = [
        {
            key: _json_safe_scalar(value)
            for key, value in row.items()
        }
        for row in sweep_df.to_dict(orient="records")
    ]

    _write_json(output_dir / "metrics.json", metrics)
    return metrics
