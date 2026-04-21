from __future__ import annotations

from pathlib import Path
from typing import Any

from src.joint_scaling_plots import (
    plot_accuracy_vs_x_by_hint,
    plot_accuracy_vs_x_by_hint_subplots_with_error_bars,
    plot_pca_component_weights,
    plot_pca_explained_variance,
)
from src.scaling_data import build_x_rows
from src.sigmoid_fits import fit_plot_sigmoid
from src.x_axes import XAxisSpec, get_pca_result


def plot_accuracy_views_for_x_axes(
    *,
    base_rows: list[dict[str, Any]],
    x_axes: list[XAxisSpec],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    plot_paths: dict[str, dict[str, str]] = {}
    model_names = {str(row["model"]) for row in base_rows}

    for x_axis in x_axes:
        missing_models = sorted(model_names - set(x_axis.model_to_x.keys()))
        if missing_models:
            raise ValueError(
                f"Configured models missing {x_axis.name} values: {missing_models}"
            )

        rows = build_x_rows(
            base_rows=base_rows,
            x_map=x_axis.model_to_x,
        )
        if not rows:
            print(f"[plot_accuracy_views_for_x_axes][WARN] no rows for x_axis={x_axis.name}")
            continue

        x_axis_output_dir = output_dir / x_axis.name
        x_axis_output_dir.mkdir(parents=True, exist_ok=True)

        x_axis_plot_paths: dict[str, str] = {
            "accuracy_vs_x_by_hint": str(
                plot_accuracy_vs_x_by_hint(
                    rows=rows,
                    benchmark=benchmark,
                    hint_type=hint_type,
                    fractioner=fractioner,
                    x_method=x_axis.name,
                    x_label=x_axis.label,
                    x_benchmark_label=x_axis.benchmark_label or "unknown",
                    x_equation=x_axis.equation,
                    output_dir=x_axis_output_dir,
                    fit_series_fn=fit_plot_sigmoid,
                )
            ),
            "accuracy_vs_x_by_hint_subplots_with_error_bars": str(
                plot_accuracy_vs_x_by_hint_subplots_with_error_bars(
                    rows=rows,
                    benchmark=benchmark,
                    hint_type=hint_type,
                    fractioner=fractioner,
                    x_method=x_axis.name,
                    x_label=x_axis.label,
                    x_benchmark_label=x_axis.benchmark_label or "unknown",
                    x_equation=x_axis.equation,
                    output_dir=x_axis_output_dir,
                    fit_series_fn=fit_plot_sigmoid,
                )
            ),
        }

        pca_result = get_pca_result(x_axis)
        if pca_result is not None:
            component_weights_path = x_axis_output_dir / "pca_component_weights.png"
            explained_variance_path = x_axis_output_dir / "pca_explained_variance.png"
            plot_pca_component_weights(
                components=pca_result.components,
                benchmarks=pca_result.feature_names,
                output_path=component_weights_path,
            )
            plot_pca_explained_variance(
                explained_variance_ratio=pca_result.explained_variance_ratio,
                output_path=explained_variance_path,
            )
            x_axis_plot_paths["pca_component_weights"] = str(component_weights_path)
            x_axis_plot_paths["pca_explained_variance"] = str(explained_variance_path)

        plot_paths[x_axis.name] = x_axis_plot_paths

    return plot_paths
