from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.pca import (
    PCAResult,
    build_baseline_benchmark_pca_result,
    build_component_score_map,
    build_hinted_pca_result,
    format_component_equation,
)
from src.scaling_data import eci_benchmark_label, load_eci_map


SUPPORTED_X_AXIS_METHODS = (
    "eci",
    "baseline_pc1",
    "hinted_pc1",
)


@dataclass
class XAxisSpec:
    name: str
    label: str
    benchmark_label: str | None
    equation: str | None
    model_to_x: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _format_hint_fraction_label(hint_fractions: list[float]) -> str:
    return ", ".join(f"{float(hint_fraction):.1f}" for hint_fraction in hint_fractions)


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


def build_baseline_pc_x_axis(
    *,
    scores_df: pd.DataFrame,
    benchmark_order: list[str],
    canonicalize_model_name: Callable[[str], str] | None = None,
    component_idx: int = 0,
) -> XAxisSpec:
    pca_result = build_baseline_benchmark_pca_result(
        scores_df=scores_df,
        benchmark_order=benchmark_order,
        canonicalize_model_name=canonicalize_model_name,
    )
    component_number = component_idx + 1
    return XAxisSpec(
        name=f"baseline_pc{component_number}",
        label=f"Baseline PC{component_number}",
        benchmark_label=", ".join(str(benchmark_name) for benchmark_name in benchmark_order),
        equation=format_component_equation(
            pca_result,
            component_idx=component_idx,
        ),
        model_to_x=build_component_score_map(
            pca_result,
            component_idx=component_idx,
        ),
        metadata={
            "pca_result": pca_result,
            "component_idx": int(component_idx),
            "benchmarks": list(benchmark_order),
        },
    )


def build_hinted_pc_x_axis(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float] | None = None,
    component_idx: int = 0,
    canonicalize_model_name: Callable[[str], str] | None = None,
) -> XAxisSpec:
    pca_result = build_hinted_pca_result(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        hint_fractions=hint_fractions,
    )
    selected_hint_fractions = [
        float(value)
        for value in pca_result.metadata.get("hint_fractions", hint_fractions or [])
    ]
    component_number = component_idx + 1
    raw_model_to_x = build_component_score_map(
        pca_result,
        component_idx=component_idx,
    )
    model_to_x = (
        {
            canonicalize_model_name(str(model_name)): float(x_value)
            for model_name, x_value in raw_model_to_x.items()
        }
        if canonicalize_model_name is not None
        else raw_model_to_x
    )
    return XAxisSpec(
        name=f"hinted_pc{component_number}",
        label=f"Hinted PC{component_number}",
        benchmark_label=_format_hint_fraction_label(selected_hint_fractions),
        equation=format_component_equation(
            pca_result,
            component_idx=component_idx,
        ),
        model_to_x=model_to_x,
        metadata={
            "pca_result": pca_result,
            "component_idx": int(component_idx),
            "benchmark": benchmark,
            "hint_type": hint_type,
            "fractioner": fractioner,
            "hint_fractions": selected_hint_fractions,
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

        if method == "baseline_pc1":
            if scores_df is None or benchmark_order is None:
                raise ValueError(
                    "Baseline PC1 requires scores_df and benchmark_order."
                )
            x_axes.append(
                build_baseline_pc_x_axis(
                    scores_df=scores_df,
                    benchmark_order=benchmark_order,
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
                    component_idx=0,
                    canonicalize_model_name=canonicalize_model_name,
                )
            )
            continue

        raise ValueError(
            f"Unsupported x-axis method: {method}. "
            f"Expected one of {SUPPORTED_X_AXIS_METHODS}."
        )

    return x_axes
