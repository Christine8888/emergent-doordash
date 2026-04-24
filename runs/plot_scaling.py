from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.fit_eci import EVAL_TO_ECI, load_baseline_scores
from src.hinted_accuracy import EXPECTED_FRACTIONS
from src.joint_scaling_plots import (
    plot_joint_x_axis_absolute_rms_comparison,
    plot_joint_x_axis_absolute_rms_family,
    plot_joint_x_axis_delta_rms_comparison,
    plot_joint_x_axis_delta_rms_family,
    plot_joint_x_axis_model_sweep_comparison,
)
from src.pca import print_pca_report
from src.scaling_data import (
    build_base_rows,
    canonicalize_model_name,
    load_eci_map,
    load_canonical_combo_results,
    resolve_models_to_use,
)
from src.scaling_runner import plot_accuracy_views_for_x_axes
from src.joint_scaling_runner import run_joint_scaling_for_x_axis
from src.x_axes import (
    SUPPORTED_X_AXIS_METHODS,
    XAxisSpec,
    build_x_axes_from_methods,
    get_pca_result,
)


PLOTS_ROOT = Path("plots/scaling_plots")
PC_BENCHMARK_ORDER = [EVAL_TO_ECI[eval_name] for eval_name in EVAL_TO_ECI]
DEFAULT_JOINT_LOWER_ASYMPTOTE = 0.0
DEFAULT_HINTED_PC_HINT_FRACTIONS = [fraction for fraction in EXPECTED_FRACTIONS if fraction > 0.0]
DEFAULT_X_AXIS_METHODS = list(SUPPORTED_X_AXIS_METHODS)
EXCLUDE_MODELS: set[str] = {
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "google/gemma-3-270m-it",
}
DEFAULT_MODELS_TO_USE: list[str] | None = [
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-1b-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]


@dataclass
class ScalingRunConfig:
    benchmark: str
    hint_type: str
    fractioner: str | None
    x_axis_methods: list[str]
    facet_by: str = "none"
    joint_x_axis: str | None = None
    run_joint_for_all_x_axes: bool = False
    eci_file: Path | None = None
    hint_fractions: list[float] | None = None
    num_holdout_models: int = 0
    include_cross: bool = True
    print_pca_report: bool = False
    output_root: Path = PLOTS_ROOT
    output_subdir: Path | None = None
    log_prefix: str = "[plot_scaling]"
    preferred_models: list[str] | None = field(
        default_factory=lambda: (
            None if DEFAULT_MODELS_TO_USE is None else list(DEFAULT_MODELS_TO_USE)
        )
    )
    restrict_models_to_x_axes: bool = False
    joint_lower_asymptote: float = DEFAULT_JOINT_LOWER_ASYMPTOTE
    pca_summary_lines_fn: Callable[[XAxisSpec], list[str] | None] | None = None


@dataclass
class ScalingRunResult:
    x_axes: list[XAxisSpec]
    output_dir: Path
    plot_paths: dict[str, dict[str, str]]
    joint_metrics: dict[str, object] | None


def _normalize_model_name(model: str) -> str:
    # Support both full model paths and basename-only names.
    return str(model).strip().split("/")[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scaling curves using reusable x-axis definitions."
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument(
        "--x-axis-methods",
        type=str,
        nargs="+",
        default=list(DEFAULT_X_AXIS_METHODS),
        choices=SUPPORTED_X_AXIS_METHODS,
    )
    parser.add_argument("--eci-file", type=str, default=None)
    parser.add_argument("--num-holdout-models", type=int, default=0)
    parser.add_argument("--facet-by", type=str, default="none", choices=["none", "family"])
    return parser.parse_args()


def _default_pca_summary_lines(
    *,
    config: ScalingRunConfig,
    x_axis: XAxisSpec,
) -> list[str]:
    return [
        f"benchmark: {config.benchmark}",
        f"hint_type: {config.hint_type}",
        f"fractioner: {config.fractioner or 'unknown'}",
        f"x_axis: {x_axis.name}",
        f"x_label: {x_axis.label}",
        f"x_benchmark_label: {x_axis.benchmark_label or 'unknown'}",
    ]


def _normalize_joint_x_axis_name(joint_x_axis: str) -> str:
    if joint_x_axis in SUPPORTED_X_AXIS_METHODS:
        return joint_x_axis
    raise ValueError(f"Unsupported joint x-axis: {joint_x_axis}")


def _build_joint_x_axis_comparison_df(
    *,
    joint_metrics: dict[str, object],
    x_axes: list[XAxisSpec],
) -> pd.DataFrame:
    metric_names = [
        "rms_train",
        "rms_test",
        "rms_all",
        "rms_indiv_train",
        "rms_indiv_test",
        "rms_indiv_all",
        "delta_rms_train",
        "delta_rms_test",
        "delta_rms_all",
        "n_train_models",
        "n_test_models",
    ]
    x_axis_order = {x_axis.name: idx for idx, x_axis in enumerate(x_axes)}
    rows: list[dict[str, Any]] = []
    for x_axis in x_axes:
        metrics = joint_metrics.get(x_axis.name)
        if not isinstance(metrics, dict):
            continue
        hint_fraction_raw = x_axis.metadata.get("hint_fraction")
        hint_fraction = (
            float(hint_fraction_raw)
            if isinstance(hint_fraction_raw, (int, float, np.floating))
            else float("nan")
        )
        is_hinted_acc_logit_fixed_fraction = (
            str(x_axis.metadata.get("method_family", "")) == "hinted_acc_logit_fixed_fraction"
        )
        comparison_label = (
            f"h={hint_fraction:.1f}" if is_hinted_acc_logit_fixed_fraction and np.isfinite(hint_fraction)
            else x_axis.name
        )
        row: dict[str, Any] = {
            "x_axis_name": x_axis.name,
            "x_axis_label": x_axis.label,
            "x_axis_benchmark_label": x_axis.benchmark_label,
            "hint_fraction": hint_fraction,
            "comparison_label": comparison_label,
            "comparison_group": (
                "hinted_acc_logit_fixed_fraction" if is_hinted_acc_logit_fixed_fraction else "other"
            ),
            "optimizer_success": bool(metrics.get("optimizer_success", False)),
            "sort_index": int(x_axis_order.get(x_axis.name, len(x_axis_order))),
        }
        for metric_name in metric_names:
            row[metric_name] = float(metrics.get(metric_name, float("nan")))
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df
    return comparison_df.sort_values("sort_index").reset_index(drop=True)


def _build_joint_x_axis_model_sweep_comparison_df(
    *,
    joint_metrics: dict[str, object],
    x_axes: list[XAxisSpec],
) -> pd.DataFrame:
    x_axis_order = {x_axis.name: idx for idx, x_axis in enumerate(x_axes)}
    frames: list[pd.DataFrame] = []
    for x_axis in x_axes:
        metrics = joint_metrics.get(x_axis.name)
        if not isinstance(metrics, dict):
            continue
        sweep_rows = metrics.get("model_sweep_rows")
        if not isinstance(sweep_rows, list) or not sweep_rows:
            continue

        sweep_df = pd.DataFrame(sweep_rows)
        if sweep_df.empty or "n_models" not in sweep_df.columns:
            continue

        hint_fraction_raw = x_axis.metadata.get("hint_fraction")
        hint_fraction = (
            float(hint_fraction_raw)
            if isinstance(hint_fraction_raw, (int, float, np.floating))
            else float("nan")
        )
        is_hinted_acc_logit_fixed_fraction = (
            str(x_axis.metadata.get("method_family", "")) == "hinted_acc_logit_fixed_fraction"
        )
        comparison_label = (
            f"h={hint_fraction:.1f}" if is_hinted_acc_logit_fixed_fraction and np.isfinite(hint_fraction)
            else x_axis.name
        )

        sweep_df = sweep_df.copy()
        sweep_df["x_axis_name"] = x_axis.name
        sweep_df["x_axis_label"] = x_axis.label
        sweep_df["x_axis_benchmark_label"] = x_axis.benchmark_label
        sweep_df["hint_fraction"] = hint_fraction
        sweep_df["comparison_label"] = comparison_label
        sweep_df["comparison_group"] = (
            "hinted_acc_logit_fixed_fraction" if is_hinted_acc_logit_fixed_fraction else "other"
        )
        sweep_df["sort_index"] = int(x_axis_order.get(x_axis.name, len(x_axis_order)))
        frames.append(sweep_df)

    if not frames:
        return pd.DataFrame()

    comparison_df = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "n_models",
        "rms_h0_test",
        "rms_indiv_h0_test",
        "rms_indiv_allfit_h0_test",
        "delta_rms_h0_test",
    ]
    for column in numeric_columns:
        if column in comparison_df.columns:
            comparison_df[column] = pd.to_numeric(comparison_df[column], errors="coerce")
    return comparison_df.sort_values(["sort_index", "n_models"]).reset_index(drop=True)


def _rank_joint_x_axis_model_sweep_by_avg_delta(
    *,
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    if comparison_df.empty or "delta_rms_h0_test" not in comparison_df.columns:
        return pd.DataFrame()

    ranking_df = (
        comparison_df.dropna(subset=["delta_rms_h0_test"])
        .groupby(["x_axis_name", "sort_index"], as_index=False)
        .agg(
            avg_delta_rms_h0_test=("delta_rms_h0_test", "mean"),
            best_delta_rms_h0_test=("delta_rms_h0_test", "min"),
        )
        .sort_values(["avg_delta_rms_h0_test", "sort_index"], ascending=[True, True])
        .reset_index(drop=True)
    )
    if ranking_df.empty:
        return ranking_df

    label_df = (
        comparison_df.sort_values(["x_axis_name", "sort_index", "n_models"])
        .groupby("x_axis_name", as_index=False)
        .first()[["x_axis_name", "comparison_label"]]
    )
    best_point_df = (
        comparison_df.dropna(subset=["delta_rms_h0_test"])
        .sort_values(["x_axis_name", "delta_rms_h0_test", "n_models", "sort_index"])
        .groupby("x_axis_name", as_index=False)
        .first()[["x_axis_name", "n_models"]]
        .rename(columns={"n_models": "best_delta_n_models"})
    )
    ranking_df = ranking_df.merge(label_df, on="x_axis_name", how="left")
    ranking_df = ranking_df.merge(best_point_df, on="x_axis_name", how="left")
    return ranking_df


def _select_top_k_x_axes_by_model_sweep_delta(
    *,
    comparison_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if comparison_df.empty or top_k <= 0:
        return pd.DataFrame()

    ranked_methods = _rank_joint_x_axis_model_sweep_by_avg_delta(comparison_df=comparison_df).head(top_k)
    if ranked_methods.empty:
        return pd.DataFrame()

    selected_names = ranked_methods["x_axis_name"].tolist()
    filtered_df = comparison_df[comparison_df["x_axis_name"].isin(selected_names)].copy()
    return filtered_df.sort_values(["sort_index", "n_models"]).reset_index(drop=True)


def _print_joint_x_axis_model_sweep_delta_ranking(
    *,
    comparison_df: pd.DataFrame,
    log_prefix: str,
) -> None:
    ranking_df = _rank_joint_x_axis_model_sweep_by_avg_delta(comparison_df=comparison_df)
    if ranking_df.empty:
        return

    print(f"{log_prefix} model_sweep_delta_ranking[criterion=mean delta_rms_h0_test over n_train]")
    for rank, row in enumerate(ranking_df.itertuples(index=False), start=1):
        print(
            f"{log_prefix}   rank={rank} "
            f"x_axis={row.x_axis_name} "
            f"label={row.comparison_label} "
            f"avg_delta={float(row.avg_delta_rms_h0_test):.6f} "
            f"best_delta={float(row.best_delta_rms_h0_test):.6f} "
            f"best_delta_n_train={int(row.best_delta_n_models)}"
        )


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


def _print_joint_x_axis_comparison_summary(
    *,
    comparison_df: pd.DataFrame,
    log_prefix: str,
) -> None:
    if comparison_df.empty:
        return

    delta_metric = "delta_rms_test" if comparison_df["delta_rms_test"].notna().any() else "delta_rms_all"
    abs_metric = "rms_test" if comparison_df["rms_test"].notna().any() else "rms_all"

    ranked_delta = comparison_df.dropna(subset=[delta_metric]).sort_values(delta_metric)
    if not ranked_delta.empty:
        best_delta = ranked_delta.iloc[0]
        print(
            f"{log_prefix} best_delta[{delta_metric}] "
            f"x_axis={best_delta['x_axis_name']} value={float(best_delta[delta_metric]):.6f}"
        )

    ranked_abs = comparison_df.dropna(subset=[abs_metric]).sort_values(abs_metric)
    if not ranked_abs.empty:
        best_abs = ranked_abs.iloc[0]
        print(
            f"{log_prefix} best_absolute[{abs_metric}] "
            f"x_axis={best_abs['x_axis_name']} value={float(best_abs[abs_metric]):.6f}"
        )

    family_df = comparison_df[
        comparison_df["comparison_group"] == "hinted_acc_logit_fixed_fraction"
    ].copy()
    if not family_df.empty:
        family_ranked_delta = family_df.dropna(subset=[delta_metric]).sort_values(delta_metric)
        if not family_ranked_delta.empty:
            best_family_delta = family_ranked_delta.iloc[0]
            print(
                f"{log_prefix} best_hinted_acc_logit_delta[{delta_metric}] "
                f"x_axis={best_family_delta['x_axis_name']} "
                f"hint_fraction={float(best_family_delta['hint_fraction']):.1f} "
                f"value={float(best_family_delta[delta_metric]):.6f}"
            )


def _write_joint_x_axis_comparison_artifacts(
    *,
    joint_metrics: dict[str, object],
    x_axes: list[XAxisSpec],
    output_dir: Path,
    label: str,
    log_prefix: str,
) -> dict[str, str]:
    comparison_df = _build_joint_x_axis_comparison_df(
        joint_metrics=joint_metrics,
        x_axes=x_axes,
    )
    if comparison_df.empty:
        return {}

    comparison_output_dir = output_dir / "joint_x_axis_comparison"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = comparison_output_dir / "metrics.csv"
    json_path = comparison_output_dir / "metrics.json"
    comparison_df.to_csv(csv_path, index=False)
    json_payload = {
        "rows": [
            {
                key: _json_safe_scalar(value)
                for key, value in row.items()
            }
            for row in comparison_df.to_dict(orient="records")
        ]
    }
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_paths = {
        "metrics_csv": str(csv_path),
        "metrics_json": str(json_path),
        "delta_rms_all_x_axes": str(
            plot_joint_x_axis_delta_rms_comparison(
                comparison_df=comparison_df,
                label=label,
                output_dir=comparison_output_dir,
                filename_stem="delta_rms_all_x_axes",
            )
        ),
        "absolute_rms_all_x_axes": str(
            plot_joint_x_axis_absolute_rms_comparison(
                comparison_df=comparison_df,
                label=label,
                output_dir=comparison_output_dir,
                filename_stem="absolute_rms_all_x_axes",
            )
        ),
    }

    model_sweep_comparison_df = _build_joint_x_axis_model_sweep_comparison_df(
        joint_metrics=joint_metrics,
        x_axes=x_axes,
    )
    if not model_sweep_comparison_df.empty:
        _print_joint_x_axis_model_sweep_delta_ranking(
            comparison_df=model_sweep_comparison_df,
            log_prefix=log_prefix,
        )
        model_sweep_csv_path = comparison_output_dir / "model_sweep_metrics.csv"
        model_sweep_json_path = comparison_output_dir / "model_sweep_metrics.json"
        model_sweep_comparison_df.to_csv(model_sweep_csv_path, index=False)
        model_sweep_json_payload = {
            "rows": [
                {
                    key: _json_safe_scalar(value)
                    for key, value in row.items()
                }
                for row in model_sweep_comparison_df.to_dict(orient="records")
            ]
        }
        model_sweep_json_path.write_text(
            json.dumps(model_sweep_json_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plot_paths["model_sweep_metrics_csv"] = str(model_sweep_csv_path)
        plot_paths["model_sweep_metrics_json"] = str(model_sweep_json_path)
        plot_paths["model_sweep_all_x_axes"] = str(
            plot_joint_x_axis_model_sweep_comparison(
                comparison_df=model_sweep_comparison_df,
                label=label,
                output_dir=comparison_output_dir,
                filename_stem="model_sweep_all_x_axes",
            )
        )
        top4_model_sweep_df = _select_top_k_x_axes_by_model_sweep_delta(
            comparison_df=model_sweep_comparison_df,
            top_k=4,
        )
        if not top4_model_sweep_df.empty:
            plot_paths["model_sweep_top4_delta_x_axes"] = str(
                plot_joint_x_axis_model_sweep_comparison(
                    comparison_df=top4_model_sweep_df,
                    label=f"{label} - top 4 x-axis methods by lowest average delta RMS",
                    output_dir=comparison_output_dir,
                    filename_stem="model_sweep_top4_delta_x_axes",
                )
            )

    family_df = comparison_df[
        comparison_df["comparison_group"] == "hinted_acc_logit_fixed_fraction"
    ].copy()
    if not family_df.empty:
        plot_paths["delta_rms_hinted_acc_logit_family"] = str(
            plot_joint_x_axis_delta_rms_family(
                comparison_df=family_df,
                label=label,
                output_dir=comparison_output_dir,
                filename_stem="delta_rms_hinted_acc_logit_family",
            )
        )
        plot_paths["absolute_rms_hinted_acc_logit_family"] = str(
            plot_joint_x_axis_absolute_rms_family(
                comparison_df=family_df,
                label=label,
                output_dir=comparison_output_dir,
                filename_stem="absolute_rms_hinted_acc_logit_family",
            )
        )

    _print_joint_x_axis_comparison_summary(
        comparison_df=comparison_df,
        log_prefix=log_prefix,
    )
    return plot_paths


def run_scaling(config: ScalingRunConfig) -> ScalingRunResult:
    if not 0.0 <= float(config.joint_lower_asymptote) < 1.0:
        raise ValueError(
            "joint_lower_asymptote must be in [0, 1), "
            f"got {config.joint_lower_asymptote}"
        )

    x_axis_methods = list(config.x_axis_methods)
    if config.eci_file is None:
        eci_dependent_methods = {"eci", "eci_pc1"}
        dropped_methods = [method for method in x_axis_methods if method in eci_dependent_methods]
        if dropped_methods:
            x_axis_methods = [method for method in x_axis_methods if method not in eci_dependent_methods]
            print(
                f"{config.log_prefix} dropping ECI-dependent x-axis methods because --eci-file was not provided: "
                f"{sorted(set(dropped_methods))}"
            )
    if config.joint_x_axis is not None and config.joint_x_axis not in x_axis_methods:
        requested_joint_x_axis = str(config.joint_x_axis)
        if config.eci_file is None and requested_joint_x_axis in {"eci", "eci_pc1"}:
            raise ValueError(
                f"joint_x_axis={requested_joint_x_axis!r} requires --eci-file, but none was provided."
            )
        x_axis_methods.append(requested_joint_x_axis)

    needs_scores_df = any(method == "eci_pc1" for method in x_axis_methods)
    scores_df = load_baseline_scores() if needs_scores_df else None

    fractioner_label = config.fractioner or "all_shared_fractioners"
    output_dir = config.output_root / f"{config.benchmark}__{config.hint_type}__{fractioner_label}"
    if config.output_subdir is not None:
        output_dir = output_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_results, available_models = load_canonical_combo_results(
        benchmark=config.benchmark,
        hint_type=config.hint_type,
        fractioner=config.fractioner,
    )
    models = resolve_models_to_use(
        available_models=available_models,
        benchmark=config.benchmark,
        preferred_models=config.preferred_models,
    )
    excluded_model_names = {_normalize_model_name(model) for model in EXCLUDE_MODELS}
    excluded_models = sorted(
        model for model in models if _normalize_model_name(model) in excluded_model_names
    )
    if excluded_models:
        models = [
            model for model in models if _normalize_model_name(model) not in excluded_model_names
        ]
        print(
            f"{config.log_prefix} excluding models via EXCLUDE_MODELS: {excluded_models}"
        )
    if not models:
        raise ValueError(
            "All selected models were excluded by EXCLUDE_MODELS. "
            "Update EXCLUDE_MODELS or adjust preferred_models."
        )
    if config.eci_file is not None:
        eci_map = load_eci_map(Path(config.eci_file))
        models_with_eci = set(eci_map.keys())
        missing_eci_models = sorted(
            model for model in models if canonicalize_model_name(model) not in models_with_eci
        )
        if missing_eci_models:
            models = [
                model for model in models if canonicalize_model_name(model) in models_with_eci
            ]
            print(
                f"{config.log_prefix} skipping models missing ECI scores from {config.eci_file}: "
                f"{missing_eci_models}"
            )
        if not models:
            raise ValueError(
                "No models left after applying EXCLUDE_MODELS and ECI availability filtering."
            )
    if config.num_holdout_models < 0:
        raise ValueError(f"num_holdout_models must be >= 0, got {config.num_holdout_models}")
    if config.num_holdout_models > len(models):
        raise ValueError(
            f"num_holdout_models ({config.num_holdout_models}) cannot exceed "
            f"number of selected models ({len(models)})"
        )
    n_train_models = len(models) - int(config.num_holdout_models)
    train_model_order = list(models[:n_train_models])
    holdout_model_order = list(models[n_train_models:])
    print(
        f"{config.log_prefix} selected_models={len(models)} "
        f"models={models}"
    )
    base_rows = build_base_rows(
        combo_results=combo_results,
        models=models,
        fractioner=config.fractioner,
        benchmark=config.benchmark,
    )
    if not base_rows:
        raise ValueError("No usable rows found after combining hinted accuracy with x-axis data.")

    x_axes = build_x_axes_from_methods(
        methods=x_axis_methods,
        benchmark=config.benchmark,
        hint_type=config.hint_type,
        fractioner=config.fractioner,
        hint_fractions=config.hint_fractions,
        selected_models=list(models),
        fit_models=list(train_model_order),
        base_rows=base_rows,
        include_cross=bool(config.include_cross),
        lower_asymptote=float(config.joint_lower_asymptote),
        eci_path=config.eci_file,
        scores_df=scores_df,
        benchmark_order=PC_BENCHMARK_ORDER if needs_scores_df else None,
        canonicalize_model_name=canonicalize_model_name,
    )
    if config.restrict_models_to_x_axes:
        model_sets = [set(x_axis.model_to_x.keys()) for x_axis in x_axes]
        models = sorted(set(models).intersection(*model_sets)) if model_sets else []
        base_rows = [row for row in base_rows if str(row["model"]) in set(models)]
        train_model_order = [model for model in train_model_order if model in set(models)]
        holdout_model_order = [model for model in holdout_model_order if model in set(models)]
    x_axis_by_name = {x_axis.name: x_axis for x_axis in x_axes}

    if config.print_pca_report:
        for x_axis in x_axes:
            pca_result = get_pca_result(x_axis)
            if pca_result is None:
                continue
            print("")
            print(f"{config.log_prefix} PCA report for x_axis={x_axis.name}")
            summary_lines = (
                config.pca_summary_lines_fn(x_axis)
                if config.pca_summary_lines_fn is not None
                else _default_pca_summary_lines(config=config, x_axis=x_axis)
            )
            print_pca_report(
                result=pca_result,
                summary_lines=summary_lines,
            )

    if config.fractioner is None:
        print("")
        print(f"{config.log_prefix} skipping plots because --fractioner was not provided.")
        print(
            f"{config.log_prefix} the generic x-axis plots expect a single fractioner, "
            "not mixed shared fractioners."
        )
        return ScalingRunResult(
            x_axes=x_axes,
            output_dir=output_dir,
            plot_paths={},
            joint_metrics=None,
        )

    plot_paths = plot_accuracy_views_for_x_axes(
        base_rows=base_rows,
        x_axes=x_axes,
        benchmark=config.benchmark,
        hint_type=config.hint_type,
        fractioner=config.fractioner,
        output_dir=output_dir,
        facet_by=config.facet_by,
    )
    for x_axis_name, x_axis_plot_paths in sorted(plot_paths.items()):
        for plot_name, path in sorted(x_axis_plot_paths.items()):
            print(f"{config.log_prefix} x_axis={x_axis_name} plot[{plot_name}]={path}")

    joint_metrics: dict[str, object] | None = None
    joint_x_axes: list[XAxisSpec] = []
    if config.run_joint_for_all_x_axes:
        joint_x_axes = list(x_axes)
    elif config.joint_x_axis is not None:
        joint_x_axis_name = _normalize_joint_x_axis_name(str(config.joint_x_axis))
        selected_x_axis = x_axis_by_name.get(joint_x_axis_name)
        if selected_x_axis is None:
            raise ValueError(
                f"Requested joint x-axis {joint_x_axis_name} was not built. "
                f"Built x-axes: {sorted(x_axis_by_name)}"
            )
        joint_x_axes = [selected_x_axis]

    if joint_x_axes:
        joint_metrics = {}
        for selected_x_axis in joint_x_axes:
            joint_output_dir = output_dir / f"joint_scaling__{selected_x_axis.name}"
            joint_result = run_joint_scaling_for_x_axis(
                base_rows=base_rows,
                x_axis=selected_x_axis,
                models=models,
                train_models=list(train_model_order),
                holdout_models=list(holdout_model_order),
                output_dir=joint_output_dir,
                label=(
                    f"{config.benchmark} {config.fractioner} "
                    f"({selected_x_axis.label} joint scaling)"
                ),
                include_cross=bool(config.include_cross),
                lower_asymptote=float(config.joint_lower_asymptote),
                num_holdout_models=int(config.num_holdout_models),
            )
            joint_metrics[selected_x_axis.name] = joint_result
            print(f"{config.log_prefix} joint_scaling_output_dir[{selected_x_axis.name}]={joint_output_dir}")
            for name, path in sorted(joint_result["plot_paths"].items()):
                print(f"{config.log_prefix} joint_plot[{selected_x_axis.name}][{name}]={path}")
        comparison_plot_paths = _write_joint_x_axis_comparison_artifacts(
            joint_metrics=joint_metrics,
            x_axes=joint_x_axes,
            output_dir=output_dir,
            label=f"{config.benchmark} {config.fractioner} (joint scaling comparison)",
            log_prefix=config.log_prefix,
        )
        for name, path in sorted(comparison_plot_paths.items()):
            print(f"{config.log_prefix} joint_x_axis_comparison[{name}]={path}")

    return ScalingRunResult(
        x_axes=x_axes,
        output_dir=output_dir,
        plot_paths=plot_paths,
        joint_metrics=joint_metrics,
    )


def main() -> None:
    args = _parse_args()
    run_scaling(
        ScalingRunConfig(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            x_axis_methods=list(args.x_axis_methods),
            facet_by=str(args.facet_by),
            eci_file=None if args.eci_file is None else Path(args.eci_file),
            hint_fractions=list(DEFAULT_HINTED_PC_HINT_FRACTIONS),
            num_holdout_models=int(args.num_holdout_models),
            include_cross=True,
            print_pca_report=True,
            run_joint_for_all_x_axes=True,
            output_root=PLOTS_ROOT,
            log_prefix="[plot_scaling]",
            preferred_models=DEFAULT_MODELS_TO_USE,
        )
    )


if __name__ == "__main__":
    main()


"""

python -m runs.plot_scaling \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --num-holdout-models 0 \
    --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh__prompt_type_answer_only--hellaswag__split_validation--math__levels_5__fewshot_0--mmlu_5_shot__language_en_us__cot_true--piqa--winogrande__dataset_name_winogrande_xl__fewshot_5.csv



python -m runs.plot_scaling \
    --benchmark aime2025_2026 \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --num-holdout-models 0 

"""
 
