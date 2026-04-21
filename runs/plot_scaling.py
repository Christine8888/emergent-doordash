from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.fit_eci import EVAL_TO_ECI, load_baseline_scores
from src.hinted_accuracy import EXPECTED_FRACTIONS
from src.pca import print_pca_report
from src.scaling_data import (
    build_base_rows,
    canonicalize_model_name,
    load_canonical_combo_results,
    resolve_models_to_use,
)
from src.scaling_runner import plot_accuracy_views_for_x_axes
from src.joint_scaling_runner import run_joint_scaling_for_x_axis
from src.x_axes import SUPPORTED_X_AXIS_METHODS, XAxisSpec, build_x_axes_from_methods, get_pca_result


PLOTS_ROOT = Path("plots/scaling_plots")
PC_BENCHMARK_ORDER = [EVAL_TO_ECI[eval_name] for eval_name in EVAL_TO_ECI]
DEFAULT_JOINT_LOWER_ASYMPTOTE = 0.0
DEFAULT_HINTED_PC_HINT_FRACTIONS = [fraction for fraction in EXPECTED_FRACTIONS if fraction > 0.0]
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
]


@dataclass
class ScalingRunConfig:
    benchmark: str
    hint_type: str
    fractioner: str | None
    x_axis_methods: list[str]
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
        default=["eci", "eci_pc1"],
        choices=SUPPORTED_X_AXIS_METHODS,
    )
    parser.add_argument("--eci-file", type=str, default=None)
    parser.add_argument("--num-holdout-models", type=int, default=0)
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
    if joint_x_axis in {"eci", "eci_pc1", "hinted_pc1", "hinted_pc12_theta"}:
        return joint_x_axis
    raise ValueError(f"Unsupported joint x-axis: {joint_x_axis}")


def run_scaling(config: ScalingRunConfig) -> ScalingRunResult:
    if not 0.0 <= float(config.joint_lower_asymptote) < 1.0:
        raise ValueError(
            "joint_lower_asymptote must be in [0, 1), "
            f"got {config.joint_lower_asymptote}"
        )

    x_axis_methods = list(config.x_axis_methods)
    if config.joint_x_axis is not None and config.joint_x_axis not in x_axis_methods:
        x_axis_methods.append(str(config.joint_x_axis))

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
    --x-axis-methods eci eci_pc1 hinted_pc1 hinted_pc12_theta \
    --num-holdout-models 0 \
    --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh__prompt_type_answer_only--hellaswag__split_validation--math__levels_5__fewshot_0--mmlu_5_shot__language_en_us__cot_true--piqa--winogrande__dataset_name_winogrande_xl__fewshot_5.csv

"""
