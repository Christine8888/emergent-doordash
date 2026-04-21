from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.fit_eci import EVAL_TO_ECI, load_baseline_scores
from src.joint_scaling_plots import (
    plot_accuracy_vs_x_by_hint,
    plot_accuracy_vs_x_by_hint_subplots_with_error_bars,
    plot_h0_fits_by_model_sweep,
    plot_joint_accuracy_vs_hint_by_model,
    plot_joint_accuracy_vs_x_by_hint,
    plot_joint_individual_fits_by_hint,
    plot_joint_model_sweep,
    plot_pca_component_weights,
    plot_pca_explained_variance,
)
from src.pca import (
    build_baseline_benchmark_pca_result,
    build_component_score_map,
    format_component_equation,
)
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
from src.scaling_data import (
    build_base_rows,
    build_x_rows,
    canonicalize_model_name,
    eci_benchmark_label,
    load_canonical_combo_results,
    load_eci_map,
    resolve_models_to_use,
)
from src.sigmoid_fits import fit_plot_sigmoid


PLOTS_ROOT = Path("plots/joint_scaling_plots")
PC_BENCHMARK_ORDER = [EVAL_TO_ECI[eval_name] for eval_name in EVAL_TO_ECI]
JOINT_LOWER_ASYMPTOTE = 0.0
MODELS_TO_USE: list[str] | None = [
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs capability, one curve per hint fraction.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument("--eci-file", type=str, required=True)
    parser.add_argument("--num-holdout-models", type=int, default=0)
    parser.add_argument(
        "--include-cross",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the capability-by-hint interaction term in the joint fit.",
    )
    return parser.parse_args()

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_eci_joint_scaling(
    *,
    df: pd.DataFrame,
    models: list[str],
    output_dir: Path,
    label: str,
    include_cross: bool,
    lower_asymptote: float | None,
    num_holdout_models: int,
) -> dict[str, Any]:
    x_field = "eci"
    x_label = "ECI"
    if num_holdout_models < 0:
        raise ValueError(f"num_holdout_models must be >= 0, got {num_holdout_models}")
    if num_holdout_models > len(models):
        raise ValueError(
            f"num_holdout_models ({num_holdout_models}) cannot exceed number of models ({len(models)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_sorted_by_eci = sorted(
        models,
        key=lambda model: float(df[df["model"] == model][x_field].iloc[0]),
    )
    holdout_models = set(models_sorted_by_eci[-num_holdout_models:]) if num_holdout_models > 0 else set()
    train_models = (
        set(models_sorted_by_eci[:-num_holdout_models])
        if num_holdout_models > 0
        else set(models_sorted_by_eci)
    )
    filename_suffix = f"__n_test_{len(holdout_models)}"

    df = df.copy()
    df["split"] = df["model"].map(lambda model: "train" if model in train_models else "test")

    joint_result = fit_joint_sigmoid_model(
        df=df,
        fit_models=train_models,
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
        fit_models=train_models,
        lower=lower_asymptote,
    )
    individual_by_model = fit_individual_sigmoids_by_model(
        df=df,
        fit_models=None,
        lower=lower_asymptote,
    )

    plot_paths = {
        "accuracy_vs_eci_by_hint": str(
            plot_joint_accuracy_vs_x_by_hint(
                df=df,
                x_field=x_field,
                x_label=x_label,
                joint_predict_fn=joint_result["predict"],
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"accuracy_vs_eci_by_hint{filename_suffix}",
            )
        ),
        "individual_fits_by_hint": str(
            plot_joint_individual_fits_by_hint(
                df=df,
                x_field=x_field,
                x_label=x_label,
                joint_predict_fn=joint_result["predict"],
                individual_by_hint_all=individual_by_hint_all,
                individual_by_hint_train=individual_by_hint_train,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"individual_fits_by_hint{filename_suffix}",
            )
        ),
        "accuracy_vs_hint_by_model": str(
            plot_joint_accuracy_vs_hint_by_model(
                df=df,
                model_to_x={
                    str(model): float(x_value)
                    for model, x_value in zip(df["model"], df[x_field])
                },
                x_label=x_label,
                joint_predict_fn=joint_result["predict"],
                individual_by_model=individual_by_model,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"accuracy_vs_hint_by_model{filename_suffix}",
            )
        ),
    }

    panels = build_h0_sweep_panels(
        df=df,
        x_field=x_field,
        models_sorted_by_x=models_sorted_by_eci,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
    )

    plot_paths["h0_fits_by_model_sweep"] = str(
        plot_h0_fits_by_model_sweep(
            panels=panels,
            x_label=x_label,
            label=label,
            output_dir=output_dir,
            filename_stem="h0_fits_by_model_sweep",
        )
    )

    sweep_df = run_joint_model_sweep(
        df=df,
        x_field=x_field,
        models_sorted_by_x=models_sorted_by_eci,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
    )
    plot_paths["model_sweep"] = str(
        plot_joint_model_sweep(
            sweep_df=sweep_df,
            x_label=x_label,
            label=label,
            output_dir=output_dir,
            filename_stem="model_sweep",
        )
    )

    metrics = {
        "joint_equation": joint_equation,
        "joint_params": [float(value) for value in np.asarray(joint_result["params"], dtype=float)],
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "optimizer_success": bool(joint_result["optimizer_success"]),
        "optimizer_status": int(joint_result["optimizer_status"]),
        "optimizer_message": str(joint_result["optimizer_message"]),
        "n_train_models": int(len(train_models)),
        "n_test_models": int(len(holdout_models)),
        "rms_train": compute_rms_joint(
            joint_result=joint_result,
            df=df,
            x_field=x_field,
            models=train_models,
        ),
        "rms_test": compute_rms_joint(
            joint_result=joint_result,
            df=df,
            x_field=x_field,
            models=holdout_models,
        ) if holdout_models else float("nan"),
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
            models=train_models,
        ),
        "rms_indiv_test": compute_rms_individual_by_hint(
            individual_by_hint=individual_by_hint_train,
            df=df,
            x_field=x_field,
            models=holdout_models,
        ) if holdout_models else float("nan"),
        "rms_indiv_all": compute_rms_individual_by_hint(
            individual_by_hint=individual_by_hint_train,
            df=df,
            x_field=x_field,
            models=None,
        ),
        "train_models": models_sorted_by_eci[: len(train_models)],
        "holdout_models": models_sorted_by_eci[len(train_models) :],
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

    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    if not 0.0 <= float(JOINT_LOWER_ASYMPTOTE) < 1.0:
        raise ValueError(
            f"JOINT_LOWER_ASYMPTOTE must be in [0, 1), got {JOINT_LOWER_ASYMPTOTE}"
        )

    scores_df = load_baseline_scores()
    eci_path = Path(args.eci_file)
    eci_map = load_eci_map(eci_path)
    eci_benchmark_label_text = eci_benchmark_label(eci_path)
    baseline_pca_result = build_baseline_benchmark_pca_result(
        scores_df=scores_df,
        benchmark_order=PC_BENCHMARK_ORDER,
        canonicalize_model_name=canonicalize_model_name,
    )
    pc1_map = build_component_score_map(
        baseline_pca_result,
        component_idx=0,
    )
    pc_benchmark_label = ", ".join(eval_name for eval_name in EVAL_TO_ECI)
    pc1_equation = format_component_equation(
        baseline_pca_result,
        component_idx=0,
    )

    combo_results, models = load_canonical_combo_results(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    models = resolve_models_to_use(
        available_models=models,
        benchmark=args.benchmark,
        preferred_models=MODELS_TO_USE,
    )
    print(
        f"[plot_accuracy_vs_eci_by_hint] selected_models={len(models)} "
        f"models={models}"
    )
    base_rows = build_base_rows(
        combo_results=combo_results,
        models=models,
        fractioner=args.fractioner,
        benchmark=args.benchmark,
    )

    if not base_rows:
        raise ValueError("No usable rows found after combining hinted accuracy with capability data.")

    output_dir = PLOTS_ROOT / f"{args.benchmark}__{args.hint_type}__{args.fractioner}"
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_output_dir = output_dir / "joint_scaling_eci"
    plot_pca_component_weights(
        components=baseline_pca_result.components,
        benchmarks=baseline_pca_result.feature_names,
        output_path=output_dir / "pca_component_weights.png",
    )
    plot_pca_explained_variance(
        explained_variance_ratio=baseline_pca_result.explained_variance_ratio,
        output_path=output_dir / "pca_explained_variance.png",
    )
    views = [
        ("eci", "ECI", eci_map, eci_benchmark_label_text, None),
        ("pc1", "PC1", pc1_map, pc_benchmark_label, pc1_equation),
    ]
    for (
        x_method,
        x_label,
        x_map,
        x_benchmark_label,
        x_equation,
    ) in views:
        rows = build_x_rows(
            base_rows=base_rows,
            x_map=x_map,
        )
        if not rows:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] no rows for x_method={x_method}"
            )
            continue

        missing_models = sorted({str(row["model"]) for row in base_rows} - set(x_map.keys()))
        if missing_models:
            raise ValueError(
                f"Configured models missing {x_method} values: {missing_models}"
            )

        output_path = plot_accuracy_vs_x_by_hint(
            rows=rows,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            x_method=x_method,
            x_label=x_label,
            x_benchmark_label=x_benchmark_label,
            x_equation=x_equation,
            output_dir=output_dir,
            fit_series_fn=fit_plot_sigmoid,
        )
        print(f"[plot_accuracy_vs_eci_by_hint] {output_path}")
        per_hint_output_path = plot_accuracy_vs_x_by_hint_subplots_with_error_bars(
            rows=rows,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            x_method=x_method,
            x_label=x_label,
            x_benchmark_label=x_benchmark_label,
            x_equation=x_equation,
            output_dir=output_dir,
            fit_series_fn=fit_plot_sigmoid,
        )
        print(f"[plot_accuracy_vs_eci_by_hint] {per_hint_output_path}")

    joint_df = build_joint_scaling_df(
        base_rows=base_rows,
        x_map=eci_map,
        x_field="eci",
        train_models=set(),
    )
    joint_metrics = _run_eci_joint_scaling(
        df=joint_df,
        models=models,
        output_dir=joint_output_dir,
        label=f"{args.benchmark} {args.fractioner} (ECI joint scaling)",
        include_cross=bool(args.include_cross),
        lower_asymptote=float(JOINT_LOWER_ASYMPTOTE),
        num_holdout_models=int(args.num_holdout_models),
    )
    print(f"[plot_accuracy_vs_eci_by_hint] joint_scaling_output_dir={joint_output_dir}")
    for name, path in sorted(joint_metrics["plot_paths"].items()):
        print(f"[plot_accuracy_vs_eci_by_hint] joint_plot[{name}]={path}")


if __name__ == "__main__":
    # python -m runs.plot_scaling_using_eci --benchmark aime2025_2026 --hint-type answer_not_revealed --include-cross --fractioner mask_word --num-holdout-models 4 --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh__prompt_type_answer_only--hellaswag__split_validation--math__levels_5__fewshot_0--mmlu_5_shot__language_en_us__cot_true--piqa--winogrande__dataset_name_winogrande_xl__fewshot_5.csv


    # python -m runs.plot_scaling_using_eci --benchmark gpqa --hint-type answer_not_revealed --fractioner mask_word
    main()
