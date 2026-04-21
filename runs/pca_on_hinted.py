from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.hinted_accuracy import EXPECTED_FRACTIONS, load_results_with_ci_for_combo
from src.joint_scaling_plots import (
    plot_accuracy_vs_x_by_hint,
    plot_accuracy_vs_x_by_hint_subplots_with_error_bars,
    plot_pca_component_weights,
    plot_pca_explained_variance,
)
from src.pca import (
    PCAResult,
    build_component_score_map,
    build_hinted_pca_result,
    format_component_equation,
)
from src.sigmoid_fits import fit_plot_sigmoid


PLOTS_ROOT = Path("plots/pca_on_hinted")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA on the same hinted-accuracy results used by the plot interface."
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument(
        "--fractioner",
        type=str,
        default=None,
        help="Optional specific fractioner. If omitted, use the fractioners discovered per model.",
    )
    parser.add_argument(
        "--hint-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Optional hint fractions to include, e.g. --hint-fractions 0.0 0.5 1.0",
    )
    return parser.parse_args()

def _hint_fraction_slug(hint_fractions: list[float]) -> str:
    return "__".join(f"h_{fraction:.1f}".replace(".", "p") for fraction in hint_fractions)


def _print_component_loadings(
    *,
    component_idx: int,
    result: PCAResult,
) -> None:
    if component_idx >= result.components.shape[0]:
        return

    print(f"PC{component_idx + 1} loadings")
    for feature_name, weight in sorted(
        zip(result.feature_names, result.components[component_idx], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    ):
        print(f"{feature_name}: {float(weight):+.4f}")
    print("")


def _print_feature_summary(
    *,
    result: PCAResult,
) -> None:
    means = result.feature_means
    stds = result.feature_stds
    component_vectors = [
        result.components[idx]
        if result.components.shape[0] > idx
        else np.zeros(len(result.feature_names), dtype=float)
        for idx in range(4)
    ]

    print("Feature summary")
    print(
        "feature  mean_acc  std_acc  "
        "pc1_loading  pc2_loading  pc3_loading  pc4_loading  "
        "pc1_delta_acc  pc2_delta_acc  pc3_delta_acc  pc4_delta_acc"
    )
    print(
        "-------  --------  -------  "
        "-----------  -----------  -----------  -----------  "
        "-------------  -------------  -------------  -------------"
    )
    for idx, feature_name in enumerate(result.feature_names):
        deltas = [float(vector[idx]) * float(stds[idx]) for vector in component_vectors]
        print(
            f"{feature_name}  "
            f"{float(means[idx]):.4f}  "
            f"{float(stds[idx]):.4f}  "
            f"{float(component_vectors[0][idx]):+.4f}  "
            f"{float(component_vectors[1][idx]):+.4f}  "
            f"{float(component_vectors[2][idx]):+.4f}  "
            f"{float(component_vectors[3][idx]):+.4f}  "
            f"{deltas[0]:+.4f}  "
            f"{deltas[1]:+.4f}  "
            f"{deltas[2]:+.4f}  "
            f"{deltas[3]:+.4f}"
        )
    print("")


def _print_ranked_component_scores(
    *,
    result: PCAResult,
    component_idx: int,
) -> None:
    if component_idx >= result.scores.shape[1]:
        return

    print(f"PC{component_idx + 1} ranking")
    for rank, (model_idx, model) in enumerate(
        sorted(
            enumerate(result.model_names),
            key=lambda item: float(result.scores[item[0], component_idx]),
            reverse=True,
        ),
        start=1,
    ):
        print(f"{rank:>2}. {model}: {float(result.scores[model_idx, component_idx]):+.4f}")
    print("")


def main() -> None:
    args = _parse_args()
    x_axis_hint_fractions = (
        sorted({float(fraction) for fraction in args.hint_fractions})
        if args.hint_fractions is not None
        else list(EXPECTED_FRACTIONS)
    )
    invalid_fractions = [
        fraction for fraction in x_axis_hint_fractions if fraction not in EXPECTED_FRACTIONS
    ]
    if invalid_fractions:
        raise ValueError(
            f"Unsupported hint fractions: {invalid_fractions}. "
            f"Expected subset of {EXPECTED_FRACTIONS}."
        )
    plot_hint_fractions = list(EXPECTED_FRACTIONS)

    pca_result = build_hinted_pca_result(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        hint_fractions=x_axis_hint_fractions,
    )
    shared_fractioners = list(pca_result.metadata.get("shared_fractioners", []))
    models = list(pca_result.model_names)

    print("PCA summary")
    print(f"benchmark: {args.benchmark}")
    print(f"hint_type: {args.hint_type}")
    print(f"fractioner: {args.fractioner or 'all_shared_fractioners'}")
    print(
        "x_axis_hint_fractions: "
        + ", ".join(f"{fraction:.1f}" for fraction in x_axis_hint_fractions)
    )
    print(
        "plot_hint_fractions: "
        + ", ".join(f"{fraction:.1f}" for fraction in plot_hint_fractions)
    )
    print(f"n_models: {len(models)}")
    print(f"n_features: {len(pca_result.feature_names)}")
    print(f"shared_fractioners: {', '.join(shared_fractioners)}")
    print("")

    print("Explained variance ratio")
    cumulative_explained_variance = np.cumsum(pca_result.explained_variance_ratio)
    for idx, value in enumerate(pca_result.explained_variance_ratio[:5], start=1):
        print(
            f"PC{idx}: {float(value):.4f} "
            f"(cumulative={float(cumulative_explained_variance[idx - 1]):.4f})"
        )
    print("")

    _print_component_loadings(
        component_idx=0,
        result=pca_result,
    )
    _print_component_loadings(
        component_idx=1,
        result=pca_result,
    )
    _print_component_loadings(
        component_idx=2,
        result=pca_result,
    )
    _print_component_loadings(
        component_idx=3,
        result=pca_result,
    )
    _print_feature_summary(
        result=pca_result,
    )
    _print_ranked_component_scores(
        result=pca_result,
        component_idx=0,
    )
    _print_ranked_component_scores(
        result=pca_result,
        component_idx=1,
    )
    _print_ranked_component_scores(
        result=pca_result,
        component_idx=2,
    )
    _print_ranked_component_scores(
        result=pca_result,
        component_idx=3,
    )
    pc1_map = build_component_score_map(pca_result, component_idx=0)

    print("Model scores")
    for idx, model in sorted(
        enumerate(models),
        key=lambda item: float(pca_result.scores[item[0], 0]),
    ):
        if pca_result.scores.shape[1] > 1:
            print(
                f"{model}: pc1={float(pca_result.scores[idx, 0]):+.4f} "
                f"pc2={float(pca_result.scores[idx, 1]):+.4f}"
            )
        else:
            print(f"{model}: pc1={float(pca_result.scores[idx, 0]):+.4f}")

    if args.fractioner is None:
        print("")
        print("[pca_on_hinted] skipping plots because --fractioner was not provided.")
        print(
            "[pca_on_hinted] the generic x-axis plots expect a single fractioner, "
            "not mixed shared fractioners."
        )
        return

    combo_results = load_results_with_ci_for_combo(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    plot_rows: list[dict[str, float | str]] = []
    for model in models:
        fraction_map = combo_results.get(model)
        if fraction_map is None:
            continue
        missing = [
            hint_fraction
            for hint_fraction in plot_hint_fractions
            if float(hint_fraction) not in fraction_map
        ]
        if missing:
            raise ValueError(
                f"Missing hint fractions for model={model}: {missing}. "
                f"Expected subset of {plot_hint_fractions}."
            )
        for hint_fraction in plot_hint_fractions:
            stats = fraction_map[float(hint_fraction)]
            plot_rows.append(
                {
                    "model": model,
                    "hint_fraction": float(hint_fraction),
                    "accuracy": float(stats["accuracy"]),
                    "ci_low": float(stats["ci_low"]),
                    "ci_high": float(stats["ci_high"]),
                    "x_value": float(pc1_map[model]),
                }
            )

    if not plot_rows:
        print("")
        print("[pca_on_hinted] no plot rows available for the requested fractioner.")
        return

    output_dir = (
        PLOTS_ROOT
        / f"{args.benchmark}__{args.hint_type}__{args.fractioner}"
        / _hint_fraction_slug(x_axis_hint_fractions)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pca_component_weights(
        components=pca_result.components,
        benchmarks=pca_result.feature_names,
        output_path=output_dir / "pca_component_weights.png",
    )
    plot_pca_explained_variance(
        explained_variance_ratio=pca_result.explained_variance_ratio,
        output_path=output_dir / "pca_explained_variance.png",
    )
    pc1_plot_path = plot_accuracy_vs_x_by_hint(
        rows=plot_rows,
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        x_method="pc1_on_hinted",
        x_label="PC1 on hinted accuracies",
        x_benchmark_label=", ".join(
            f"{fraction:.1f}" for fraction in x_axis_hint_fractions
        ),
        x_equation=format_component_equation(
            pca_result,
            component_idx=0,
        ),
        output_dir=output_dir,
        fit_series_fn=fit_plot_sigmoid,
    )
    pc1_subplots_path = plot_accuracy_vs_x_by_hint_subplots_with_error_bars(
        rows=plot_rows,
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        x_method="pc1_on_hinted",
        x_label="PC1 on hinted accuracies",
        x_benchmark_label=", ".join(
            f"{fraction:.1f}" for fraction in x_axis_hint_fractions
        ),
        x_equation=format_component_equation(
            pca_result,
            component_idx=0,
        ),
        output_dir=output_dir,
        fit_series_fn=fit_plot_sigmoid,
    )
    print("")
    print(f"[pca_on_hinted] wrote_pca_component_weights= {output_dir / 'pca_component_weights.png'}")
    print(f"[pca_on_hinted] wrote_pca_explained_variance= {output_dir / 'pca_explained_variance.png'}")
    print(f"[pca_on_hinted] wrote_plot= {pc1_plot_path}")
    print(f"[pca_on_hinted] wrote_plot= {pc1_subplots_path}")


if __name__ == "__main__":
    # python -m runs.pca_on_hinted --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word --hint-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    main()
