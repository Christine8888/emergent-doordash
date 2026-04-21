from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.scaling_common import ScalingRunConfig, run_scaling
from src.hinted_accuracy import EXPECTED_FRACTIONS
from src.x_axes import XAxisSpec, get_pca_result


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
    parser.add_argument(
        "--run-joint-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to also run the heavier joint-scaling analysis using hinted PC1 as x.",
    )
    parser.add_argument("--num-holdout-models", type=int, default=0)
    parser.add_argument(
        "--include-cross",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the capability-by-hint interaction term in the joint fit.",
    )
    return parser.parse_args()


def _hint_fraction_slug(hint_fractions: list[float]) -> str:
    return "__".join(f"h_{fraction:.1f}".replace(".", "p") for fraction in hint_fractions)


def _build_pca_summary_lines(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    x_axis_hint_fractions: list[float],
    x_axis: XAxisSpec,
) -> list[str]:
    pca_result = get_pca_result(x_axis)
    if pca_result is None:
        return [
            f"benchmark: {benchmark}",
            f"hint_type: {hint_type}",
            f"fractioner: {fractioner or 'all_shared_fractioners'}",
            f"x_axis: {x_axis.name}",
        ]

    plot_hint_fractions = list(EXPECTED_FRACTIONS)
    return [
        f"benchmark: {benchmark}",
        f"hint_type: {hint_type}",
        f"fractioner: {fractioner or 'all_shared_fractioners'}",
        "x_axis_hint_fractions: " + ", ".join(f"{fraction:.1f}" for fraction in x_axis_hint_fractions),
        "plot_hint_fractions: " + ", ".join(f"{fraction:.1f}" for fraction in plot_hint_fractions),
        f"n_models: {len(pca_result.model_names)}",
        f"n_features: {len(pca_result.feature_names)}",
        "shared_fractioners: "
        + ", ".join(str(value) for value in pca_result.metadata.get("shared_fractioners", [])),
    ]


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
    run_scaling(
        ScalingRunConfig(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            x_axis_methods=["hinted_pc1"],
            joint_x_axis="hinted_pc1" if args.run_joint_scaling else None,
            hint_fractions=x_axis_hint_fractions,
            num_holdout_models=int(args.num_holdout_models),
            include_cross=bool(args.include_cross),
            print_pca_report=True,
            output_root=PLOTS_ROOT,
            output_subdir=Path(_hint_fraction_slug(x_axis_hint_fractions)),
            log_prefix="[pca_on_hinted]",
            preferred_models=None,
            restrict_models_to_x_axes=True,
            pca_summary_lines_fn=lambda x_axis: _build_pca_summary_lines(
                benchmark=args.benchmark,
                hint_type=args.hint_type,
                fractioner=args.fractioner,
                x_axis_hint_fractions=x_axis_hint_fractions,
                x_axis=x_axis,
            ),
        )
    )


if __name__ == "__main__":
    main()
