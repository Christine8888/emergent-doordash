from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.scaling_common import DEFAULT_MODELS_TO_USE, ScalingRunConfig, run_scaling


PLOTS_ROOT = Path("plots/joint_scaling_plots")


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


def main() -> None:
    args = _parse_args()
    run_scaling(
        ScalingRunConfig(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            x_axis_methods=["eci", "baseline_pc1"],
            joint_x_axis="eci",
            eci_file=Path(args.eci_file),
            num_holdout_models=int(args.num_holdout_models),
            include_cross=bool(args.include_cross),
            output_root=PLOTS_ROOT,
            log_prefix="[plot_accuracy_vs_eci_by_hint]",
            preferred_models=DEFAULT_MODELS_TO_USE,
        )
    )


if __name__ == "__main__":
    main()
