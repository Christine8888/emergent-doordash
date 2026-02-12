"""Experiment runner for 20260202 hint-fraction scaling plots.

Usage:
  python suze_experiments/20260202/experiments.py --list
  python suze_experiments/20260202/experiments.py -e joint_scaling_gpqa_solution_intext_masked
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from plot_helpers import run_joint_scaling_plots

# ------------------------------ constants ------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent  # repo root (emergent-doordash/)

RESULTS_ROOT = EXPERIMENT_DIR / "results"


def _ensure_project_root_on_path() -> None:
    # This script is outside the packaged `src/` tree, so we add the repo root so
    # `import src....` works without `pip install -e .`.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------ run directory helpers ------------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_dir(*, run_name: str) -> Path:
    run_dir = RESULTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ------------------------------ experiments ------------------------------

def joint_scaling_gpqa_train_on_all() -> None:
    # python experiments.py -e joint_scaling_gpqa_train_on_all
    _ensure_project_root_on_path()

    # User-defined run name: edit here per experiment.
    run_name = "joint_scaling_gpqa_train_on_all" + "_feb_5"
    run_dir = make_run_dir(run_name=run_name)


    run_joint_scaling_plots(
        base_folder=PROJECT_ROOT / "christine_experiments/20251113/results",
        eci_file=PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv",
        eval_name="gpqa",
        solver="solution_intext_masked",
        condition="0shot",
        label="GPQA solution intext masked",
        all_models=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ],
        num_holdout_models=0,
        hint_fractions=[round(i / 20.0, 2) for i in range(21)],
        eval_hints_for_sweep=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_cross=True,
        lower_asymptote=0.2,
        hint_transform="identity",
        output_dir=run_dir,
    )

def joint_scaling_gpqa_train_on_some() -> None:
    # python experiments.py -e joint_scaling_gpqa_train_on_some
    _ensure_project_root_on_path()

    # User-defined run name: edit here per experiment.
    run_name = "joint_scaling_gpqa_train_on_some" + "_feb_5"
    run_dir = make_run_dir(run_name=run_name)


    run_joint_scaling_plots(
        base_folder=PROJECT_ROOT / "christine_experiments/20251113/results",
        eci_file=PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv",
        eval_name="gpqa",
        solver="solution_intext_masked",
        condition="0shot",
        label="GPQA solution intext masked",
        all_models=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ],
        num_holdout_models=9,
        hint_fractions=[round(i / 20.0, 2) for i in range(21)],
        eval_hints_for_sweep=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_cross=True,
        lower_asymptote=0.2,
        hint_transform="identity",
        output_dir=run_dir,
    )


def joint_scaling_learned_hint_fixed_endpoints() -> None:
    # python experiments.py -e joint_scaling_learned_hint_fixed_endpoints
    _ensure_project_root_on_path()

    # User-defined run name: edit here per experiment.
    run_name = "joint_scaling_learned_hint_fixed_endpoints" + "_feb_5"
    run_dir = make_run_dir(run_name=run_name)

    run_joint_scaling_plots(
        base_folder=PROJECT_ROOT / "christine_experiments/20251113/results",
        eci_file=PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv",
        eval_name="gpqa",
        solver="solution_intext_masked",
        condition="0shot",
        label="GPQA solution intext masked",
        all_models=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ],
        num_holdout_models=0,
        hint_fractions=[round(i / 20.0, 2) for i in range(21)],
        eval_hints_for_sweep=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_cross=True,
        lower_asymptote=0.2,
        hint_transform="learned_piecewise_linear_fixed_endpoints",
        hint_knots=[round(i / 20.0, 2) for i in range(21)],
        output_dir=run_dir,
    )


def joint_scaling_learned_hint_free_endpoints() -> None:
    # python experiments.py -e joint_scaling_learned_hint_free_endpoints
    _ensure_project_root_on_path()

    # User-defined run name: edit here per experiment.
    run_name = "joint_scaling_learned_hint_free_endpoints" + "_feb_5"
    run_dir = make_run_dir(run_name=run_name)

    run_joint_scaling_plots(
        base_folder=PROJECT_ROOT / "christine_experiments/20251113/results",
        eci_file=PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv",
        eval_name="gpqa",
        solver="solution_intext_masked",
        condition="0shot",
        label="GPQA solution intext masked",
        all_models=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ],
        num_holdout_models=0,
        hint_fractions=[round(i / 20.0, 2) for i in range(21)],
        eval_hints_for_sweep=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_cross=True,
        lower_asymptote=0.2,
        hint_transform="learned_piecewise_linear_free_endpoints",
        # hint_knots=[0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0],
        hint_knots=[round(i / 20.0, 2) for i in range(21)],
        output_dir=run_dir,
    )


def joint_scaling_learned_hint_fixed_endpoints_train_on_some() -> None:
    # python experiments.py -e joint_scaling_learned_hint_fixed_endpoints_train_on_some
    _ensure_project_root_on_path()

    # User-defined run name: edit here per experiment.
    run_name = "joint_scaling_learned_hint_fixed_endpoints_train_on_some" + "_feb_5"
    run_dir = make_run_dir(run_name=run_name)


    run_joint_scaling_plots(
        base_folder=PROJECT_ROOT / "christine_experiments/20251113/results",
        eci_file=PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv",
        eval_name="gpqa",
        solver="solution_intext_masked",
        condition="0shot",
        label="GPQA solution intext masked",
        all_models=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ],
        num_holdout_models=9,
        hint_fractions=[round(i / 20.0, 2) for i in range(21)],
        eval_hints_for_sweep=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_cross=True,
        lower_asymptote=0.2,
        hint_transform="learned_piecewise_linear_fixed_endpoints",
        hint_knots=[round(i / 20.0, 2) for i in range(21)],
        output_dir=run_dir,
    )


def joint_scaling_learned_hint_fixed_endpoints_sequential_solution() -> None:
    # python experiments.py -e joint_scaling_learned_hint_fixed_endpoints_sequential_solution
    _ensure_project_root_on_path()

    # User-defined run name: edit here per experiment.
    run_name = "joint_scaling_learned_hint_fixed_endpoints_sequential_solution" + "_feb_5"
    run_dir = make_run_dir(run_name=run_name)

    run_joint_scaling_plots(
        base_folder=PROJECT_ROOT / "christine_experiments/20251113/results",
        eci_file=PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv",
        eval_name="gpqa",
        solver="solution_intext_sequential",
        condition="0shot",
        label="GPQA solution intext sequential",
        all_models=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-0.6B",
            "Qwen3-1.7B",
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ],
        num_holdout_models=0,
        hint_fractions=[round(i / 20.0, 2) for i in range(21)],
        eval_hints_for_sweep=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_cross=True,
        lower_asymptote=0.2,
        hint_transform="learned_piecewise_linear_fixed_endpoints",
        hint_knots=[round(i / 20.0, 2) for i in range(21)],
        output_dir=run_dir,
    )



EXPERIMENTS = {
    "joint_scaling_gpqa_train_on_all": joint_scaling_gpqa_train_on_all,
    "joint_scaling_gpqa_train_on_some": joint_scaling_gpqa_train_on_some,
    "joint_scaling_learned_hint_fixed_endpoints": joint_scaling_learned_hint_fixed_endpoints,
    "joint_scaling_learned_hint_free_endpoints": joint_scaling_learned_hint_free_endpoints,
    "joint_scaling_learned_hint_fixed_endpoints_train_on_some": joint_scaling_learned_hint_fixed_endpoints_train_on_some,
    "joint_scaling_learned_hint_fixed_endpoints_sequential_solution": joint_scaling_learned_hint_fixed_endpoints_sequential_solution,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 20260202 experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Available experiments:",
                *[f"  - {name}" for name in EXPERIMENTS.keys()],
                "",
                "Examples:",
                "  python suze_experiments/20260202/experiments.py --list",
                "  python suze_experiments/20260202/experiments.py -e joint_scaling_gpqa_solution_intext_masked",
                "  python suze_experiments/20260202/experiments.py --all",
            ]
        ),
    )
    parser.add_argument("--experiment", "-e", choices=list(EXPERIMENTS.keys()), help="Experiment to run")
    parser.add_argument("--all", "-all", action="store_true", help="Run all experiments")
    parser.add_argument("--list", "-l", action="store_true", help="List available experiments")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS.keys():
            print(f"  - {name}")
        return

    if args.all and args.experiment:
        parser.error("Please specify only one of --all or --experiment / -e")

    if args.all:
        for name, experiment_fn in EXPERIMENTS.items():
            print(f"Running experiment: {name}")
            experiment_fn()
        return

    if not args.experiment:
        parser.error("Please specify an experiment with --experiment / -e (use --list to see options)")

    print(f"Running experiment: {args.experiment}")
    EXPERIMENTS[args.experiment]()


if __name__ == "__main__":
    main()

