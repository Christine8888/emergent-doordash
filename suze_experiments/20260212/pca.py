"""Wrapper script for PCA + PC-capability scaling (moved to 20260202).

This script originally lived entirely under `suze_experiments/20260212/`.
All PCA + PC-capability joint-scaling functionality has been moved into the
`suze_experiments/20260202/` experiment framework (to share the plotting +
metrics pipeline with the ECI-based experiments).

This file remains as a convenience wrapper so existing commands keep working.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTDIR = PROJECT_ROOT / "suze_experiments/20260212"

DEFAULT_BASELINE_FOLDER = PROJECT_ROOT / "christine_experiments/20251113/baseline"
DEFAULT_RESULTS_BASE_FOLDER = PROJECT_ROOT / "christine_experiments/20251113/results"
DEFAULT_ECI_FILE = PROJECT_ROOT / "christine_experiments/20260129_fitting/eci_model_capabilities.csv"

DEFAULT_HINT_FRACTIONS = [round(i / 20.0, 2) for i in range(21)]
DEFAULT_EVAL_HINTS_FOR_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _ensure_import_paths() -> None:
    # Allow importing src.* and 20260202 helpers.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    exp_20260202 = PROJECT_ROOT / "suze_experiments/20260202"
    if str(exp_20260202) not in sys.path:
        sys.path.insert(0, str(exp_20260202))


def _parse_csv_floats(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def main() -> None:
    _ensure_import_paths()

    parser = argparse.ArgumentParser(description="PCA + PC-based joint scaling (wrapper)")
    parser.add_argument("--mode", choices=["pca", "joint_scaling"], default="pca")

    # PCA args
    parser.add_argument("--baseline-folder", type=str, default=str(DEFAULT_BASELINE_FOLDER))
    parser.add_argument("--n-components", type=int, default=5)

    # Joint scaling args
    parser.add_argument("--results-base-folder", type=str, default=str(DEFAULT_RESULTS_BASE_FOLDER))
    parser.add_argument("--eci-file", type=str, default=str(DEFAULT_ECI_FILE))
    parser.add_argument("--eval-name", type=str, default="gpqa")
    parser.add_argument("--solver", type=str, default="solution_intext_masked")
    parser.add_argument("--condition", type=str, default="0shot")
    parser.add_argument("--output-dir", type=str, default=str(OUTDIR / "results" / "pc_joint_scaling"))
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list (optional)")
    parser.add_argument("--hint-fractions", type=str, default=",".join([str(x) for x in DEFAULT_HINT_FRACTIONS]))
    parser.add_argument("--eval-hints-for-sweep", type=str, default=",".join([str(x) for x in DEFAULT_EVAL_HINTS_FOR_SWEEP]))
    parser.add_argument("--include-cross", action="store_true")
    parser.add_argument("--lower-asymptote", type=float, default=None)
    parser.add_argument("--hint-transform", type=str, default="identity")
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--num-holdout-models", type=int, default=0)

    # Alpha controls (optional)
    parser.add_argument("--alpha", type=str, default="fit", help="'fit' or comma-separated vector (len n_pcs)")
    parser.add_argument("--alpha-scales", type=str, default="", help="Optional comma-separated scales to multiply a base alpha direction")

    args = parser.parse_args()

    baseline_folder = Path(args.baseline_folder)

    if args.mode == "pca":
        from pca_helpers import compute_pc_scores, plot_component_weights_heatmap, plot_explained_variance

        _pivot, pca, _pc_scores_map = compute_pc_scores(baseline_folder=baseline_folder, n_components=int(args.n_components))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        plot_component_weights_heatmap(pca=pca, outfile=OUTDIR / "pca_component_weights.png")
        plot_explained_variance(pca=pca, outfile=OUTDIR / "pca_explained_variance.png")
        return

    # joint scaling
    from plot_helpers import run_joint_scaling_plots_pc

    results_base_folder = Path(args.results_base_folder)
    eci_file = Path(args.eci_file) if args.eci_file else None
    output_dir = Path(args.output_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()] if str(args.models).strip() else []
    hint_fractions = _parse_csv_floats(args.hint_fractions)
    eval_hints_for_sweep = _parse_csv_floats(args.eval_hints_for_sweep)

    alpha_fixed: np.ndarray | None
    if str(args.alpha).strip().lower() == "fit":
        alpha_fixed = None
    else:
        alpha_fixed = np.asarray(_parse_csv_floats(args.alpha), dtype=float)

    alpha_scales = _parse_csv_floats(args.alpha_scales) if str(args.alpha_scales).strip() else None

    # If no explicit models list, fall back to the same default list used in 20260202 experiments.
    if not models:
        models = [
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
        ]

    def run_one(out: Path, alpha: np.ndarray | None) -> dict[str, object]:
        return run_joint_scaling_plots_pc(
            base_folder=results_base_folder,
            baseline_folder=baseline_folder,
            eci_file=eci_file,
            eval_name=str(args.eval_name),
            solver=str(args.solver),
            condition=str(args.condition),
            label=f"{args.eval_name} ({args.solver}/{args.condition})",
            all_models=models,
            num_holdout_models=int(args.num_holdout_models),
            hint_fractions=hint_fractions,
            eval_hints_for_sweep=eval_hints_for_sweep,
            include_cross=bool(args.include_cross),
            lower_asymptote=(float(args.lower_asymptote) if args.lower_asymptote is not None else None),
            hint_transform=str(args.hint_transform),
            n_pcs=int(args.n_pcs),
            output_dir=out,
            alpha_fixed=alpha,
        )

    if alpha_scales:
        # Determine a base alpha direction (either from provided alpha, or by fitting once).
        base_metrics = run_one(output_dir / "alpha_base", alpha_fixed)
        base_alpha = np.asarray(base_metrics["alpha"], dtype=float)

        rows: list[dict[str, object]] = []
        for s in alpha_scales:
            scaled = float(s) * base_alpha
            m = run_one(output_dir / f"alpha_scale_{s:g}", scaled)
            rows.append({"alpha_scale": float(s), "rms_all": float(m["rms_all"]), "mse_all": float(m["mse_all"])})
        (output_dir / "alpha_scale_sweep.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return

    run_one(output_dir, alpha_fixed)


if __name__ == "__main__":
    main()

