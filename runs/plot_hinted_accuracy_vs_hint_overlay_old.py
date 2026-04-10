from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from runs.plot_hinted_accuracy_vs_hint import (
    EXPECTED_FRACTIONS,
    PLOTS_ROOT,
    RANDOM_SEED,
    _collect_stats_for_fraction,
    _discover_fraction_files,
    _discover_models,
    _fit_sigmoid_for_series,
    _is_complete_fraction,
    _load_external_mask_word_results,
    _safe_component,
    _sigmoid_curve,
)


OLD_TRUNC16K_PNG = Path(
    "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/20260321/plots/"
    "aime_2023_2024_solution_intext_masked_trunc16k_accuracy_vs_hint_by_model.png"
)
OLD_TRUNC16K_ROLLOUT_CSV = Path(
    "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/20260321/plots/"
    "aime_2023_2024_solution_intext_masked_trunc16k_rollouts.csv"
)
OLD_TRUNC16K_AGG_CSV = Path(
    "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/20260321/plots/"
    "aime_2023_2024_solution_intext_masked_trunc16k_accuracy_by_model_hint.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-off plot: current mask_word vs old 20260321 trunc16k AIME 2023/2024."
    )
    parser.add_argument("--benchmark", type=str, default="aime2025_2026")
    parser.add_argument("--hint-type", type=str, default="answer_not_revealed")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'.")
    parser.add_argument("--show-values", action="store_true")
    return parser.parse_args()


def _collect_current_mask_word_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    models_available = _discover_models(args.benchmark)
    external_mask_word = _load_external_mask_word_results()
    if args.model == "all":
        models_to_plot = sorted(set(models_available) | set((external_mask_word or {}).keys()))
    else:
        available_models = sorted(set(models_available) | set((external_mask_word or {}).keys()))
        if args.model not in available_models:
            raise ValueError(
                f"Requested model {args.model!r} not found. Available: {available_models}"
            )
        models_to_plot = [args.model]

    rng = np.random.default_rng(RANDOM_SEED)
    expected_fraction_set = {float(f"{value:.6f}") for value in EXPECTED_FRACTIONS}
    rows: list[dict[str, Any]] = []

    for model in models_to_plot:
        fraction_files = _discover_fraction_files(
            benchmark=args.benchmark,
            model=model,
            hint_type=args.hint_type,
            fractioner="mask_word",
        )
        if fraction_files:
            complete_fraction_files: list[tuple[float, Path]] = []
            for hint_fraction, path in fraction_files:
                is_complete, _ = _is_complete_fraction(path)
                if is_complete:
                    complete_fraction_files.append((float(hint_fraction), path))

            by_fraction = {float(f"{frac:.6f}"): path for frac, path in complete_fraction_files}
            if not (expected_fraction_set - set(by_fraction.keys())):
                for hint_fraction in EXPECTED_FRACTIONS:
                    path = by_fraction[float(f"{hint_fraction:.6f}")]
                    stats = _collect_stats_for_fraction(path=path, rng=rng)
                    if stats is None:
                        continue
                    rows.append(
                        {
                            "model": model,
                            "series": "mask_word_current",
                            "hint_fraction": float(hint_fraction),
                            **stats,
                            "path": str(path),
                        }
                    )

        if external_mask_word:
            model_payload = external_mask_word.get(model)
            if model_payload and not (expected_fraction_set - set(model_payload.keys())):
                for hint_fraction in EXPECTED_FRACTIONS:
                    stats = model_payload[float(hint_fraction)]
                    rows.append(
                        {
                            "model": model,
                            "series": "mask_word_results_with_ci",
                            "hint_fraction": float(hint_fraction),
                            "accuracy": float(stats["accuracy"]),
                            "ci_low": float(stats["ci_low"]),
                            "ci_high": float(stats["ci_high"]),
                            "n_samples": 0,
                            "n_rollouts": 0,
                            "path": "data/results_with_ci_mask_word.json",
                        }
                    )

    if not rows:
        raise ValueError("No usable current mask_word rows collected.")
    return sorted(rows, key=lambda r: (str(r["model"]), float(r["hint_fraction"])))


def _bootstrap_accuracy_from_sample_arrays(
    *,
    sample_to_scores: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_bootstrap: int = 5000,
) -> tuple[float, float, float]:
    sample_arrays = list(sample_to_scores.values())
    if not sample_arrays:
        raise ValueError("No sample arrays available for bootstrap.")

    point_accuracy = float(np.mean([arr.mean() for arr in sample_arrays]))
    boot_sums = np.zeros(n_bootstrap, dtype=float)
    for arr in sample_arrays:
        draw_idx = rng.integers(low=0, high=arr.size, size=n_bootstrap)
        boot_sums += arr[draw_idx]
    boot_means = boot_sums / float(len(sample_arrays))
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    return point_accuracy, float(ci_low), float(ci_high)


def _load_old_trunc16k_rows(*, model_filter: str) -> list[dict[str, Any]]:
    if not OLD_TRUNC16K_ROLLOUT_CSV.exists():
        raise FileNotFoundError(f"Missing old rollout CSV: {OLD_TRUNC16K_ROLLOUT_CSV}")
    if not OLD_TRUNC16K_AGG_CSV.exists():
        raise FileNotFoundError(f"Missing old aggregate CSV: {OLD_TRUNC16K_AGG_CSV}")

    agg_means: dict[tuple[str, float], float] = {}
    with open(OLD_TRUNC16K_AGG_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            try:
                hint_fraction = float(row["hint"])
                new_accuracy = float(row["new_accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            agg_means[(model, float(hint_fraction))] = float(new_accuracy)

    sample_to_scores: dict[tuple[str, float], dict[str, list[float]]] = {}
    rollout_counts: dict[tuple[str, float], int] = {}

    with open(OLD_TRUNC16K_ROLLOUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            if model_filter != "all" and model != model_filter:
                continue
            try:
                hint_fraction = float(row["hint"])
            except (KeyError, TypeError, ValueError):
                continue
            if hint_fraction not in EXPECTED_FRACTIONS:
                continue

            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue

            raw_value = str(row.get("new_is_correct", "")).strip().lower()
            if raw_value not in {"0", "1", "0.0", "1.0", "false", "true"}:
                continue
            is_correct = raw_value in {"1", "1.0", "true"}

            key = (model, float(hint_fraction))
            sample_to_scores.setdefault(key, {}).setdefault(sample_id, []).append(
                1.0 if is_correct else 0.0
            )
            rollout_counts[key] = rollout_counts.get(key, 0) + 1

    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for (model, hint_fraction), sample_scores in sorted(sample_to_scores.items()):
        arrays = {
            sample_id: np.asarray(scores, dtype=float)
            for sample_id, scores in sample_scores.items()
            if scores
        }
        if not arrays:
            continue
        point_accuracy, ci_low, ci_high = _bootstrap_accuracy_from_sample_arrays(
            sample_to_scores=arrays,
            rng=rng,
        )

        agg_mean = agg_means.get((model, hint_fraction))
        if agg_mean is not None:
            point_accuracy = float(agg_mean)

        rows.append(
            {
                "model": model,
                "series": "old_trunc16k",
                "hint_fraction": float(hint_fraction),
                "accuracy": float(point_accuracy),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_samples": int(len(arrays)),
                "n_rollouts": int(rollout_counts[(model, hint_fraction)]),
                "path": str(OLD_TRUNC16K_ROLLOUT_CSV),
                "source_png": str(OLD_TRUNC16K_PNG),
                "source_agg_csv": str(OLD_TRUNC16K_AGG_CSV),
            }
        )

    if not rows:
        raise ValueError("No usable old trunc16k rows collected.")
    return rows


def _series_fit_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    fit_map: dict[tuple[str, str], dict[str, float]] = {}
    series_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["series"]))
        series_map.setdefault(key, []).append(row)
    for key, series_rows in sorted(series_map.items()):
        fit = _fit_sigmoid_for_series(
            [
                {
                    "hint_fraction": row["hint_fraction"],
                    "accuracy": row["accuracy"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                }
                for row in series_rows
            ]
        )
        if fit is not None:
            fit_map[key] = fit
    return fit_map


def _plot(
    current_rows: list[dict[str, Any]],
    *,
    current_fit_map: dict[tuple[str, str], dict[str, float]],
    old_rows: list[dict[str, Any]],
    old_fit_map: dict[tuple[str, str], dict[str, float]],
    output_png: Path,
    show_values: bool,
    title: str,
) -> None:
    models = sorted({row["model"] for row in current_rows} | {row["model"] for row in old_rows})
    n_models = len(models)

    if n_models == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axes = [ax]
    else:
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
        axes = axes_obj.flatten() if hasattr(axes_obj, "flatten") else [axes_obj]

    current_color = "#1f77b4"
    results_with_ci_color = "#111111"
    old_color = "#b22222"

    for idx, model in enumerate(models):
        ax = axes[idx]

        for rows, color, marker, linestyle, label, fit_map, value_offset in [
            (
                [row for row in current_rows if row["series"] == "mask_word_current"],
                current_color,
                "o",
                "-",
                "mask_word_current",
                current_fit_map,
                (4, 4),
            ),
            (
                [row for row in current_rows if row["series"] == "mask_word_results_with_ci"],
                results_with_ci_color,
                "s",
                "--",
                "mask_word_results_with_ci",
                current_fit_map,
                (4, -9),
            ),
            (old_rows, old_color, "^", ":", "aime2023_2024_trunc16k", old_fit_map, (4, -9)),
        ]:
            series_rows = sorted(
                [row for row in rows if row["model"] == model],
                key=lambda row: float(row["hint_fraction"]),
            )
            if not series_rows:
                continue

            x = np.asarray([float(row["hint_fraction"]) for row in series_rows], dtype=float)
            y = np.asarray([float(row["accuracy"]) for row in series_rows], dtype=float)
            low = np.asarray([float(row["ci_low"]) for row in series_rows], dtype=float)
            high = np.asarray([float(row["ci_high"]) for row in series_rows], dtype=float)
            yerr = np.vstack([y - low, high - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=marker,
                alpha=0.92,
                markersize=4.0,
                capsize=2.0,
                elinewidth=1.0,
                capthick=1.0,
                color=color,
                linestyle="none",
                linewidth=1.0,
                label=label,
            )

            if show_values:
                for x_i, y_i in zip(x, y):
                    ax.annotate(
                        f"{y_i:.2f}",
                        (float(x_i), float(y_i)),
                        xytext=value_offset,
                        textcoords="offset points",
                        fontsize=6,
                        color=color,
                        alpha=0.9,
                    )

            fit = fit_map.get((str(model), str(series_rows[0]["series"])))
            if fit is not None:
                x_fit = np.linspace(0.0, 1.0, 200, dtype=float)
                y_fit = _sigmoid_curve(
                    x_fit,
                    float(fit["sigmoid_lower"]),
                    float(fit["sigmoid_slope"]),
                    float(fit["sigmoid_bias"]),
                )
                ax.plot(x_fit, y_fit, linestyle, color=color, linewidth=1.4, alpha=0.9)

        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Hint Fraction")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    current_rows = _collect_current_mask_word_rows(args)
    old_rows = _load_old_trunc16k_rows(model_filter=args.model)

    current_fit_map = _series_fit_map(current_rows)
    old_fit_map = _series_fit_map(old_rows)

    stem = (
        f"{_safe_component(args.benchmark)}__{_safe_component(args.hint_type)}__"
        f"mask_word_vs_aime2023_2024_trunc16k__{_safe_component(args.model)}"
    )
    png_path = PLOTS_ROOT / f"{stem}__bootstrap.png"
    summary_json_path = PLOTS_ROOT / f"{stem}__summary_percent.json"

    summary_payload = {
        "benchmark": args.benchmark,
        "hint_type": args.hint_type,
        "comparison": "current_mask_word_vs_old_aime2023_2024_trunc16k",
        "old_source_png": str(OLD_TRUNC16K_PNG),
        "old_source_rollout_csv": str(OLD_TRUNC16K_ROLLOUT_CSV),
        "old_source_agg_csv": str(OLD_TRUNC16K_AGG_CSV),
        "current_rows": current_rows,
        "old_rows": old_rows,
    }
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    _plot(
        current_rows,
        current_fit_map=current_fit_map,
        old_rows=old_rows,
        old_fit_map=old_fit_map,
        output_png=png_path,
        show_values=args.show_values,
        title=(
            f"Mask Word vs AIME 2023/2024 Trunc16k\n"
            f"benchmark={args.benchmark} hint_type={args.hint_type}"
        ),
    )

    print(f"[plot_hinted_accuracy_vs_hint_overlay_old] old_source_png={OLD_TRUNC16K_PNG}")
    print(f"[plot_hinted_accuracy_vs_hint_overlay_old] old_source_rollout_csv={OLD_TRUNC16K_ROLLOUT_CSV}")
    print(f"[plot_hinted_accuracy_vs_hint_overlay_old] wrote_summary_json={summary_json_path}")
    print(f"[plot_hinted_accuracy_vs_hint_overlay_old] wrote_plot={png_path}")


if __name__ == "__main__":
    # python -m runs.plot_hinted_accuracy_vs_hint_overlay_old
    main()
