from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.hinted_accuracy import (
    discover_fractioners as canonical_discover_fractioners,
    discover_models_for_benchmark as canonical_discover_models_for_benchmark,
    discover_models_for_combo,
    load_external_results_with_ci_for_fractioner,
    load_results_with_ci_for_combo,
)


DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/accuracy_vs_hint")

EXPECTED_FRACTIONS = [i / 10 for i in range(11)]


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hinted inference accuracy vs hint fraction."
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'.")
    parser.add_argument(
        "--fractioner",
        type=str,
        default=None,
        help="Optional specific fractioner to plot. If omitted, use all complete fractioners.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Annotate plotted points with their mean accuracy values.",
    )
    return parser.parse_args()


def _discover_models(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
) -> list[str]:
    if fractioner is not None:
        models = discover_models_for_combo(
            benchmark=benchmark,
            hint_type=hint_type,
            fractioner=fractioner,
        )
        if models:
            return models

    models = set(canonical_discover_models_for_benchmark(benchmark))

    fractioners = [fractioner] if fractioner is not None else []
    if not fractioners:
        for local_model in sorted(models):
            fractioners.extend(
                canonical_discover_fractioners(
                    benchmark=benchmark,
                    model=local_model,
                    hint_type=hint_type,
                )
            )
    for current_fractioner in sorted(set(fractioners)):
        models.update(
            discover_models_for_combo(
                benchmark=benchmark,
                hint_type=hint_type,
                fractioner=current_fractioner,
            )
        )

    if not models:
        benchmark_dir = DATA_ROOT / "hinted_inference" / _safe_component(benchmark)
        raise FileNotFoundError(f"Missing benchmark directory: {benchmark_dir}")
    return sorted(models)


def _sigmoid_curve(x: np.ndarray, lower: float, slope: float, bias: float) -> np.ndarray:
    return lower + (1.0 - lower) * (1.0 / (1.0 + np.exp(-(slope * x + bias))))


def _fit_sigmoid_for_series(series_rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if len(series_rows) < 4:
        return None

    rows_sorted = sorted(series_rows, key=lambda row: float(row["hint_fraction"]))
    x = np.asarray([float(row["hint_fraction"]) for row in rows_sorted], dtype=float)
    y = np.asarray([float(row["accuracy"]) for row in rows_sorted], dtype=float)
    if np.allclose(y, y[0]):
        return None

    # Approximate per-point standard error from 95% CI half-width.
    ci_half_width = np.asarray(
        [
            max(
                1e-6,
                0.5 * (float(row["ci_high"]) - float(row["ci_low"])),
            )
            for row in rows_sorted
        ],
        dtype=float,
    )
    sigma = np.maximum(ci_half_width / 1.96, 1e-4)

    lower0 = float(np.clip(np.min(y) - 0.02, 0.0, 0.95))
    y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
    mid_idx = int(np.argmin(np.abs(y - y_mid)))
    x_mid = float(x[mid_idx])
    slope0 = 8.0
    bias0 = -slope0 * x_mid

    try:
        params, _ = curve_fit(
            _sigmoid_curve,
            x,
            y,
            p0=[lower0, slope0, bias0],
            sigma=sigma,
            absolute_sigma=True,
            bounds=([0.0, 1e-6, -50.0], [0.99, 100.0, 50.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    lower, slope, bias = [float(v) for v in params]
    midpoint = float(-bias / slope) if slope > 0 else float("nan")
    y_hat = _sigmoid_curve(x, lower, slope, bias)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return {
        "sigmoid_lower": lower,
        "sigmoid_slope": slope,
        "sigmoid_bias": bias,
        "sigmoid_midpoint": midpoint,
        "sigmoid_rmse": rmse,
    }


def _plot(
    results: list[dict[str, Any]],
    *,
    fit_map: dict[tuple[str, str], dict[str, float]],
    external_results: list[dict[str, Any]],
    external_fit_map: dict[tuple[str, str], dict[str, float]],
    output_png,
    show_values: bool,
    title: str,
) -> None:
    models = sorted({row["model"] for row in results} | {row["model"] for row in external_results})
    n_models = len(models)

    if n_models == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axes = [ax]
    else:
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
        axes = axes_obj.flatten() if hasattr(axes_obj, "flatten") else [axes_obj]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_rows = [row for row in results if row["model"] == model]
        external_model_rows = [row for row in external_results if row["model"] == model]
        fractioners = sorted({str(row["fractioner"]) for row in model_rows})
        cmap = plt.cm.tab10
        for j, fractioner in enumerate(fractioners):
            color = cmap(j / max(len(fractioners) - 1, 1))
            series_rows = sorted(
                [row for row in model_rows if row["fractioner"] == fractioner],
                key=lambda row: float(row["hint_fraction"]),
            )
            x = np.asarray([float(row["hint_fraction"]) for row in series_rows], dtype=float)
            y = np.asarray([float(row["accuracy"]) for row in series_rows], dtype=float)
            low = np.asarray([float(row["ci_low"]) for row in series_rows], dtype=float)
            high = np.asarray([float(row["ci_high"]) for row in series_rows], dtype=float)
            yerr = np.vstack([y - low, high - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                alpha=0.9,
                markersize=3.8,
                capsize=2.0,
                elinewidth=1.0,
                capthick=1.0,
                color=color,
                label=fractioner,
            )
            if show_values:
                for x_i, y_i in zip(x, y):
                    ax.annotate(
                        f"{y_i:.2f}",
                        (float(x_i), float(y_i)),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6,
                        color=color,
                        alpha=0.9,
                    )

            fit_key = (str(model), str(fractioner))
            fit = fit_map.get(fit_key)
            if fit is not None:
                x_fit = np.linspace(0.0, 1.0, 200, dtype=float)
                y_fit = _sigmoid_curve(
                    x_fit,
                    float(fit["sigmoid_lower"]),
                    float(fit["sigmoid_slope"]),
                    float(fit["sigmoid_bias"]),
                )
                ax.plot(x_fit, y_fit, "-", color=color, linewidth=1.25, alpha=0.85)

        external_fractioners = sorted({str(row["fractioner"]) for row in external_model_rows})
        for fractioner in external_fractioners:
            series_rows = sorted(
                [row for row in external_model_rows if row["fractioner"] == fractioner],
                key=lambda row: float(row["hint_fraction"]),
            )
            x = np.asarray([float(row["hint_fraction"]) for row in series_rows], dtype=float)
            y = np.asarray([float(row["accuracy"]) for row in series_rows], dtype=float)
            low = np.asarray([float(row["ci_low"]) for row in series_rows], dtype=float)
            high = np.asarray([float(row["ci_high"]) for row in series_rows], dtype=float)
            yerr = np.vstack([y - low, high - y])

            overlay_color = "#111111"
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="s",
                alpha=0.95,
                markersize=3.8,
                capsize=2.0,
                elinewidth=1.0,
                capthick=1.0,
                color=overlay_color,
                linestyle="none",
                linewidth=1.0,
                label=f"{fractioner}_luke",
            )
            if show_values:
                for x_i, y_i in zip(x, y):
                    ax.annotate(
                        f"{y_i:.2f}",
                        (float(x_i), float(y_i)),
                        xytext=(4, -9),
                        textcoords="offset points",
                        fontsize=6,
                        color=overlay_color,
                        alpha=0.9,
                    )

            fit_key = (str(model), str(fractioner))
            fit = external_fit_map.get(fit_key)
            if fit is not None:
                x_fit = np.linspace(0.0, 1.0, 200, dtype=float)
                y_fit = _sigmoid_curve(
                    x_fit,
                    float(fit["sigmoid_lower"]),
                    float(fit["sigmoid_slope"]),
                    float(fit["sigmoid_bias"]),
                )
                ax.plot(x_fit, y_fit, "--", color=overlay_color, linewidth=1.25, alpha=0.9)
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
    models_available = _discover_models(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    if not models_available:
        raise ValueError(f"No models found for benchmark={args.benchmark!r}.")

    if args.model == "all":
        models_to_plot = sorted(set(models_available))
    else:
        if args.model not in models_available:
            raise ValueError(
                f"Requested model {args.model!r} not found. "
                f"Available: {models_available}"
            )
        models_to_plot = [args.model]

    rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    combo_cache: dict[str, dict[str, dict[float, dict[str, float]]]] = {}
    external_cache: dict[str, dict[str, dict[float, dict[str, float]]]] = {}

    for model in models_to_plot:
        if args.fractioner is not None:
            fractioners = [args.fractioner]
        else:
            fractioners = canonical_discover_fractioners(
                benchmark=args.benchmark,
                model=model,
                hint_type=args.hint_type,
            )
        if not fractioners:
            print(
                f"[plot_hinted_accuracy_vs_hint][WARN] no fractioners for "
                f"model={model} hint_type={args.hint_type}"
            )
            continue

        for fractioner in fractioners:
            combo_payload = combo_cache.get(fractioner)
            if combo_payload is None:
                combo_payload = load_results_with_ci_for_combo(
                    benchmark=args.benchmark,
                    hint_type=args.hint_type,
                    fractioner=fractioner,
                )
                combo_cache[fractioner] = combo_payload

            external_payload = external_cache.get(fractioner)
            if external_payload is None:
                external_payload = load_external_results_with_ci_for_fractioner(fractioner)
                external_cache[fractioner] = external_payload

            model_payload = combo_payload.get(model)
            if not model_payload:
                print(
                    f"[plot_hinted_accuracy_vs_hint][WARN] no combo results for "
                    f"model={model} fractioner={fractioner}"
                )
                continue

            fraction_rows: list[dict[str, Any]] = []
            missing = sorted(
                {float(f"{value:.6f}") for value in EXPECTED_FRACTIONS} - set(model_payload.keys())
            )
            if missing:
                print(
                    f"[plot_hinted_accuracy_vs_hint][WARN] missing combo fractions "
                    f"model={model} fractioner={fractioner} missing_fractions={missing}"
                )
                continue

            for hint_fraction in EXPECTED_FRACTIONS:
                stats = model_payload[float(hint_fraction)]
                fraction_rows.append(
                    {
                        "model": model,
                        "fractioner": fractioner,
                        "hint_fraction": float(hint_fraction),
                        "accuracy": float(stats["accuracy"]),
                        "ci_low": float(stats["ci_low"]),
                        "ci_high": float(stats["ci_high"]),
                        "n_samples": 0,
                        "n_rollouts": 0,
                        "rows_total": 0,
                        "rows_with_known_label": 0,
                        "rows_without_known_label": 0,
                        "path": f"combo::{args.benchmark}::{args.hint_type}::{fractioner}",
                    }
                )

            rows.extend(fraction_rows)
            print(
                f"[plot_hinted_accuracy_vs_hint] included model={model} "
                f"fractioner={fractioner} n_points={len(fraction_rows)}"
            )
            means_text = ", ".join(
                f"{float(row['hint_fraction']):.1f}:{float(row['accuracy']):.4f}"
                for row in fraction_rows
            )
            print(
                f"[plot_hinted_accuracy_vs_hint] means model={model} "
                f"fractioner={fractioner} {means_text}"
            )

            external_model_payload = external_payload.get(model)
            if external_model_payload:
                missing_external = sorted(
                    {float(f"{value:.6f}") for value in EXPECTED_FRACTIONS}
                    - set(external_model_payload.keys())
                )
                if missing_external:
                    print(
                        f"[plot_hinted_accuracy_vs_hint][WARN] missing external fractions "
                        f"model={model} fractioner={fractioner} missing_fractions={missing_external}"
                    )
                else:
                    current_external_rows: list[dict[str, Any]] = []
                    for hint_fraction in EXPECTED_FRACTIONS:
                        stats = external_model_payload[float(hint_fraction)]
                        current_external_rows.append(
                            {
                                "model": model,
                                "fractioner": fractioner,
                                "hint_fraction": float(hint_fraction),
                                "accuracy": float(stats["accuracy"]),
                                "ci_low": float(stats["ci_low"]),
                                "ci_high": float(stats["ci_high"]),
                            }
                        )
                    external_rows.extend(current_external_rows)
                    external_means_text = ", ".join(
                        f"{float(row['hint_fraction']):.1f}:{float(row['accuracy']):.4f}"
                        for row in current_external_rows
                    )
                    print(
                        f"[plot_hinted_accuracy_vs_hint] external means model={model} "
                        f"fractioner={fractioner} {external_means_text}"
                    )

    if not rows:
        raise ValueError("No usable rows collected. Check benchmark/model/hint_type/fractioner.")

    rows_sorted = sorted(
        rows,
        key=lambda r: (str(r["model"]), str(r["fractioner"]), float(r["hint_fraction"])),
    )
    series_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_sorted:
        key = (str(row["model"]), str(row["fractioner"]))
        series_map.setdefault(key, []).append(row)

    fit_map: dict[tuple[str, str], dict[str, float]] = {}
    for key, series_rows in sorted(series_map.items()):
        model, fractioner = key
        fit = _fit_sigmoid_for_series(series_rows)
        if fit is None:
            print(
                f"[plot_hinted_accuracy_vs_hint][WARN] sigmoid fit failed/skipped "
                f"model={model} fractioner={fractioner}"
            )
            continue
        fit_map[key] = fit
        print(
            f"[plot_hinted_accuracy_vs_hint] fit model={model} fractioner={fractioner} "
            f"midpoint={float(fit['sigmoid_midpoint']):.4f} "
            f"rmse={float(fit['sigmoid_rmse']):.4f}"
        )

    external_fit_map: dict[tuple[str, str], dict[str, float]] = {}
    external_series_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(
        external_rows,
        key=lambda r: (str(r["model"]), str(r["fractioner"]), float(r["hint_fraction"])),
    ):
        key = (str(row["model"]), str(row["fractioner"]))
        external_series_map.setdefault(key, []).append(row)

    for key, series_rows in sorted(external_series_map.items()):
        fit = _fit_sigmoid_for_series(series_rows)
        if fit is not None:
            external_fit_map[key] = fit

    stem = (
        f"{_safe_component(args.benchmark)}__{_safe_component(args.hint_type)}__"
        f"{_safe_component(args.fractioner) if args.fractioner is not None else 'all_complete_fractioners'}__"
        f"{_safe_component(args.model)}"
    )
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    means_json_path = PLOTS_ROOT / f"{stem}__means_percent.json"
    summary_json_path = PLOTS_ROOT / f"{stem}__summary_percent.json"
    png_path = PLOTS_ROOT / f"{stem}__bootstrap.png"

    means_payload: dict[str, dict[str, dict[str, float]]] = {}
    summary_payload: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (model, fractioner), series_rows in sorted(series_map.items()):
        means_payload.setdefault(model, {})
        summary_payload.setdefault(model, {})
        means_payload[model][fractioner] = {
            f"{float(row['hint_fraction']):.1f}": round(100.0 * float(row["accuracy"]), 1)
            for row in sorted(series_rows, key=lambda row: float(row["hint_fraction"]))
        }
        summary_payload[model][fractioner] = {
            f"{float(row['hint_fraction']):.1f}": {
                "mean": round(100.0 * float(row["accuracy"]), 1),
                "ci_low": round(100.0 * float(row["ci_low"]), 1),
                "ci_high": round(100.0 * float(row["ci_high"]), 1),
            }
            for row in sorted(series_rows, key=lambda row: float(row["hint_fraction"]))
        }

    with open(means_json_path, "w", encoding="utf-8") as f:
        json.dump(means_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    _plot(
        rows_sorted,
        fit_map=fit_map,
        external_results=external_rows,
        external_fit_map=external_fit_map,
        output_png=png_path,
        show_values=args.show_values,
        title=(
            f"Hinted Accuracy vs Hint Fraction\n"
            f"benchmark={args.benchmark} hint_type={args.hint_type} "
            f"({args.fractioner if args.fractioner is not None else 'all complete fractioners'})"
        ),
    )

    print(f"[plot_hinted_accuracy_vs_hint] wrote_means_json={means_json_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_summary_json={summary_json_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_plot={png_path}")


if __name__ == "__main__":
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner truncate_word
    main()
