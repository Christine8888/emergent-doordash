from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.hinted_accuracy import (
    collect_complete_fraction_stats,
    discover_fractioners as canonical_discover_fractioners,
    discover_models_for_benchmark as canonical_discover_models_for_benchmark,
    load_luke_results_with_ci_for_combo,
)
from src.model_config import (
    filter_models_for_fractioner,
    is_model_excluded_for_fractioner,
    models_excluded_from_selection,
)


DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/accuracy_vs_hint")

EXPECTED_FRACTIONS = [i / 10 for i in range(11)]
AIME_SPLIT_BENCHMARK = "aime2025_2026"
AIME_SPLIT_HINT_TYPE = "answer_not_revealed"
ProblemIdPredicate = Callable[[str], bool]


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _extract_problem_index(problem_id: str) -> int | None:
    match = re.search(r"_(\d+)$", problem_id.strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _problem_ids_for_aime_2025(problem_id: str) -> bool:
    idx = _extract_problem_index(problem_id)
    return idx is not None and 1 <= idx <= 30


def _problem_ids_for_aime_2026(problem_id: str) -> bool:
    idx = _extract_problem_index(problem_id)
    return idx is not None and 31 <= idx <= 60


def _split_plot_specs(
    *,
    benchmark: str,
    hint_type: str,
) -> list[tuple[str, ProblemIdPredicate]]:
    if benchmark != AIME_SPLIT_BENCHMARK or hint_type != AIME_SPLIT_HINT_TYPE:
        return []
    return [
        ("aime2025", _problem_ids_for_aime_2025),
        ("aime2026", _problem_ids_for_aime_2026),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hinted inference accuracy vs hint fraction."
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["all"],
        help="One-or-more model names, or 'all'.",
    )
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
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Render a single-source plot: prefer local/basic rows, and only use luke rows "
            "for points missing from local rows."
        ),
    )
    parser.add_argument(
        "--include-split-plots",
        action="store_true",
        help=(
            "Also render split plots for benchmark subsets (e.g., aime2025/aime2026 when "
            "supported). Disabled by default."
        ),
    )
    return parser.parse_args()


def _discover_models(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
) -> list[str]:
    models = set(canonical_discover_models_for_benchmark(benchmark, fractioner=fractioner))

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
    local_models = set()
    for local_model in sorted(models):
        local_fractioners = canonical_discover_fractioners(
            benchmark=benchmark,
            model=local_model,
            hint_type=hint_type,
        )
        if any(current_fractioner in local_fractioners for current_fractioner in set(fractioners)):
            local_models.add(local_model)

    luke_models = set()
    for current_fractioner in sorted(set(fractioners)):
        luke_models.update(
            load_luke_results_with_ci_for_combo(
                benchmark=benchmark,
                hint_type=hint_type,
                fractioner=current_fractioner,
            ).keys()
        )

    models = set(filter_models_for_fractioner(sorted(local_models | luke_models), fractioner))

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


def _fraction_column_name(hint_fraction: float) -> str:
    return f"{hint_fraction:.1f}"


def _merge_rows_for_accuracy_table(
    local_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, float], dict[str, Any]] = {}

    for row in external_rows:
        key = (str(row["model"]), float(row["hint_fraction"]))
        merged_by_key[key] = row

    for row in local_rows:
        key = (str(row["model"]), float(row["hint_fraction"]))
        merged_by_key[key] = row

    return sorted(
        merged_by_key.values(),
        key=lambda row: (str(row["model"]), float(row["hint_fraction"])),
    )


def _merge_rows_for_clean_plot(
    local_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}

    for row in external_rows:
        key = (str(row["model"]), str(row["fractioner"]), float(row["hint_fraction"]))
        merged_by_key[key] = row

    for row in local_rows:
        key = (str(row["model"]), str(row["fractioner"]), float(row["hint_fraction"]))
        merged_by_key[key] = row

    return sorted(
        merged_by_key.values(),
        key=lambda row: (str(row["model"]), str(row["fractioner"]), float(row["hint_fraction"])),
    )


def _build_accuracy_table_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, str]]]:
    fractions = sorted({float(row["hint_fraction"]) for row in rows})
    fieldnames = ["model", *[_fraction_column_name(fraction) for fraction in fractions]]

    series_map: dict[str, dict[float, float]] = {}
    for row in rows:
        model = str(row["model"])
        hint_fraction = float(row["hint_fraction"])
        fraction_map = series_map.setdefault(model, {})
        if hint_fraction in fraction_map:
            raise ValueError(
                "Accuracy table export found duplicate hint fractions for the same model "
                "after merging local and luke_results rows."
            )
        fraction_map[hint_fraction] = float(row["accuracy"])

    table_rows: list[dict[str, str]] = []
    for model, fraction_map in sorted(series_map.items()):
        table_row: dict[str, str] = {
            "model": model,
        }
        for fraction in fractions:
            value = fraction_map.get(fraction)
            table_row[_fraction_column_name(fraction)] = "" if value is None else f"{value:.4f}"
        table_rows.append(table_row)
    return fieldnames, table_rows


def _print_accuracy_table(fieldnames: list[str], table_rows: list[dict[str, str]]) -> None:
    widths = {
        fieldname: max(
            len(fieldname),
            max((len(row.get(fieldname, "")) for row in table_rows), default=0),
        )
        for fieldname in fieldnames
    }

    print("[plot_hinted_accuracy_vs_hint] accuracy table")
    print("  ".join(fieldname.ljust(widths[fieldname]) for fieldname in fieldnames))
    print("  ".join("-" * widths[fieldname] for fieldname in fieldnames))
    for row in table_rows:
        print("  ".join(row.get(fieldname, "").ljust(widths[fieldname]) for fieldname in fieldnames))


def _write_accuracy_table_csv(
    *,
    fieldnames: list[str],
    table_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


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
    fractioners_all = sorted(
        {str(row["fractioner"]) for row in results} | {str(row["fractioner"]) for row in external_results}
    )
    if fractioners_all:
        cmap = plt.cm.get_cmap("tab20", len(fractioners_all))
        fractioner_color_map = {
            fractioner: cmap(index) for index, fractioner in enumerate(fractioners_all)
        }
    else:
        fractioner_color_map: dict[str, Any] = {}

    if n_models == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axes = [ax]
    elif n_models == 4:
        n_cols = 2
        n_rows = 2
        fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
        axes = axes_obj.flatten() if hasattr(axes_obj, "flatten") else [axes_obj]
    else:
        n_cols = 5
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
        axes = axes_obj.flatten() if hasattr(axes_obj, "flatten") else [axes_obj]

    all_y_values = [
        float(row["ci_low"])
        for row in results
        if isinstance(row, dict) and "ci_low" in row
    ] + [
        float(row["ci_high"])
        for row in results
        if isinstance(row, dict) and "ci_high" in row
    ] + [
        float(row["ci_low"])
        for row in external_results
        if isinstance(row, dict) and "ci_low" in row
    ] + [
        float(row["ci_high"])
        for row in external_results
        if isinstance(row, dict) and "ci_high" in row
    ]
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
    else:
        y_min, y_max = 0.0, 1.0
    y_padding = 0.03
    y_min_plot = max(0.0, y_min - y_padding)
    y_max_plot = min(1.0, y_max + y_padding)
    if y_max_plot - y_min_plot < 0.08:
        y_mid = 0.5 * (y_min_plot + y_max_plot)
        y_min_plot = max(0.0, y_mid - 0.04)
        y_max_plot = min(1.0, y_mid + 0.04)

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_rows = [row for row in results if row["model"] == model]
        external_model_rows = [row for row in external_results if row["model"] == model]
        fractioners = sorted({str(row["fractioner"]) for row in model_rows})
        for fractioner in fractioners:
            color = fractioner_color_map.get(fractioner, "#1f77b4")
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

            overlay_color = fractioner_color_map.get(fractioner, "#111111")
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
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.legend(fontsize=7)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_fit_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    fit_map: dict[tuple[str, str], dict[str, float]] = {}
    series_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(
        rows,
        key=lambda r: (str(r["model"]), str(r["fractioner"]), float(r["hint_fraction"])),
    ):
        key = (str(row["model"]), str(row["fractioner"]))
        series_map.setdefault(key, []).append(row)
    for key, series_rows in sorted(series_map.items()):
        fit = _fit_sigmoid_for_series(series_rows)
        if fit is not None:
            fit_map[key] = fit
    return fit_map


def _collect_rows_for_models(
    *,
    benchmark: str,
    hint_type: str,
    models_to_plot: list[str],
    fractioner: str | None,
    problem_id_predicate: ProblemIdPredicate | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    external_cache: dict[str, dict[str, dict[float, dict[str, float]]]] = {}

    for model in models_to_plot:
        if fractioner is not None:
            fractioners = [fractioner]
        else:
            fractioners = canonical_discover_fractioners(
                benchmark=benchmark,
                model=model,
                hint_type=hint_type,
            )
        if not fractioners:
            print(
                f"[plot_hinted_accuracy_vs_hint][WARN] no fractioners for "
                f"model={model} hint_type={hint_type}"
            )
            continue

        for current_fractioner in fractioners:
            if is_model_excluded_for_fractioner(model, current_fractioner):
                print(
                    f"[plot_hinted_accuracy_vs_hint] excluding model={model} "
                    f"for fractioner={current_fractioner} via model_config",
                    flush=True,
                )
                continue
            local_fraction_rows, local_warnings = collect_complete_fraction_stats(
                benchmark=benchmark,
                model=model,
                hint_type=hint_type,
                fractioner=current_fractioner,
                data_root=DATA_ROOT,
                problem_id_predicate=problem_id_predicate,
            )
            if not local_fraction_rows:
                if local_warnings:
                    print(
                        f"[plot_hinted_accuracy_vs_hint][WARN] no local combo results for "
                        f"model={model} fractioner={current_fractioner} warnings={local_warnings}"
                    )
            else:
                rows.extend(local_fraction_rows)
                print(
                    f"[plot_hinted_accuracy_vs_hint] included model={model} "
                    f"fractioner={current_fractioner} n_points={len(local_fraction_rows)}"
                )
                means_text = ", ".join(
                    f"{float(row['hint_fraction']):.1f}:{float(row['accuracy']):.4f}"
                    for row in local_fraction_rows
                )
                print(
                    f"[plot_hinted_accuracy_vs_hint] means model={model} "
                    f"fractioner={current_fractioner} {means_text}"
                )

            external_payload = external_cache.get(current_fractioner)
            if external_payload is None:
                external_payload = load_luke_results_with_ci_for_combo(
                    benchmark=benchmark,
                    hint_type=hint_type,
                    fractioner=current_fractioner,
                    problem_id_predicate=problem_id_predicate,
                )
                external_cache[current_fractioner] = external_payload

            external_model_payload = external_payload.get(model)
            if external_model_payload:
                missing_external = sorted(
                    {float(f"{value:.6f}") for value in EXPECTED_FRACTIONS}
                    - set(external_model_payload.keys())
                )
                if missing_external:
                    print(
                        f"[plot_hinted_accuracy_vs_hint][WARN] missing external fractions "
                        f"model={model} fractioner={current_fractioner} missing_fractions={missing_external}"
                    )
                else:
                    current_external_rows: list[dict[str, Any]] = []
                    for hint_fraction in EXPECTED_FRACTIONS:
                        stats = external_model_payload[float(hint_fraction)]
                        current_external_rows.append(
                            {
                                "model": model,
                                "fractioner": current_fractioner,
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
                        f"fractioner={current_fractioner} {external_means_text}"
                    )
    return rows, external_rows


def main() -> None:
    args = _parse_args()
    models_available = _discover_models(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    if not models_available:
        raise ValueError(f"No models found for benchmark={args.benchmark!r}.")

    requested_models = list(args.model)
    if requested_models == ["all"]:
        models_to_plot = sorted(set(models_available))
    else:
        if "all" in requested_models:
            raise ValueError(
                "When passing specific models, do not include 'all'. "
                f"Requested: {requested_models}"
            )
        excluded_requested_models = models_excluded_from_selection(
            requested_models,
            args.fractioner,
        )
        if excluded_requested_models:
            raise ValueError(
                f"Requested model(s) excluded for fractioner={args.fractioner!r}: "
                f"{excluded_requested_models}"
            )
        missing_models = sorted(set(requested_models) - set(models_available))
        if missing_models:
            raise ValueError(
                f"Requested model(s) not found: {missing_models}. "
                f"Available: {models_available}"
            )
        models_to_plot = sorted(set(requested_models))
    if not models_to_plot:
        raise ValueError(
            "All selected models were excluded by model_config. "
            "Update FRACTIONER_EXCLUDED_MODELS or pass different --model values."
        )

    rows, external_rows = _collect_rows_for_models(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        models_to_plot=models_to_plot,
        fractioner=args.fractioner,
    )

    if not rows and not external_rows:
        raise ValueError("No usable rows collected. Check benchmark/model/hint_type/fractioner.")

    rows_sorted = sorted(
        rows,
        key=lambda r: (str(r["model"]), str(r["fractioner"]), float(r["hint_fraction"])),
    )
    series_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_sorted:
        key = (str(row["model"]), str(row["fractioner"]))
        series_map.setdefault(key, []).append(row)
    fit_map = _build_fit_map(rows_sorted)
    external_fit_map = _build_fit_map(external_rows)

    model_component = (
        "all" if requested_models == ["all"] else "__".join(_safe_component(model) for model in models_to_plot)
    )
    clean_component = "clean" if args.clean else "full"
    stem = (
        f"{_safe_component(args.benchmark)}__{_safe_component(args.hint_type)}__"
        f"{_safe_component(args.fractioner) if args.fractioner is not None else 'all_complete_fractioners'}__"
        f"{model_component}__{clean_component}"
    )
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    means_json_path = PLOTS_ROOT / f"{stem}__means_percent.json"
    summary_json_path = PLOTS_ROOT / f"{stem}__summary_percent.json"
    accuracy_table_csv_path = PLOTS_ROOT / f"{stem}__accuracy_table.csv"
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

    accuracy_table_source_rows = _merge_rows_for_accuracy_table(rows_sorted, external_rows)
    accuracy_table_fieldnames, accuracy_table_rows = _build_accuracy_table_rows(
        accuracy_table_source_rows
    )
    _print_accuracy_table(accuracy_table_fieldnames, accuracy_table_rows)
    _write_accuracy_table_csv(
        fieldnames=accuracy_table_fieldnames,
        table_rows=accuracy_table_rows,
        output_path=accuracy_table_csv_path,
    )

    plot_rows = rows_sorted
    plot_fit_map = fit_map
    plot_external_rows = external_rows
    plot_external_fit_map = external_fit_map
    if args.clean:
        plot_rows = _merge_rows_for_clean_plot(rows_sorted, external_rows)
        plot_fit_map = _build_fit_map(plot_rows)
        plot_external_rows = []
        plot_external_fit_map = {}

    _plot(
        plot_rows,
        fit_map=plot_fit_map,
        external_results=plot_external_rows,
        external_fit_map=plot_external_fit_map,
        output_png=png_path,
        show_values=args.show_values,
        title=(
            f"Hinted Accuracy vs Hint Fraction\n"
            f"benchmark={args.benchmark} hint_type={args.hint_type} "
            f"({args.fractioner if args.fractioner is not None else 'all complete fractioners'}) "
            f"[{'clean' if args.clean else 'full'}]"
        ),
    )

    print(f"[plot_hinted_accuracy_vs_hint] wrote_means_json= {means_json_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_summary_json= {summary_json_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_accuracy_table_csv= {accuracy_table_csv_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_plot= {png_path}")

    if args.include_split_plots:
        for split_name, split_predicate in _split_plot_specs(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
        ):
            split_rows, split_external_rows = _collect_rows_for_models(
                benchmark=args.benchmark,
                hint_type=args.hint_type,
                models_to_plot=models_to_plot,
                fractioner=args.fractioner,
                problem_id_predicate=split_predicate,
            )
            if not split_rows and not split_external_rows:
                print(
                    "[plot_hinted_accuracy_vs_hint][WARN] skipping split plot with no data "
                    f"split={split_name}"
                )
                continue

            split_rows_sorted = sorted(
                split_rows,
                key=lambda r: (str(r["model"]), str(r["fractioner"]), float(r["hint_fraction"])),
            )
            split_plot_rows = split_rows_sorted
            split_plot_fit_map = _build_fit_map(split_rows_sorted)
            split_plot_external_rows = split_external_rows
            split_plot_external_fit_map = _build_fit_map(split_external_rows)
            if args.clean:
                split_plot_rows = _merge_rows_for_clean_plot(split_rows_sorted, split_external_rows)
                split_plot_fit_map = _build_fit_map(split_plot_rows)
                split_plot_external_rows = []
                split_plot_external_fit_map = {}

            split_png_path = PLOTS_ROOT / f"{stem}__{split_name}__bootstrap.png"
            _plot(
                split_plot_rows,
                fit_map=split_plot_fit_map,
                external_results=split_plot_external_rows,
                external_fit_map=split_plot_external_fit_map,
                output_png=split_png_path,
                show_values=args.show_values,
                title=(
                    f"Hinted Accuracy vs Hint Fraction ({split_name})\n"
                    f"benchmark={args.benchmark} hint_type={args.hint_type} "
                    f"({args.fractioner if args.fractioner is not None else 'all complete fractioners'}) "
                    f"[{'clean' if args.clean else 'full'}]"
                ),
            )
            print(f"[plot_hinted_accuracy_vs_hint] wrote_plot= {split_png_path}")


if __name__ == "__main__":
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word --clean
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed --clean
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner truncate_word --clean

    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark hle --hint-type answer_not_revealed --fractioner mask_word --clean --model Qwen/Qwen2.5-1.5B-Instruct
   

    main()
