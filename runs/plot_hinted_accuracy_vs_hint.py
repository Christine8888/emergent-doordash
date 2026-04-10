from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/accuracy_vs_hint")
N_BOOTSTRAP = 5000
RANDOM_SEED = 0
EXPECTED_FRACTIONS = [i / 10 for i in range(11)]


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hinted inference accuracy vs hint fraction with bootstrap CIs."
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


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_is_correct(row: dict[str, Any]) -> bool | None:
    graders = row.get("graders")
    if not isinstance(graders, list):
        return None

    for grader in graders:
        if not isinstance(grader, dict):
            continue
        is_correct = grader.get("is_correct")
        if isinstance(is_correct, bool):
            return is_correct
    return None


def _discover_models(benchmark: str) -> list[str]:
    benchmark_dir = DATA_ROOT / "hinted_inference" / _safe_component(benchmark)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Missing benchmark directory: {benchmark_dir}")
    return sorted(path.name for path in benchmark_dir.iterdir() if path.is_dir())


def _parse_fraction_from_filename(name: str) -> float:
    match = re.match(r"^fraction_(.+)\.jsonl$", name)
    if not match:
        raise ValueError(f"Unexpected fraction filename: {name}")
    return float(match.group(1))


def _discover_fraction_files(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
    fractioner: str,
) -> list[tuple[float, Path]]:
    benchmark_name = _safe_component(benchmark)
    model_name = _safe_component(model)
    hint_fractioner = f"{_safe_component(hint_type)}__{_safe_component(fractioner)}"
    combo_dir = DATA_ROOT / "hinted_inference" / benchmark_name / model_name / hint_fractioner

    if not combo_dir.exists():
        return []

    out: list[tuple[float, Path]] = []
    for path in combo_dir.glob("fraction_*.jsonl"):
        try:
            fraction = _parse_fraction_from_filename(path.name)
        except ValueError:
            continue
        out.append((fraction, path))
    return sorted(out, key=lambda pair: pair[0])


def _checkpoint_path_for_fraction(path: Path) -> Path:
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected .jsonl path, got: {path}")
    return path.with_suffix(".ckpt.json")


def _is_complete_fraction(path: Path) -> tuple[bool, str | None]:
    ckpt_path = _checkpoint_path_for_fraction(path)
    if not ckpt_path.exists():
        return False, f"missing checkpoint {ckpt_path}"

    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
    except Exception as exc:
        return False, f"failed to read checkpoint {ckpt_path}: {exc}"

    if not isinstance(ckpt, dict):
        return False, f"invalid checkpoint payload {ckpt_path}"

    total_candidates = ckpt.get("total_candidates")
    processed_this_run = ckpt.get("processed_this_run")
    skipped_existing = ckpt.get("skipped_existing")
    remaining = ckpt.get("remaining")

    if not isinstance(total_candidates, int) or total_candidates < 0:
        return False, f"invalid total_candidates in {ckpt_path}"
    if not isinstance(processed_this_run, int) or processed_this_run < 0:
        return False, f"invalid processed_this_run in {ckpt_path}"
    if not isinstance(skipped_existing, int) or skipped_existing < 0:
        return False, f"invalid skipped_existing in {ckpt_path}"
    if not isinstance(remaining, int) or remaining < 0:
        return False, f"invalid remaining in {ckpt_path}"

    completed_total = processed_this_run + skipped_existing
    if remaining != 0:
        return False, f"remaining={remaining}"
    if completed_total < total_candidates:
        return False, (
            f"incomplete completed_total={completed_total} total_candidates={total_candidates}"
        )
    return True, None


def _discover_fractioners(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
) -> list[str]:
    benchmark_name = _safe_component(benchmark)
    model_name = _safe_component(model)
    hint_prefix = f"{_safe_component(hint_type)}__"
    model_dir = DATA_ROOT / "hinted_inference" / benchmark_name / model_name
    if not model_dir.exists():
        return []

    fractioners: list[str] = []
    for path in model_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(hint_prefix):
            continue
        parts = path.name.split("__", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        fractioners.append(parts[1])
    return sorted(set(fractioners))


def _bootstrap_accuracy(
    *,
    sample_to_scores: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    sample_arrays = list(sample_to_scores.values())
    if not sample_arrays:
        raise ValueError("No sample arrays available for bootstrap.")

    point_accuracy = float(np.mean([arr.mean() for arr in sample_arrays]))

    boot_sums = np.zeros(N_BOOTSTRAP, dtype=float)
    for arr in sample_arrays:
        draw_idx = rng.integers(low=0, high=arr.size, size=N_BOOTSTRAP)
        boot_sums += arr[draw_idx]
    boot_means = boot_sums / float(len(sample_arrays))
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    return point_accuracy, float(ci_low), float(ci_high)


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


def _collect_stats_for_fraction(
    *,
    path: Path,
    rng: np.random.Generator,
) -> dict[str, float | int] | None:
    sample_to_scores: dict[str, list[float]] = {}
    rows_total = 0
    rows_with_known_label = 0
    rows_without_known_label = 0

    for row in _iter_jsonl(path):
        rows_total += 1

        if not isinstance(row, dict):
            rows_without_known_label += 1
            continue

        problem_id = str(row.get("problem_id", "")).strip()
        if not problem_id:
            rows_without_known_label += 1
            continue

        is_correct = _extract_is_correct(row)
        if is_correct is None:
            rows_without_known_label += 1
            continue

        rows_with_known_label += 1
        sample_to_scores.setdefault(problem_id, []).append(1.0 if is_correct else 0.0)

    sample_to_arrays = {
        sample_id: np.asarray(values, dtype=float)
        for sample_id, values in sample_to_scores.items()
        if len(values) > 0
    }
    if not sample_to_arrays:
        return None

    point_accuracy, ci_low, ci_high = _bootstrap_accuracy(
        sample_to_scores=sample_to_arrays,
        rng=rng,
    )
    n_rollouts = int(sum(arr.size for arr in sample_to_arrays.values()))

    return {
        "accuracy": point_accuracy,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_samples": int(len(sample_to_arrays)),
        "n_rollouts": n_rollouts,
        "rows_total": int(rows_total),
        "rows_with_known_label": int(rows_with_known_label),
        "rows_without_known_label": int(rows_without_known_label),
    }


def _plot(
    results: list[dict[str, Any]],
    *,
    fit_map: dict[tuple[str, str], dict[str, float]],
    output_png: Path,
    show_values: bool,
    title: str,
) -> None:
    models = sorted({row["model"] for row in results})
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
    models_available = _discover_models(args.benchmark)
    if not models_available:
        raise ValueError(f"No models found for benchmark={args.benchmark!r}.")

    if args.model == "all":
        models_to_plot = models_available
    else:
        if args.model not in models_available:
            raise ValueError(
                f"Requested model {args.model!r} not found. "
                f"Available: {models_available}"
            )
        models_to_plot = [args.model]

    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []

    expected_fraction_set = {float(f"{value:.6f}") for value in EXPECTED_FRACTIONS}

    for model in models_to_plot:
        if args.fractioner is not None:
            fractioners = [args.fractioner]
        else:
            fractioners = _discover_fractioners(
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
            fraction_files = _discover_fraction_files(
                benchmark=args.benchmark,
                model=model,
                hint_type=args.hint_type,
                fractioner=fractioner,
            )
            if not fraction_files:
                print(
                    f"[plot_hinted_accuracy_vs_hint][WARN] no files for "
                    f"model={model} fractioner={fractioner}"
                )
                continue

            fraction_rows: list[dict[str, Any]] = []
            usable = False
            complete_fraction_files: list[tuple[float, Path]] = []
            incomplete_fraction_reasons: list[str] = []
            for hint_fraction, path in fraction_files:
                is_complete, reason = _is_complete_fraction(path)
                if is_complete:
                    complete_fraction_files.append((float(hint_fraction), path))
                else:
                    incomplete_fraction_reasons.append(
                        f"{float(hint_fraction):.1f}:{reason or 'incomplete'}"
                    )

            by_fraction = {
                float(f"{frac:.6f}"): path for frac, path in complete_fraction_files
            }
            available_fraction_set = set(by_fraction.keys())
            missing = sorted(expected_fraction_set - available_fraction_set)
            if missing:
                print(
                    f"[plot_hinted_accuracy_vs_hint][WARN] skipping incomplete fractioner "
                    f"model={model} fractioner={fractioner} missing_fractions={missing} "
                    f"incomplete_points={incomplete_fraction_reasons}"
                )
                continue
            fractions_to_use = [
                (float(h), by_fraction[float(f"{h:.6f}")]) for h in EXPECTED_FRACTIONS
            ]

            for hint_fraction, path in fractions_to_use:
                stats = _collect_stats_for_fraction(path=path, rng=rng)
                if stats is None:
                    print(
                        f"[plot_hinted_accuracy_vs_hint][WARN] skipping fraction point due to "
                        f"unusable fraction rows model={model} fractioner={fractioner} "
                        f"fraction={hint_fraction} path={path}"
                    )
                    continue
                fraction_rows.append(
                    {
                        "model": model,
                        "fractioner": fractioner,
                        "hint_fraction": float(hint_fraction),
                        **stats,
                        "path": str(path),
                    }
                )
                usable = True

            if not usable:
                continue

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
    main()
