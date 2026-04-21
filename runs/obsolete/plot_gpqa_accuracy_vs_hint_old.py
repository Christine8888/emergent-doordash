from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


OLD_GPQA_ROOT = Path(
    "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/"
    "gpqa/solution_intext_masked/0shot"
)
PLOTS_ROOT = Path("plots/accuracy_vs_hint")


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GPQA OLD accuracy vs hint fraction from christine_experiments/20251113."
    )
    parser.add_argument("--models", nargs="+", required=True, help="One or more model names.")
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Annotate plotted points with their mean accuracy values.",
    )
    return parser.parse_args()


def _parse_fraction_from_filename(name: str) -> float:
    match = re.match(r"^gpqa_solution_intext_masked_0shot_(.+)\.json$", name)
    if not match:
        raise ValueError(f"Unexpected GPQA filename: {name}")
    return float(match.group(1))


def _iter_model_files(model: str) -> list[tuple[float, Path]]:
    model_dir = OLD_GPQA_ROOT / model
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    files: list[tuple[float, Path]] = []
    for path in model_dir.glob("gpqa_solution_intext_masked_0shot_*.json"):
        try:
            hint_fraction = _parse_fraction_from_filename(path.name)
        except ValueError:
            continue
        files.append((hint_fraction, path))
    return sorted(files, key=lambda item: item[0])


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    stderr = np.asarray(
        [max(1e-4, float(row["stderr"])) for row in rows_sorted],
        dtype=float,
    )

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
            sigma=stderr,
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


def _collect_model_rows(model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hint_fraction, path in _iter_model_files(model):
        payload = _read_json(path)
        metric = payload.get("manual_bootstrap")
        if not isinstance(metric, dict):
            print(
                f"[plot_gpqa_accuracy_vs_hint_old][WARN] missing manual_bootstrap "
                f"model={model} path={path}"
            )
            continue

        accuracy = metric.get("accuracy")
        stderr = metric.get("stderr")
        if not isinstance(accuracy, (int, float)) or not isinstance(stderr, (int, float)):
            print(
                f"[plot_gpqa_accuracy_vs_hint_old][WARN] invalid manual_bootstrap metric "
                f"model={model} path={path}"
            )
            continue

        ci_low = max(0.0, float(accuracy) - 1.96 * float(stderr))
        ci_high = min(1.0, float(accuracy) + 1.96 * float(stderr))
        rows.append(
            {
                "model": model,
                "hint_fraction": float(hint_fraction),
                "accuracy": float(accuracy),
                "stderr": float(stderr),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "path": str(path),
            }
        )

    if not rows:
        raise ValueError(f"No usable GPQA rows found for model={model!r}.")
    return sorted(rows, key=lambda row: float(row["hint_fraction"]))


def _plot(
    results: list[dict[str, Any]],
    *,
    fit_map: dict[str, dict[str, float]],
    output_png: Path,
    show_values: bool,
    title: str,
) -> None:
    models = sorted({str(row["model"]) for row in results})
    n_models = len(models)

    if n_models == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axes = [ax]
    else:
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
        axes = axes_obj.flatten() if hasattr(axes_obj, "flatten") else [axes_obj]

    cmap = plt.cm.tab10
    series_color = cmap(0.0)
    for idx, model in enumerate(models):
        ax = axes[idx]
        color = series_color
        model_rows = sorted(
            [row for row in results if row["model"] == model],
            key=lambda row: float(row["hint_fraction"]),
        )
        x = np.asarray([float(row["hint_fraction"]) for row in model_rows], dtype=float)
        y = np.asarray([float(row["accuracy"]) for row in model_rows], dtype=float)
        low = np.asarray([float(row["ci_low"]) for row in model_rows], dtype=float)
        high = np.asarray([float(row["ci_high"]) for row in model_rows], dtype=float)
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
            label=model,
        )
        if show_values:
            for x_i, y_i in zip(x, y):
                ax.annotate(
                    f"{y_i:.2f}",
                    (float(x_i), float(y_i)),
                    xytext=(5, 9),
                    textcoords="offset points",
                    fontsize=6,
                    color=color,
                    alpha=0.9,
                )

        fit = fit_map.get(model)
        if fit is not None:
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200, dtype=float)
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

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    all_rows: list[dict[str, Any]] = []
    for model in args.models:
        model_rows = _collect_model_rows(model)
        all_rows.extend(model_rows)
        means_text = ", ".join(
            f"{float(row['hint_fraction']):.2f}:{float(row['accuracy']):.4f}"
            for row in model_rows
        )
        print(f"[plot_gpqa_accuracy_vs_hint_old] means model={model} {means_text}")

    fit_map: dict[str, dict[str, float]] = {}
    series_map: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        series_map.setdefault(str(row["model"]), []).append(row)

    for model, series_rows in sorted(series_map.items()):
        fit = _fit_sigmoid_for_series(series_rows)
        if fit is None:
            print(f"[plot_gpqa_accuracy_vs_hint_old][WARN] sigmoid fit failed/skipped model={model}")
            continue
        fit_map[model] = fit
        print(
            f"[plot_gpqa_accuracy_vs_hint_old] fit model={model} "
            f"midpoint={float(fit['sigmoid_midpoint']):.4f} "
            f"rmse={float(fit['sigmoid_rmse']):.4f}"
        )

    model_slug = (
        _safe_component(args.models[0])
        if len(args.models) == 1
        else f"multi_{len(args.models)}_models"
    )
    stem = f"gpqa_old_solution_intext_masked__{model_slug}"
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_json_path = PLOTS_ROOT / f"{stem}__summary_percent.json"
    png_path = PLOTS_ROOT / f"{stem}__bootstrap.png"

    summary_payload: dict[str, dict[str, dict[str, float]]] = {}
    for model, series_rows in sorted(series_map.items()):
        summary_payload[model] = {
            f"{float(row['hint_fraction']):.2f}": {
                "mean": round(100.0 * float(row["accuracy"]), 1),
                "ci_low": round(100.0 * float(row["ci_low"]), 1),
                "ci_high": round(100.0 * float(row["ci_high"]), 1),
            }
            for row in sorted(series_rows, key=lambda row: float(row["hint_fraction"]))
        }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    _plot(
        all_rows,
        fit_map=fit_map,
        output_png=png_path,
        show_values=args.show_values,
        title=(
            "GPQA OLD Accuracy vs Hint Fraction\n"
            "source=christine_experiments/20251113/results/gpqa/solution_intext_masked/0shot"
        ),
    )

    print(f"[plot_gpqa_accuracy_vs_hint_old] wrote_summary_json={summary_json_path}")
    print(f"[plot_gpqa_accuracy_vs_hint_old] wrote_plot={png_path}")


if __name__ == "__main__":
    # python -m runs.plot_gpqa_accuracy_vs_hint_old   --models Qwen3-4B Qwen2.5-7B-Instruct gemma-3-4b-it --show-values
    main()
