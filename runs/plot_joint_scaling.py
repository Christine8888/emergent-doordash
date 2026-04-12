from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hinted_accuracy import discover_models_for_benchmark, load_results_with_ci_for_combo


PLOTS_ROOT = Path("plots/joint_scaling_plots")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs ECI, one curve per hint fraction.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument("--eci-file", type=str, required=True)
    return parser.parse_args()


def _load_eci_map(path: Path) -> dict[str, float]:
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "model" not in reader.fieldnames or "eci_our_fit" not in reader.fieldnames:
            raise ValueError(f"Expected columns 'model' and 'eci_our_fit' in {path}")

        out: dict[str, float] = {}
        for row in reader:
            model = str(row.get("model", "")).strip()
            eci_raw = row.get("eci_our_fit")
            if not model or eci_raw in (None, ""):
                continue
            out[model] = float(eci_raw)
    return out


def _eci_benchmark_label(path: Path) -> str:
    stem = path.stem
    prefix = "eci_model_capabilities__simple__"
    if not stem.startswith(prefix):
        return "unknown"
    encoded = stem[len(prefix) :]
    if not encoded:
        return "unknown"
    return ", ".join(encoded.split("--"))


def _sigmoid_curve(x: np.ndarray, lower: float, slope: float, bias: float) -> np.ndarray:
    return lower + (1.0 - lower) * (1.0 / (1.0 + np.exp(-(slope * x + bias))))


def _fit_sigmoid(xs: list[float], ys: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(xs) < 4:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.allclose(y, y[0]):
        return None

    try:
        from scipy.optimize import curve_fit

        lower0 = float(np.clip(np.min(y) - 0.02, 0.0, 0.95))
        y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
        mid_idx = int(np.argmin(np.abs(y - y_mid)))
        x_mid = float(x[mid_idx])
        slope0 = 0.2
        bias0 = -slope0 * x_mid

        params, _ = curve_fit(
            _sigmoid_curve,
            x,
            y,
            p0=[lower0, slope0, bias0],
            bounds=([0.0, 1e-6, -200.0], [0.99, 10.0, 200.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    x_fit = np.linspace(float(np.min(x)) - 2.0, float(np.max(x)) + 2.0, 200, dtype=float)
    y_fit = _sigmoid_curve(x_fit, *params)
    return x_fit, y_fit


def main() -> None:
    args = _parse_args()
    eci_path = Path(args.eci_file)
    eci_map = _load_eci_map(eci_path)
    eci_benchmark_label = _eci_benchmark_label(eci_path)
    combo_results = load_results_with_ci_for_combo(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )

    rows: list[dict[str, Any]] = []
    models = sorted(set(discover_models_for_benchmark(args.benchmark)) | set(combo_results.keys()))
    print(
        f"[plot_accuracy_vs_eci_by_hint] discovered_models={len(models)} "
        f"models={models}"
    )
    for model in models:
        if model not in combo_results:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] dropping model={model} "
                f"reason=no_combo_results"
            )
            continue
        if model not in eci_map:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] dropping model={model} "
                f"reason=missing_eci"
            )
            continue
        for hint_fraction, stats in sorted(combo_results[model].items()):
            rows.append(
                {
                    "model": model,
                    "fractioner": args.fractioner,
                    "hint_fraction": float(hint_fraction),
                    "accuracy": float(stats["accuracy"]),
                    "ci_low": float(stats["ci_low"]),
                    "ci_high": float(stats["ci_high"]),
                    "eci": float(eci_map[model]),
                }
            )

    if not rows:
        raise ValueError("No usable rows found after combining hinted accuracy with ECI data.")

    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}

    for hint_fraction in hint_fractions:
        series_rows = sorted(
            [row for row in rows if float(row["hint_fraction"]) == hint_fraction],
            key=lambda row: float(row["eci"]),
        )
        xs = [float(row["eci"]) for row in series_rows]
        ys = [float(row["accuracy"]) for row in series_rows]
        color = colors[hint_fraction]

        ax.scatter(xs, ys, color=color, alpha=0.85, s=45, label=f"h={hint_fraction:.2f}")

        fit = _fit_sigmoid(xs, ys)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color=color, alpha=0.7, linewidth=2)

    plotted_models = sorted(
        {
            (str(row["model"]), float(row["eci"]))
            for row in rows
        },
        key=lambda item: item[1],
    )
    ax.set_xticks([eci for _, eci in plotted_models])
    ax.set_xticklabels([model for model, _ in plotted_models], rotation=60, ha="right", fontsize=8)

    ax.set_xlabel("Model (positioned by ECI)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Accuracy vs ECI by Hint Fraction\n"
        f"benchmark={args.benchmark} hint_type={args.hint_type} fractioner={args.fractioner}\n"
        f"eci_benchmarks={eci_benchmark_label}",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    output_dir = PLOTS_ROOT / f"{args.benchmark}__{args.hint_type}__{args.fractioner}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "accuracy_vs_eci_by_hint.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_accuracy_vs_eci_by_hint] {output_path}")


if __name__ == "__main__":
    # python -m runs.plot_joint_scaling --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh--hellaswag--mmlu_5_shot_cot--piqa--winogrande.csv

    main()
