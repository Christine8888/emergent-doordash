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

from runs.fit_eci import EVAL_TO_ECI, load_baseline_scores
from src.hinted_accuracy import discover_models_for_benchmark, load_results_with_ci_for_combo


PLOTS_ROOT = Path("plots/joint_scaling_plots")
PC_BENCHMARK_ORDER = [EVAL_TO_ECI[eval_name] for eval_name in EVAL_TO_ECI]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs capability, one curve per hint fraction.")
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


def _compute_pca_from_baselines() -> dict[str, Any]:
    scores_df = load_baseline_scores()
    df = scores_df[scores_df["benchmark"].isin(PC_BENCHMARK_ORDER)].copy()
    pivot = df.pivot(index="model", columns="benchmark", values="score")
    pivot = pivot.reindex(columns=PC_BENCHMARK_ORDER)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError("No models have complete benchmark coverage for PC capability.")

    matrix = pivot.to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std <= 0, 1.0, std)
    z = (matrix - mean) / std

    _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    pc1 = vt[0]
    scores = z @ pc1

    correlation = np.corrcoef(scores, z.mean(axis=1))[0, 1]
    if np.isnan(correlation):
        correlation = 1.0
    if correlation < 0:
        vt = -vt
        scores = -scores

    explained_variance = (singular_values**2) / max(z.shape[0] - 1, 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    return {
        "capability_map": {
        str(model): float(score)
        for model, score in zip(pivot.index.tolist(), scores)
        },
        "benchmarks": list(PC_BENCHMARK_ORDER),
        "components": np.asarray(vt, dtype=float),
        "explained_variance_ratio": np.asarray(explained_variance_ratio, dtype=float),
        "benchmark_label": ", ".join(eval_name for eval_name in EVAL_TO_ECI),
    }


def _plot_pca_component_weights(
    *,
    components: np.ndarray,
    benchmarks: list[str],
    output_path: Path,
) -> None:
    n_components = min(5, components.shape[0])
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(benchmarks)), 1.0 + 0.8 * n_components))
    image = ax.imshow(components[:n_components], aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_components))
    ax.set_yticklabels([f"PC-{i}" for i in range(1, n_components + 1)])
    ax.set_title("Principal component weights")
    fig.colorbar(image, ax=ax, shrink=0.9, label="weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_explained_variance(
    *,
    explained_variance_ratio: np.ndarray,
    output_path: Path,
) -> None:
    n_components = min(10, explained_variance_ratio.shape[0])
    x = np.arange(1, n_components + 1)
    y = explained_variance_ratio[:n_components]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, y, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Explained variance by component")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def _print_accuracy_summary_table(
    rows: list[dict[str, Any]],
    hint_fractions: list[float],
    *,
    capability_label: str,
) -> None:
    model_to_capability = {
        str(row["model"]): float(row["capability"])
        for row in rows
    }
    score_map: dict[tuple[str, float], float] = {}
    for row in rows:
        score_map[(str(row["model"]), float(row["hint_fraction"]))] = float(row["accuracy"])

    models = sorted(model_to_capability.keys(), key=lambda model: model_to_capability[model])
    model_width = max(len("Model"), max(len(model) for model in models))
    capability_width = max(len(capability_label), 10)
    frac_headers = [f"h={hint_fraction:.1f}" for hint_fraction in hint_fractions]
    frac_width = max(7, max(len(header) for header in frac_headers))

    header = [
        "Model".ljust(model_width),
        capability_label.rjust(capability_width),
        *[header.rjust(frac_width) for header in frac_headers],
    ]
    separator = [
        "-" * model_width,
        "-" * capability_width,
        *["-" * frac_width for _ in frac_headers],
    ]

    print("\nAccuracy summary by model and hint fraction:")
    print("  " + " ".join(header))
    print("  " + " ".join(separator))
    for model in models:
        row = [
            model.ljust(model_width),
            f"{model_to_capability[model]:.3f}".rjust(capability_width),
        ]
        for hint_fraction in hint_fractions:
            value = score_map.get((model, float(hint_fraction)))
            row.append("--".rjust(frac_width) if value is None else f"{value:.4f}".rjust(frac_width))
        print("  " + " ".join(row))


def _plot_capability_view(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    capability_method: str,
    capability_label: str,
    capability_benchmark_label: str,
    output_dir: Path,
) -> Path:
    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    _print_accuracy_summary_table(rows, hint_fractions, capability_label=capability_label)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}

    for hint_fraction in hint_fractions:
        series_rows = sorted(
            [row for row in rows if float(row["hint_fraction"]) == hint_fraction],
            key=lambda row: float(row["capability"]),
        )
        xs = [float(row["capability"]) for row in series_rows]
        ys = [float(row["accuracy"]) for row in series_rows]
        color = colors[hint_fraction]

        ax.scatter(xs, ys, color=color, alpha=0.85, s=45, label=f"h={hint_fraction:.2f}")

        fit = _fit_sigmoid(xs, ys)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color=color, alpha=0.7, linewidth=2)

    plotted_models = sorted(
        {
            (str(row["model"]), float(row["capability"]))
            for row in rows
        },
        key=lambda item: item[1],
    )
    ax.set_xticks([capability for _, capability in plotted_models])
    ax.set_xticklabels([model for model, _ in plotted_models], rotation=60, ha="right", fontsize=8)

    ax.set_xlabel(f"Model (positioned by {capability_label})", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Accuracy vs {capability_label} by Hint Fraction\n"
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}\n"
        f"{capability_method}_benchmarks={capability_benchmark_label}",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    output_path = output_dir / f"accuracy_vs_{capability_method}_by_hint.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_hint_fraction_for_filename(hint_fraction: float) -> str:
    return f"{hint_fraction:.2f}".replace(".", "p")


def _plot_capability_view_per_hint_with_error_bars(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    capability_method: str,
    capability_label: str,
    capability_benchmark_label: str,
    output_dir: Path,
 ) -> Path:
    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    n_panels = len(hint_fractions)
    ncols = min(3, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, hint_fraction in enumerate(hint_fractions):
        ax = axes[idx // ncols][idx % ncols]
        series_rows = sorted(
            [row for row in rows if float(row["hint_fraction"]) == hint_fraction],
            key=lambda row: float(row["capability"]),
        )
        if not series_rows:
            continue

        xs = np.asarray([float(row["capability"]) for row in series_rows], dtype=float)
        ys = np.asarray([float(row["accuracy"]) for row in series_rows], dtype=float)
        yerr = np.asarray(
            [
                [
                    max(0.0, float(row["accuracy"]) - float(row["ci_low"])),
                    max(0.0, float(row["ci_high"]) - float(row["accuracy"])),
                ]
                for row in series_rows
            ],
            dtype=float,
        ).T

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.5,
            capsize=4,
            alpha=0.85,
            markersize=6,
        )

        fit = _fit_sigmoid(xs.tolist(), ys.tolist())
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color="#ff7f0e", alpha=0.8, linewidth=2)

        ax.set_xlabel(capability_label, fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"hint_fraction={hint_fraction:.2f}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"Accuracy vs {capability_label} with Error Bars by Hint Fraction\n"
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}\n"
        f"{capability_method}_benchmarks={capability_benchmark_label}",
        fontsize=14,
    )
    output_path = output_dir / f"accuracy_vs_{capability_method}_by_hint_subplots_with_error_bars.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    eci_path = Path(args.eci_file)
    eci_map = _load_eci_map(eci_path)
    eci_benchmark_label = _eci_benchmark_label(eci_path)
    pca_result = _compute_pca_from_baselines()
    pc1_map = pca_result["capability_map"]
    pc_benchmark_label = pca_result["benchmark_label"]

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
    base_rows: list[dict[str, Any]] = []
    for model in models:
        if model not in combo_results:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] dropping model={model} "
                f"reason=no_combo_results"
            )
            continue
        for hint_fraction, stats in sorted(combo_results[model].items()):
            base_rows.append(
                {
                    "model": model,
                    "fractioner": args.fractioner,
                    "hint_fraction": float(hint_fraction),
                    "accuracy": float(stats["accuracy"]),
                    "ci_low": float(stats["ci_low"]),
                    "ci_high": float(stats["ci_high"]),
                }
            )

    if not base_rows:
        raise ValueError("No usable rows found after combining hinted accuracy with capability data.")

    output_dir = PLOTS_ROOT / f"{args.benchmark}__{args.hint_type}__{args.fractioner}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_pca_component_weights(
        components=pca_result["components"],
        benchmarks=pca_result["benchmarks"],
        output_path=output_dir / "pca_component_weights.png",
    )
    _plot_pca_explained_variance(
        explained_variance_ratio=pca_result["explained_variance_ratio"],
        output_path=output_dir / "pca_explained_variance.png",
    )
    views = [
        ("eci", "ECI", eci_map, eci_benchmark_label),
        ("pc1", "PC1", pc1_map, pc_benchmark_label),
    ]
    for capability_method, capability_label, capability_map, capability_benchmark_label in views:
        rows = []
        for row in base_rows:
            model = str(row["model"])
            if model not in capability_map:
                continue
            rows.append(
                {
                    **row,
                    "capability": float(capability_map[model]),
                }
            )
        if not rows:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] no rows for capability_method={capability_method}"
            )
            continue

        missing_models = sorted({str(row["model"]) for row in base_rows} - set(capability_map.keys()))
        for model in missing_models:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] dropping model={model} "
                f"reason=missing_{capability_method}"
            )

        output_path = _plot_capability_view(
            rows=rows,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            capability_method=capability_method,
            capability_label=capability_label,
            capability_benchmark_label=capability_benchmark_label,
            output_dir=output_dir,
        )
        print(f"[plot_accuracy_vs_eci_by_hint] {output_path}")
        per_hint_output_path = _plot_capability_view_per_hint_with_error_bars(
            rows=rows,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            capability_method=capability_method,
            capability_label=capability_label,
            capability_benchmark_label=capability_benchmark_label,
            output_dir=output_dir,
        )
        print(f"[plot_accuracy_vs_eci_by_hint] {per_hint_output_path}")


if __name__ == "__main__":
    # python -m runs.plot_joint_scaling --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh--hellaswag--mmlu_5_shot_cot--piqa--winogrande.csv
    # python -m runs.plot_joint_scaling --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner truncate_word --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh--hellaswag--mmlu_5_shot_cot--piqa--winogrande.csv

    main()
