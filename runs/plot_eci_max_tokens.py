from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from runs.generate_eci import BENCHMARKS
from src.model_config import ALL_MODEL_PATHS
from src.storage import _model_storage_component, build_eci_score_path

DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/eci_token_distributions")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Plot per-model ECI output token distributions aggregated across selected benchmarks.")
    )
    parser.add_argument("--benchmark", type=str, choices=["all"] + BENCHMARKS, default="all")
    parser.add_argument("--model", type=str, choices=["all"] + list(ALL_MODEL_PATHS), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT,
        help="Directory for plots and summary CSV.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bins per subplot.",
    )
    return parser.parse_args()


def _selected_benchmarks(benchmark: str) -> list[str]:
    if benchmark == "all":
        return list(BENCHMARKS)
    return [benchmark]


def _selected_models(model: str) -> list[str]:
    if model == "all":
        return list(ALL_MODEL_PATHS)
    return [model]


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_model_rows(
    *,
    model: str,
    benchmark_names: list[str],
    data_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark_name in benchmark_names:
        path = build_eci_score_path(
            benchmark_name=benchmark_name,
            model=model,
            data_root=data_root,
        )
        if not path.exists():
            continue
        for record in _iter_jsonl(path):
            if not isinstance(record, dict):
                continue
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "model": model,
                    "output_token_count": record.get("output_token_count"),
                    "is_error": bool(record.get("is_error")),
                    "jsonl_path": str(path),
                }
            )
    return rows


def _values_for_metric(rows: list[dict[str, Any]], metric: str) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, int):
            values.append(value)
    return values


def _plot_all_models_distribution(
    *,
    rows: list[dict[str, Any]],
    models: list[str],
    benchmark_names: list[str],
    output_dir: Path,
    bins: int,
) -> Path:
    if not models:
        raise ValueError("No models with ECI rows found.")

    n_cols = min(4, len(models))
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), squeeze=False)
    axes_flat = list(axes.flatten())

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        model_rows = rows_by_model.get(model, [])
        output_values = _values_for_metric(model_rows, "output_token_count")
        if output_values:
            ax.hist(
                output_values,
                bins=bins,
                color="#ff7f0e",
                alpha=0.40,
                edgecolor="none",
                label=f"output (n={len(output_values)})",
            )
            if max(output_values) == 0:
                ax.set_xlim(-0.5, 0.5)
                ax.text(
                    0.5,
                    0.88,
                    "all output_token_count = 0",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#aa3333",
                )
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("output_token_count")
        ax.set_ylabel("count")
        ax.axvline(32000, color="#cc4444", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.grid(True, alpha=0.25)
        if output_values:
            ax.legend(loc="upper right", fontsize=8)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    error_count = sum(1 for row in rows if row.get("is_error"))
    fig.suptitle(
        (
            "ECI output token distributions by model\n"
            f"benchmarks={len(benchmark_names)} records={len(rows)} errors={error_count}"
        ),
        fontsize=13,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    if len(benchmark_names) == len(BENCHMARKS):
        benchmark_part = "all_benchmarks"
    else:
        benchmark_part = "__".join(_model_storage_component(benchmark) for benchmark in benchmark_names)
    out_path = output_dir / f"{benchmark_part}__eci_output_token_distributions_all_models.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_summary_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "eci_token_distribution_summary.csv"
    fieldnames = [
        "model",
        "benchmark",
        "record_count",
        "error_count",
        "output_token_mean",
        "output_token_max",
        "jsonl_path",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["benchmark"]))
        grouped.setdefault(key, []).append(row)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, benchmark), group_rows in sorted(grouped.items()):
            output_values = _values_for_metric(group_rows, "output_token_count")
            writer.writerow(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "record_count": len(group_rows),
                    "error_count": sum(1 for row in group_rows if row.get("is_error")),
                    "output_token_mean": (
                        f"{sum(output_values) / len(output_values):.2f}" if output_values else ""
                    ),
                    "output_token_max": max(output_values) if output_values else "",
                    "jsonl_path": build_eci_score_path(
                        benchmark_name=benchmark,
                        model=model,
                        data_root=DATA_ROOT,
                    ),
                }
            )
    return out_path


def main() -> None:
    args = _parse_args()
    benchmark_names = _selected_benchmarks(args.benchmark)
    models = _selected_models(args.model)

    all_rows: list[dict[str, Any]] = []
    plottable_models: list[str] = []
    for model in models:
        model_rows = _collect_model_rows(
            model=model,
            benchmark_names=benchmark_names,
            data_root=DATA_ROOT,
        )
        if not model_rows:
            print(f"skip model={model}: no ECI rows found", flush=True)
            continue
        all_rows.extend(model_rows)
        plottable_models.append(model)
        print(f"collected model={model} records={len(model_rows)}", flush=True)

    if not all_rows:
        raise ValueError("No ECI rows found for the requested benchmark/model filters.")

    plot_path = _plot_all_models_distribution(
        rows=all_rows,
        models=plottable_models,
        benchmark_names=benchmark_names,
        output_dir=args.output_dir,
        bins=args.bins,
    )
    csv_path = _write_summary_csv(all_rows, args.output_dir)
    print(f"wrote plot -> {plot_path}", flush=True)
    print(f"wrote summary -> {csv_path}", flush=True)
    print("plots=1", flush=True)


if __name__ == "__main__":
    # python -m runs.plot_eci_max_tokens
    main()
