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
        description=(
            "Plot per-model ECI token count distributions aggregated across selected benchmarks."
        )
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
                    "input_token_count": record.get("input_token_count"),
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


def _rows_by_benchmark(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["benchmark"]), []).append(row)
    return grouped


def _plot_model_distribution(
    *,
    model: str,
    rows: list[dict[str, Any]],
    benchmark_names: list[str],
    output_dir: Path,
    bins: int,
) -> Path:
    grouped = _rows_by_benchmark(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
    input_ax = axes[0][0]
    output_ax = axes[0][1]

    plotted_any = False
    for benchmark_name in benchmark_names:
        benchmark_rows = grouped.get(benchmark_name, [])
        input_values = _values_for_metric(benchmark_rows, "input_token_count")
        output_values = _values_for_metric(benchmark_rows, "output_token_count")
        if input_values:
            input_ax.hist(
                input_values,
                bins=bins,
                alpha=0.35,
                edgecolor="none",
                label=f"{benchmark_name} (n={len(input_values)})",
            )
            plotted_any = True
        if output_values:
            output_ax.hist(
                output_values,
                bins=bins,
                alpha=0.35,
                edgecolor="none",
                label=f"{benchmark_name} (n={len(output_values)})",
            )
            plotted_any = True

    if not plotted_any:
        raise ValueError(f"No token counts found for model={model}")

    input_ax.set_title("Input Token Distribution")
    input_ax.set_xlabel("input_token_count")
    input_ax.set_ylabel("count")
    input_ax.grid(True, alpha=0.25)
    input_ax.legend(loc="upper right", fontsize=8)

    output_ax.set_title("Output Token Distribution")
    output_ax.set_xlabel("output_token_count")
    output_ax.set_ylabel("count")
    output_ax.axvline(32000, color="#cc4444", linestyle="--", linewidth=1.0, alpha=0.8)
    output_ax.grid(True, alpha=0.25)
    output_ax.legend(loc="upper right", fontsize=8)

    error_count = sum(1 for row in rows if row.get("is_error"))
    fig.suptitle(
        (
            f"ECI token distributions for {model}\n"
            f"benchmarks={len(grouped)} records={len(rows)} errors={error_count}"
        ),
        fontsize=13,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_model_storage_component(model)}__eci_token_distributions.png"
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
        "input_token_mean",
        "input_token_max",
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
            input_values = _values_for_metric(group_rows, "input_token_count")
            output_values = _values_for_metric(group_rows, "output_token_count")
            writer.writerow(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "record_count": len(group_rows),
                    "error_count": sum(1 for row in group_rows if row.get("is_error")),
                    "input_token_mean": (
                        f"{sum(input_values) / len(input_values):.2f}" if input_values else ""
                    ),
                    "input_token_max": max(input_values) if input_values else "",
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
    plotted_paths: list[Path] = []
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
        plot_path = _plot_model_distribution(
            model=model,
            rows=model_rows,
            benchmark_names=benchmark_names,
            output_dir=args.output_dir,
            bins=args.bins,
        )
        plotted_paths.append(plot_path)
        print(
            f"plotted model={model} records={len(model_rows)} -> {plot_path}",
            flush=True,
        )

    if not all_rows:
        raise ValueError("No ECI rows found for the requested benchmark/model filters.")

    csv_path = _write_summary_csv(all_rows, args.output_dir)
    print(f"wrote summary -> {csv_path}", flush=True)
    print(f"plots={len(plotted_paths)}", flush=True)


if __name__ == "__main__":
    # python -m runs.plot_eci_token_distributions_all_models
    # python -m runs.plot_eci_token_distributions_all_models --model Qwen/Qwen3-0.6B
    # python -m runs.plot_eci_token_distributions_all_models --benchmark arc_challenge
    main()
