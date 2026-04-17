from __future__ import annotations

import argparse

from runs.generate_eci import BENCHMARKS, _get_active_eci_jobs_by_model
from src.eci_progress import ECIBenchmarkProgress, compute_eci_benchmark_progress, print_eci_progress_report
from src.model_config import ALL_MODEL_PATHS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print ECI progress by benchmark and model.")
    parser.add_argument("--benchmark", type=str, choices=["all"] + BENCHMARKS, default="all")
    parser.add_argument("--model", type=str, choices=["all"] + list(ALL_MODEL_PATHS), default="all")
    parser.add_argument("--data-root", type=str, default="data")
    return parser


def _selected_benchmarks(benchmark: str) -> list[str]:
    if benchmark == "all":
        return list(BENCHMARKS)
    return [benchmark]


def _selected_models(model: str) -> list[str]:
    if model == "all":
        return list(ALL_MODEL_PATHS)
    return [model]


def _print_models_to_run_summary(rows: list[ECIBenchmarkProgress]) -> None:
    print("  Models neither running nor queued", flush=True)
    if not rows:
        print("    none", flush=True)
        return

    active_jobs_by_model = _get_active_eci_jobs_by_model()
    rows_by_model: dict[str, list[ECIBenchmarkProgress]] = {}
    for row in rows:
        rows_by_model.setdefault(row.model, []).append(row)

    ready_models: list[str] = []
    for model, model_rows in sorted(rows_by_model.items()):
        if all(row.status == "complete" for row in model_rows):
            continue
        if model in active_jobs_by_model:
            continue
        ready_models.append(model)

    if not ready_models:
        print("    none", flush=True)
        return

    for model in ready_models:
        print(f"    {model}", flush=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    rows = [
        compute_eci_benchmark_progress(
            benchmark_name=benchmark_name,
            model=model,
            data_root=args.data_root,
        )
        for benchmark_name in _selected_benchmarks(args.benchmark)
        for model in _selected_models(args.model)
    ]
    print_eci_progress_report(rows)
    _print_models_to_run_summary(rows)


if __name__ == "__main__":
    # python -m runs.print_eci_progress
    main()
