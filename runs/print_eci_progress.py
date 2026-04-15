from __future__ import annotations

import argparse

from runs.generate_eci import BENCHMARKS
from src.eci_progress import compute_eci_benchmark_progress, print_eci_progress_report
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


if __name__ == "__main__":
    # python -m runs.print_eci_progress
    # python -m runs.print_eci_progress --benchmark arc_challenge
    # python -m runs.print_eci_progress --model Qwen/Qwen2.5-1.5B-Instruct
    main()
