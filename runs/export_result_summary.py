from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hinted_accuracy import (
    DATA_ROOT,
    load_results_with_ci_for_combo,
    safe_component,
)


EXPORT_ROOT = DATA_ROOT / "results_with_ci_by_combo"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one results-with-CI summary file per hint_type+fractioner, merging "
            "the configured external CI file with stats aggregated from this repo's rollouts."
        )
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path. Defaults to data/results_with_ci_by_combo/<benchmark>/<hint_type>__<fractioner>.json",
    )
    return parser.parse_args()


def _default_output_path(*, benchmark: str, hint_type: str, fractioner: str) -> Path:
    combo_name = f"{safe_component(hint_type)}__{safe_component(fractioner)}.json"
    return EXPORT_ROOT / safe_component(benchmark) / combo_name


def main() -> None:
    args = _parse_args()
    combo_results = load_results_with_ci_for_combo(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    payload = {
        model: {
            f"{float(hint_fraction):.1f}": {
                "mean": float(stats["accuracy"]),
                "ci_lower": float(stats["ci_low"]),
                "ci_upper": float(stats["ci_high"]),
            }
            for hint_fraction, stats in sorted(fraction_map.items())
        }
        for model, fraction_map in sorted(combo_results.items())
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[export_result_summary] {output_path}")


if __name__ == "__main__":
    # python -m runs.export_result_summary --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner truncate_word
    # python -m runs.export_result_summary --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word
    main()
