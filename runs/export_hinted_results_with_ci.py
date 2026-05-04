from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hinted_accuracy import load_results_with_ci_for_combo


DEFAULT_BENCHMARK = "aime2025_2026"
DEFAULT_HINT_TYPE = "answer_not_revealed"
DEFAULT_FRACTIONERS = ("mask_word", "truncate_word")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export hinted accuracy means and bootstrap confidence intervals to "
            "results_with_ci_{fractioner}.json files."
        )
    )
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK)
    parser.add_argument("--hint-type", type=str, default=DEFAULT_HINT_TYPE)
    parser.add_argument(
        "--fractioner",
        type=str,
        nargs="+",
        default=list(DEFAULT_FRACTIONERS),
        help="One or more fractioners to export.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing hinted_inference and receiving output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output JSON files. Defaults to --data-root.",
    )
    return parser.parse_args()


def _format_payload(
    results: dict[str, dict[float, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    payload: dict[str, dict[str, dict[str, float]]] = {}
    for model, fraction_map in sorted(results.items()):
        payload[model] = {}
        for hint_fraction, stats in sorted(fraction_map.items()):
            payload[model][f"{float(hint_fraction):.1f}"] = {
                "mean": round(float(stats["accuracy"]), 6),
                "ci_lower": round(float(stats["ci_low"]), 6),
                "ci_upper": round(float(stats["ci_high"]), 6),
            }
    return payload


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.data_root
    output_dir.mkdir(parents=True, exist_ok=True)

    for fractioner in args.fractioner:
        results = load_results_with_ci_for_combo(
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=fractioner,
            data_root=args.data_root,
        )
        payload = _format_payload(results)
        output_path = output_dir / f"results_with_ci_{fractioner}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

        print(f"[export_hinted_results_with_ci] wrote {output_path} models={len(payload)}")


if __name__ == "__main__":
    # python -m runs.export_hinted_results_with_ci
    # python -m runs.export_hinted_results_with_ci --fractioner mask_word truncate_word
    main()
