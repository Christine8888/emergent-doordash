from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hinted_accuracy import (
    DATA_ROOT,
    EXTERNAL_RESULTS_WITH_CI_PATHS,
    collect_complete_fraction_stats,
    discover_models_for_benchmark,
    external_results_to_payload,
    load_external_results_with_ci,
    rows_to_results_with_ci_payload,
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


def _merge_ci_payloads(
    base: dict[str, dict[str, dict[str, float]]],
    override: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Union of model → hint-fraction stats; ``override`` wins on duplicate fraction keys."""
    out: dict[str, dict[str, dict[str, float]]] = {model: dict(frac_map) for model, frac_map in base.items()}
    for model, frac_map in override.items():
        out.setdefault(model, {}).update(frac_map)
    return out


def _build_local_payload(*, benchmark: str, hint_type: str, fractioner: str) -> dict[str, dict[str, dict[str, float]]]:
    rows = []
    models = discover_models_for_benchmark(benchmark)
    for model in models:
        model_rows, warnings = collect_complete_fraction_stats(
            benchmark=benchmark,
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            data_root=DATA_ROOT,
        )
        if not model_rows:
            if warnings:
                print(
                    f"[export_result_summary][WARN] skipping local model={model} "
                    f"hint_type={hint_type} fractioner={fractioner} warnings={warnings}"
                )
            continue

        rows.extend(model_rows)
        means_text = ", ".join(
            f"{float(row['hint_fraction']):.1f}:{float(row['accuracy']):.4f}"
            for row in model_rows
        )
        print(
            f"[export_result_summary] local model={model} "
            f"hint_type={hint_type} fractioner={fractioner} {means_text}"
        )
    return rows_to_results_with_ci_payload(rows)


def main() -> None:
    args = _parse_args()

    external_path = EXTERNAL_RESULTS_WITH_CI_PATHS.get(args.fractioner)
    if external_path is None:
        raise ValueError(
            f"No external results-with-CI file configured for fractioner={args.fractioner!r}. "
            f"Known: {sorted(EXTERNAL_RESULTS_WITH_CI_PATHS)}"
        )

    external_payload = external_results_to_payload(load_external_results_with_ci(external_path))
    local_payload = _build_local_payload(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    payload = _merge_ci_payloads(external_payload, local_payload)

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
    main()
