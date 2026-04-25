from __future__ import annotations

import argparse

from src.hint_types import HintType
from src.hinted_progress import ModelHintProgress, compute_model_hint_progress, print_progress_report
from src.model_config import ALL_MODEL_PATHS, filter_models_for_fractioner, get_model_spec

DEFAULT_HINT_FRACTIONS = [i / 10 for i in range(11)]


def _parse_hint_fractions(value: str) -> list[float]:
    pieces = [part.strip() for part in value.split(",")]
    fractions: list[float] = []
    for piece in pieces:
        if not piece:
            continue
        fractions.append(float(piece))
    if not fractions:
        raise ValueError("hint-fractions cannot be empty")
    return fractions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print hinted inference progress by model and hint type.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument(
        "--hint-type",
        type=str,
        choices=["all"] + [h.value for h in HintType],
        default="all",
    )
    parser.add_argument(
        "--hint-fractions",
        type=_parse_hint_fractions,
        default=list(DEFAULT_HINT_FRACTIONS),
        help="Comma-separated fractions, e.g. 0,0.1,0.2,...,1.0",
    )
    parser.add_argument("--data-root", type=str, default="data")
    return parser


def _selected_models(fractioner: str) -> list[str]:
    model_paths = filter_models_for_fractioner(list(ALL_MODEL_PATHS), fractioner)
    if not model_paths:
        raise ValueError(f"All models are excluded for fractioner={fractioner!r}")
    return [get_model_spec(model_path).name for model_path in model_paths]


def _selected_hint_types(hint_type: str) -> list[str]:
    if hint_type == "all":
        return [h.value for h in HintType]
    return [hint_type]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_names = _selected_models(args.fractioner)
    hint_types = _selected_hint_types(args.hint_type)

    rows: list[ModelHintProgress] = []
    for hint_type in hint_types:
        for model_name in model_names:
            rows.append(
                compute_model_hint_progress(
                    benchmark_name=args.benchmark,
                    model=model_name,
                    hint_type=hint_type,
                    fractioner=args.fractioner,
                    hint_fractions=args.hint_fractions,
                    data_root=args.data_root,
                )
            )

    print_progress_report(rows, show_complete=False)


if __name__ == "__main__":
    main()

"""
python -m runs.print_hinted_progress --benchmark aime2025_2026 --fractioner mask_word --hint-type answer_not_revealed
python -m runs.print_hinted_progress --benchmark aime2025_2026 --fractioner truncate_word --hint-type answer_not_revealed

"""
