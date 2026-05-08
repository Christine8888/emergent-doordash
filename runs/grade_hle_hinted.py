from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.hle_hinted_grading import discover_hle_hinted_models, grade_hle_hinted_outputs


def _parse_fractions(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return [i / 10 for i in range(11)]
    fractions: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        fractions.append(float(part))
    return fractions


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid bool value: {value!r}")


def _now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _log(message: str) -> None:
    print(f"[{_now_hhmm()}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grade HLE hinted inference outputs into sidecar JSONL files.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to grade. If omitted, grades all discovered models with pending HLE hinted outputs.",
    )
    parser.add_argument("--hint-type", default="answer_not_revealed")
    parser.add_argument("--fractioner", required=True)
    parser.add_argument(
        "--hint-fractions",
        default=None,
        help="Comma-separated fractions. Defaults to 0.0,0.1,...,1.0.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--grader-concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", type=_parse_bool, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hint_fractions = _parse_fractions(args.hint_fractions)

    if args.model:
        grade_hle_hinted_outputs(
            model=args.model,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            hint_fractions=hint_fractions,
            data_root=args.data_root,
            grader_concurrency=args.grader_concurrency,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        return

    models = discover_hle_hinted_models(
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        hint_fractions=hint_fractions,
        data_root=args.data_root,
    )
    if not models:
        _log(
            "[hle_hinted_grading] no models discovered with matching hinted inference files "
            f"for hint_type={args.hint_type} fractioner={args.fractioner}"
        )
        return

    _log(
        f"[hle_hinted_grading] discovered_models={len(models)} "
        f"hint_type={args.hint_type} fractioner={args.fractioner}"
    )
    remaining_limit = args.limit
    for model in models:
        if remaining_limit is not None and remaining_limit <= 0:
            break
        _log(f"[hle_hinted_grading] grading model={model}")
        summary = grade_hle_hinted_outputs(
            model=model,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            hint_fractions=hint_fractions,
            data_root=args.data_root,
            grader_concurrency=args.grader_concurrency,
            limit=remaining_limit,
            dry_run=args.dry_run,
        )
        if remaining_limit is not None:
            remaining_limit = max(0, remaining_limit - int(summary.get("pending", 0)))


if __name__ == "__main__":
    main()


"""
python -m runs.grade_hle_hinted \
    --hint-type answer_not_revealed \
    --fractioner mask_word \
    --grader-concurrency 10
"""