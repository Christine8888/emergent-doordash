from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.storage import (
    build_expanded_hinted_prompt_path,
    build_hint_generation_path,
    build_hinted_inference_path,
)


@dataclass(frozen=True)
class ModelHintProgress:
    model: str
    hint_type: str
    fractioner: str
    completed: int
    total: int
    remaining: int
    fractions_complete: int
    fractions_total: int

    @property
    def percent_complete(self) -> float:
        if self.total <= 0:
            return 0.0
        return (100.0 * self.completed) / self.total


def _normalize_fractions(hint_fractions: list[float]) -> list[float]:
    normalized = {float(f"{float(value):.6f}") for value in hint_fractions}
    return sorted(normalized)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_unique_inference_ids(path: Path) -> int:
    if not path.exists():
        return 0

    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            inference_id = row.get("inference_id")
            if isinstance(inference_id, str) and inference_id:
                ids.add(inference_id)
    return len(ids)


def compute_model_hint_progress(
    *,
    benchmark_name: str,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fractions: list[float],
    data_root: str | Path = "data",
) -> ModelHintProgress:
    fractions = _normalize_fractions(hint_fractions)
    hint_generation_path = build_hint_generation_path(
        benchmark_name=benchmark_name,
        hint_type=hint_type,
        data_root=data_root,
    )
    hint_count_fallback = _count_jsonl_rows(hint_generation_path)

    total = 0
    completed = 0
    fractions_complete = 0

    for fraction in fractions:
        expanded_path = build_expanded_hinted_prompt_path(
            benchmark_name=benchmark_name,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=fraction,
            data_root=data_root,
        )
        fraction_total = _count_jsonl_rows(expanded_path)
        if fraction_total == 0:
            fraction_total = hint_count_fallback

        output_path = build_hinted_inference_path(
            benchmark_name=benchmark_name,
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=fraction,
            data_root=data_root,
        )
        fraction_completed = _count_unique_inference_ids(output_path)
        if fraction_total > 0:
            fraction_completed = min(fraction_completed, fraction_total)

        total += fraction_total
        completed += fraction_completed
        if fraction_total > 0 and fraction_completed >= fraction_total:
            fractions_complete += 1

    remaining = max(0, total - completed)
    return ModelHintProgress(
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        completed=completed,
        total=total,
        remaining=remaining,
        fractions_complete=fractions_complete,
        fractions_total=len(fractions),
    )


def print_progress_report(progress_rows: list[ModelHintProgress]) -> None:
    print("[hinted_progress] progress", flush=True)
    if not progress_rows:
        print("  no models selected", flush=True)
        return

    for row in progress_rows:
        is_complete = (
            row.total > 0
            and row.remaining == 0
            and row.fractions_complete >= row.fractions_total
        )
        if is_complete:
            print(f"  {row.model} complete", flush=True)
            continue

        print(
            (
                f"  {row.model} hint_type={row.hint_type} "
                f"completed={row.completed}/{row.total} ({row.percent_complete:.1f}%) "
                f"remaining={row.remaining} "
                f"fractions={row.fractions_complete}/{row.fractions_total}"
            ),
            flush=True,
        )

    total_completed = sum(row.completed for row in progress_rows)
    total_expected = sum(row.total for row in progress_rows)
    total_remaining = max(0, total_expected - total_completed)
    overall_pct = (100.0 * total_completed / total_expected) if total_expected > 0 else 0.0
    print(
        (
            f"  TOTAL completed={total_completed}/{total_expected} ({overall_pct:.1f}%) "
            f"remaining={total_remaining} models={len(progress_rows)}"
        ),
        flush=True,
    )


"""
python -m runs.print_hinted_progress \
    --benchmark aime2025_2026 \
    --fractioner truncate_sentence \
    --model all \
    --hint-type answer_not_revealed 
"""
