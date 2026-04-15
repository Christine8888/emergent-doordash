from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.eci_runner import BENCHMARK_CONFIGS, EPOCHS, _load_task_sample_ids
from src.storage import _safe_component
from src.storage import build_eci_score_path
from src.types import ECIScoreRecord

BENCHMARK_ALIASES = {
    "mmlu_5_shot__language_en_us__cot_true": "MMLU",
    "bbh__prompt_type_answer_only": "BBH",
    "arc_challenge": "ARC",
    # "math__levels_5__fewshot_0": "MATH5",
    "hellaswag__split_validation": "Hella",
    "piqa": "PIQA",
    "winogrande__dataset_name_winogrande_xl__fewshot_5": "Wino",
}


@dataclass(frozen=True)
class ECIBenchmarkProgress:
    benchmark_name: str
    model: str
    completed: int
    total: int
    remaining: int
    status: str
    updated_at: str | None

    @property
    def percent_complete(self) -> float:
        if self.total <= 0:
            return 0.0
        return (100.0 * self.completed) / self.total


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


def _extract_is_correct(record: ECIScoreRecord) -> bool | None:
    for grader in record.graders:
        if isinstance(grader.is_correct, bool):
            return grader.is_correct
    return None


def _read_complete_benchmark_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None

    deduped: dict[str, ECIScoreRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = ECIScoreRecord.model_validate_json(line)
            except Exception:
                continue
            deduped.setdefault(record.inference_id, record)

    judged = [flag for flag in (_extract_is_correct(record) for record in deduped.values()) if flag is not None]
    if not judged:
        return None
    return sum(1.0 if flag else 0.0 for flag in judged) / len(judged)


def _checkpoint_path_for_output(output_path: Path) -> Path:
    if output_path.suffix == ".jsonl":
        return output_path.with_suffix(".ckpt.json")
    return output_path.with_name(output_path.name + ".ckpt.json")


def _read_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _default_total_samples(benchmark_name: str, *, data_root: str | Path) -> int:
    config = BENCHMARK_CONFIGS[benchmark_name]
    benchmark_dir = Path(data_root) / "eci_scores" / _safe_component(benchmark_name)
    if benchmark_dir.exists():
        totals: list[int] = []
        for ckpt_path in benchmark_dir.glob("*.ckpt.json"):
            ckpt = _read_checkpoint(ckpt_path)
            if ckpt is None:
                continue
            total_value = ckpt.get("total_samples")
            if isinstance(total_value, int) and total_value > 0:
                totals.append(total_value)
        if totals:
            return max(totals)
    try:
        return len(_load_task_sample_ids(config)) * EPOCHS
    except Exception:
        return 0


def compute_eci_benchmark_progress(
    *,
    benchmark_name: str,
    model: str,
    data_root: str | Path = "data",
) -> ECIBenchmarkProgress:
    output_path = build_eci_score_path(
        benchmark_name=benchmark_name,
        model=model,
        data_root=data_root,
    )
    ckpt_path = _checkpoint_path_for_output(output_path)
    ckpt = _read_checkpoint(ckpt_path)

    completed_from_rows = _count_unique_inference_ids(output_path)
    completed_from_ckpt = 0
    total_from_ckpt = 0
    updated_at: str | None = None
    if ckpt is not None:
        completed_value = ckpt.get("completed_samples")
        if isinstance(completed_value, int):
            completed_from_ckpt = completed_value
        total_value = ckpt.get("total_samples")
        if isinstance(total_value, int):
            total_from_ckpt = total_value
        updated_value = ckpt.get("updated_at")
        if isinstance(updated_value, str) and updated_value:
            updated_at = updated_value

    total = total_from_ckpt if total_from_ckpt > 0 else _default_total_samples(
        benchmark_name,
        data_root=data_root,
    )
    completed = max(completed_from_rows, completed_from_ckpt)
    if total > 0:
        completed = min(completed, total)
    remaining = max(0, total - completed)

    if total > 0 and remaining == 0:
        status = "complete"
    elif completed > 0 or ckpt is not None:
        status = "in_progress"
    else:
        status = "not_started"

    return ECIBenchmarkProgress(
        benchmark_name=benchmark_name,
        model=model,
        completed=completed,
        total=total,
        remaining=remaining,
        status=status,
        updated_at=updated_at,
    )


def print_eci_progress_report(rows: list[ECIBenchmarkProgress]) -> None:
    print("[eci_progress] progress", flush=True)
    if not rows:
        print("  no benchmarks or models selected", flush=True)
        return

    rows = sorted(rows, key=lambda row: (row.model, row.benchmark_name))
    benchmark_names = sorted({row.benchmark_name for row in rows})
    print(f"  benchmarks={', '.join(benchmark_names)}", flush=True)

    rows_by_model: dict[str, list[ECIBenchmarkProgress]] = {}
    for row in rows:
        rows_by_model.setdefault(row.model, []).append(row)

    for model, model_rows in rows_by_model.items():
        model_completed = sum(row.completed for row in model_rows)
        model_total = sum(row.total for row in model_rows)
        model_remaining = max(0, model_total - model_completed)
        model_pct = (100.0 * model_completed / model_total) if model_total > 0 else 0.0
        unfinished_rows = [row for row in model_rows if row.remaining > 0 or row.status != "complete"]

        if model_total > 0 and not unfinished_rows:
            print(f"  {model} complete", flush=True)
            continue

        print(
            (
                f"  {model} completed={model_completed}/{model_total} ({model_pct:.1f}%) "
                f"remaining={model_remaining}"
            ),
            flush=True,
        )
        for row in unfinished_rows:
            suffix = f" updated_at={row.updated_at}" if row.updated_at else ""
            print(
                (
                    f"    benchmark={row.benchmark_name} status={row.status} "
                    f"completed={row.completed}/{row.total} ({row.percent_complete:.1f}%) "
                    f"remaining={row.remaining}{suffix}"
                ),
                flush=True,
            )

    total_completed = sum(row.completed for row in rows)
    total_expected = sum(row.total for row in rows)
    total_remaining = max(0, total_expected - total_completed)
    overall_pct = (100.0 * total_completed / total_expected) if total_expected > 0 else 0.0
    print(
        (
            f"  TOTAL completed={total_completed}/{total_expected} ({overall_pct:.1f}%) "
            f"remaining={total_remaining} rows={len(rows)}"
        ),
        flush=True,
    )

    model_names = sorted(rows_by_model.keys())
    benchmark_order = benchmark_names
    model_width = max(len("Model"), max(len(model) for model in model_names))
    score_width = 8

    print("  Accuracy Table", flush=True)
    header = ["Model".ljust(model_width)] + [
        BENCHMARK_ALIASES.get(benchmark_name, benchmark_name)[:score_width].rjust(score_width)
        for benchmark_name in benchmark_order
    ]
    separator = ["-" * model_width] + ["-" * score_width for _ in benchmark_order]
    print("  " + " ".join(header), flush=True)
    print("  " + " ".join(separator), flush=True)

    progress_by_key = {(row.model, row.benchmark_name): row for row in rows}
    for model in model_names:
        table_row = [model.ljust(model_width)]
        for benchmark_name in benchmark_order:
            progress = progress_by_key[(model, benchmark_name)]
            if progress.status != "complete":
                table_row.append("-".rjust(score_width))
                continue
            output_path = build_eci_score_path(
                benchmark_name=benchmark_name,
                model=model,
                data_root="data",
            )
            accuracy = _read_complete_benchmark_accuracy(output_path)
            table_row.append("-".rjust(score_width) if accuracy is None else f"{accuracy:.4f}".rjust(score_width))
        print("  " + " ".join(table_row), flush=True)
