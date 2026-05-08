from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
import math
import re
from pathlib import Path
from typing import Any

from src.datasets import Problem, get_dataset_spec
from src.storage import (
    append_jsonl,
    build_hinted_grade_path,
    build_hinted_inference_path,
    make_stable_id,
    read_jsonl,
    write_jsonl,
)
from src.types import HintedInferenceRecord


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fraction_text(value: float) -> str:
    return f"{value:.6f}"


def _strip_latex_wrappers(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"^\s*\\boxed\s*\{(.*)\}\s*$", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"^\s*\\fbox\s*\{(.*)\}\s*$", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"^\s*\$(.*)\$\s*$", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"^\s*\\\((.*)\\\)\s*$", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"^\s*\\\[(.*)\\\]\s*$", r"\1", value, flags=re.DOTALL)
    return value.strip()


def _normalize_answer_text(text: str | None) -> str:
    if text is None:
        return ""
    value = _strip_latex_wrappers(text)
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip("\"'`")
    value = value.rstrip(".")
    return value.casefold()


def _parse_numeric(text: str | None) -> float | None:
    if text is None:
        return None
    value = _strip_latex_wrappers(text)
    value = value.replace(",", "").replace("\u2212", "-").strip()
    value = re.sub(r"^\s*[~≈]\s*", "", value)
    value = re.sub(r"\\approx|\\simeq|\\sim", "", value).strip()
    frac_match = re.fullmatch(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", value)
    try:
        if frac_match is not None:
            numerator = Fraction(frac_match.group(1).strip())
            denominator = Fraction(frac_match.group(2).strip())
            if denominator == 0:
                return None
            return float(numerator / denominator)
        if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
            return float(Fraction(value.replace(" ", "")))
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?", value):
            is_percent = value.endswith("%")
            number = float(value[:-1] if is_percent else value)
            return number / 100.0 if is_percent else number
    except Exception:
        return None
    return None


def _local_exact_match(extracted_answer: str | None, gold_answer: str) -> tuple[bool, str]:
    normalized_pred = _normalize_answer_text(extracted_answer)
    normalized_gold = _normalize_answer_text(gold_answer)
    if normalized_pred and normalized_pred == normalized_gold:
        return True, "normalized_string_exact"

    pred_number = _parse_numeric(extracted_answer)
    gold_number = _parse_numeric(gold_answer)
    if pred_number is not None and gold_number is not None:
        if math.isclose(pred_number, gold_number, rel_tol=1e-6, abs_tol=1e-9):
            return True, "numeric_close"
    return False, "no_local_match"


def _problem_from_record(row: HintedInferenceRecord) -> Problem:
    problem_metadata = row.hint.metadata.get("problem_metadata")
    if not isinstance(problem_metadata, dict):
        problem_metadata = {}
    metadata = dict(problem_metadata)
    metadata.setdefault("answer_type", row.hint.metadata.get("problem_answer_type"))
    return Problem(
        problem_id=row.problem_id,
        question=row.hint.question,
        answer=row.hint.answer,
        source=str(row.hint.metadata.get("problem_source", "")),
        metadata=metadata,
    )


def _grade_one(row: HintedInferenceRecord, dataset_spec: Any) -> dict[str, Any]:
    problem = _problem_from_record(row)
    answer_type = str(problem.metadata.get("answer_type") or "")
    extracted_answer = dataset_spec.extract_answer(row.model_output)

    if answer_type == "multipleChoice":
        grade_result = dataset_spec.grade_response(row.model_output, problem)
        grader_type = str(grade_result["metadata"].get("grader_type", "hle_multiple_choice_exact_match"))
        return _grade_record(
            row=row,
            extracted_answer=grade_result["extracted_answer"],
            is_correct=bool(grade_result["is_correct"]),
            grader_type=grader_type,
            metadata=grade_result["metadata"],
        )

    if answer_type == "exactMatch":
        local_match, local_method = _local_exact_match(extracted_answer, problem.answer)
        if local_match:
            return _grade_record(
                row=row,
                extracted_answer=extracted_answer,
                is_correct=True,
                grader_type="hle_exact_match_local",
                metadata={
                    "answer_type": answer_type,
                    "local_method": local_method,
                    "gold_answer": problem.answer,
                },
            )
        try:
            grade_result = dataset_spec.grade_response(row.model_output, problem)
        except Exception as exc:
            return _grade_record(
                row=row,
                extracted_answer=extracted_answer,
                is_correct=None,
                grader_type="hle_llm_judge_error",
                metadata={
                    "answer_type": answer_type,
                    "judge_error_type": type(exc).__name__,
                    "judge_error": str(exc),
                    "needs_regrade": True,
                },
            )

        metadata = dict(grade_result["metadata"])
        if str(metadata.get("grader_type")) == "hle_official_style_llm_judge_error":
            metadata["needs_regrade"] = True
            return _grade_record(
                row=row,
                extracted_answer=grade_result["extracted_answer"],
                is_correct=None,
                grader_type="hle_llm_judge_error",
                metadata=metadata,
            )
        return _grade_record(
            row=row,
            extracted_answer=grade_result["extracted_answer"],
            is_correct=bool(grade_result["is_correct"]),
            grader_type=str(metadata.get("grader_type", "hle_official_style_llm_judge")),
            metadata=metadata,
        )

    local_match, local_method = _local_exact_match(extracted_answer, problem.answer)
    return _grade_record(
        row=row,
        extracted_answer=extracted_answer,
        is_correct=local_match,
        grader_type="hle_unknown_answer_type_local",
        metadata={
            "answer_type": answer_type,
            "local_method": local_method,
            "gold_answer": problem.answer,
        },
    )


def _grade_record(
    *,
    row: HintedInferenceRecord,
    extracted_answer: str | None,
    is_correct: bool | None,
    grader_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    grade_id = make_stable_id(row.inference_id, grader_type, length=16)
    return {
        "grade_id": grade_id,
        "inference_id": row.inference_id,
        "problem_id": row.problem_id,
        "benchmark_name": row.benchmark_name,
        "model": row.model,
        "hint_type": row.hint_type,
        "fractioner": row.fractioner,
        "hint_fraction": row.hint_fraction,
        "hint_id": row.hint.hint_id,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
        "grader_type": grader_type,
        "metadata": metadata,
        "created_at": _utcnow_iso(),
    }


def _grade_needs_regrade(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    return (
        row.get("grader_type") == "hle_llm_judge_error"
        or row.get("is_correct") is None
        or (isinstance(metadata, dict) and metadata.get("needs_regrade") is True)
    )


def _read_existing_completed_grade_ids(path: Path) -> set[str]:
    existing: set[str] = set()
    for row in read_jsonl(path, model_cls=None):
        if isinstance(row, dict) and isinstance(row.get("inference_id"), str):
            if not _grade_needs_regrade(row):
                existing.add(row["inference_id"])
    return existing


def _remove_regrade_rows(path: Path, current_inference_ids: set[str]) -> int:
    if not path.exists() or not current_inference_ids:
        return 0

    rows = [row for row in read_jsonl(path, model_cls=None) if isinstance(row, dict)]
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        inference_id = row.get("inference_id")
        if (
            isinstance(inference_id, str)
            and inference_id in current_inference_ids
            and _grade_needs_regrade(row)
        ):
            removed += 1
            continue
        kept.append(row)

    if removed:
        write_jsonl(path, kept)
    return removed


def grade_hle_hinted_outputs(
    *,
    model: str,
    hint_type: str,
    fractioner: str,
    hint_fractions: list[float],
    data_root: str | Path = "data",
    grader_concurrency: int = 8,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if grader_concurrency < 1:
        raise ValueError("grader_concurrency must be >= 1")

    total_pending = 0
    total_written = 0
    total_skipped_existing = 0
    total_llm_judge_errors = 0
    total_removed_regrade_rows = 0
    by_grader_type: dict[str, int] = {}

    for hint_fraction in sorted(set(float(f"{value:.6f}") for value in hint_fractions)):
        inference_path = build_hinted_inference_path(
            benchmark_name="hle",
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            data_root=data_root,
        )
        grade_path = build_hinted_grade_path(
            benchmark_name="hle",
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            hint_fraction=hint_fraction,
            data_root=data_root,
        )
        rows = [
            row
            for row in read_jsonl(inference_path, model_cls=HintedInferenceRecord)
            if isinstance(row, HintedInferenceRecord) and not row.is_error
        ]
        existing_inference_ids = _read_existing_completed_grade_ids(grade_path)
        pending = [row for row in rows if row.inference_id not in existing_inference_ids]
        total_skipped_existing += len(rows) - len(pending)
        if limit is not None:
            remaining_limit = max(0, limit - total_pending)
            pending = pending[:remaining_limit]
        total_pending += len(pending)
        print(
            f"[hle_hinted_grading] fraction={hint_fraction} rows={len(rows)} "
            f"pending={len(pending)} skipped_existing={len(rows) - len(pending)} "
            f"inference_path={inference_path} grade_path={grade_path}",
            flush=True,
        )
        if dry_run or not pending:
            if limit is not None and total_pending >= limit:
                break
            continue

        pending_inference_ids = {row.inference_id for row in pending}
        removed_regrade_rows = _remove_regrade_rows(grade_path, pending_inference_ids)
        total_removed_regrade_rows += removed_regrade_rows
        if removed_regrade_rows:
            print(
                f"[hle_hinted_grading] removed_regrade_rows={removed_regrade_rows} "
                f"grade_path={grade_path}",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=grader_concurrency) as executor:
            dataset_spec = get_dataset_spec("hle")
            futures = [executor.submit(_grade_one, row, dataset_spec) for row in pending]
            for future in as_completed(futures):
                grade = future.result()
                append_jsonl(grade_path, grade)
                total_written += 1
                grader_type = str(grade.get("grader_type", "unknown"))
                by_grader_type[grader_type] = by_grader_type.get(grader_type, 0) + 1
                if grader_type == "hle_llm_judge_error":
                    total_llm_judge_errors += 1
        if limit is not None and total_pending >= limit:
            break

    summary = {
        "pending": total_pending,
        "written": total_written,
        "skipped_existing": total_skipped_existing,
        "removed_regrade_rows": total_removed_regrade_rows,
        "hle_llm_judge_error": total_llm_judge_errors,
        "by_grader_type": by_grader_type,
    }
    print(f"[hle_hinted_grading] summary={summary}", flush=True)
    if total_llm_judge_errors:
        print(
            f"[hle_hinted_grading][WARN] hle_llm_judge_error count={total_llm_judge_errors}",
            flush=True,
        )
    else:
        print("[hle_hinted_grading] hle_llm_judge_error count=0", flush=True)
    return summary
