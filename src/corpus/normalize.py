"""Normalization helpers for score/extraction fields."""

from __future__ import annotations

from typing import Any


def normalize_score_value(value: Any) -> str:
    """Map heterogeneous score values to C/I/U."""
    if value is None:
        return "U"

    if isinstance(value, bool):
        return "C" if value else "I"

    if isinstance(value, (int, float)):
        if value == 1:
            return "C"
        if value == 0:
            return "I"
        return "U"

    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"C", "CORRECT", "TRUE", "T", "YES", "Y"}:
            return "C"
        if v in {"I", "INCORRECT", "FALSE", "F", "NO", "N"}:
            return "I"
        if v in {"U", "UNKNOWN", "N/A", "NA", ""}:
            return "U"
        return "U"

    return "U"


def score_extracted_answer(score_dict: dict[str, Any]) -> str | None:
    """Extract model answer from scorer payload, if present."""
    answer = score_dict.get("answer")
    if answer is not None and str(answer).strip() != "":
        return str(answer)

    metadata = score_dict.get("metadata")
    if isinstance(metadata, dict):
        extracted = metadata.get("extracted_answer")
        if extracted is not None and str(extracted).strip() != "":
            return str(extracted)
    return None


def extraction_status(extracted_answer: str | None) -> str:
    return "ok" if extracted_answer is not None and extracted_answer.strip() != "" else "failed"

