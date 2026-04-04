from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraderResult(BaseModel):
    """Result from one extractor+grader pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor_grader_type: str
    extracted_answer: str | None
    is_correct: bool | None
    metadata: dict[str, Any]


class HintGenerationRecord(BaseModel):
    """One generated hint for one problem and rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hint_id: str
    problem_id: str
    benchmark_name: str
    hint_type: str
    rollout_id: int
    generator_model: str
    question: str
    answer: str
    model_output: str
    full_hint: str
    input_token_count: int
    output_token_count: int
    created_at: str = Field(default_factory=_utcnow_iso, frozen=True)
    metadata: dict[str, Any]


class HintedInferenceRecord(BaseModel):
    """One inference output using one hint at one hint fraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inference_id: str
    problem_id: str
    benchmark_name: str
    model: str
    hint_type: str
    fractioner: str
    hint_fraction: float
    hint_text_used: str
    model_output: str
    input_token_count: int
    output_token_count: int
    cost: float
    is_error: bool
    graders: list[GraderResult]
    hint: HintGenerationRecord
    created_at: str = Field(default_factory=_utcnow_iso, frozen=True)
    metadata: dict[str, Any]

    @field_validator("hint_fraction")
    @classmethod
    def validate_hint_fraction(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("hint_fraction must be in [0.0, 1.0]")
        return value
