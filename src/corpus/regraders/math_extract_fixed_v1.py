"""Post-hoc regrader using extract_answer_fixed + canonical math grading."""

from __future__ import annotations

from typing import Any

from environments.math.utils import extract_answer_fixed, grade_math_answer

from corpus.normalize import normalize_score_value
from corpus.regraders.base import RegraderSpec


def applies(rollout_row: dict[str, Any]) -> bool:
    # Benchmark targeting is controlled by SPEC.benchmark_slugs.
    # Here we only enforce row-level requirements.
    return rollout_row.get("output_text") is not None


def extracted(output: str, extracted_answer: str) -> bool:
    # extract_answer_fixed falls back to completion text when it cannot extract.
    return extracted_answer.strip() != output.strip()


async def run(rollout_row: dict[str, Any]) -> dict[str, Any]:
    output = str(rollout_row.get("output_text") or "")
    target = str(rollout_row.get("target") or "")
    new_answer = extract_answer_fixed(output)

    if not extracted(output=output, extracted_answer=new_answer):
        return {
            "score_raw_value": None,
            "score_normalized": "U",
            "extracted_answer": new_answer,
            "extraction_status": "failed",
            "explanation": "No answer extracted by fixed math extractor.",
            "metadata": {"extracted": False},
        }

    correct = await grade_math_answer(
        answer=new_answer,
        target=target,
        exact_match=True,
        use_sympy=True,
    )
    raw = "C" if correct else "I"
    return {
        "score_raw_value": raw,
        "score_normalized": normalize_score_value(raw),
        "extracted_answer": new_answer,
        "extraction_status": "ok",
        "explanation": "Fixed math regrader result.",
        "metadata": {"extracted": True},
    }


SPEC = RegraderSpec(
    name="math_extract_fixed_v1",
    version="math_extract_fixed_v1",
    benchmark_slugs=("aime", "math", "math_level_5_task"),
    applies=applies,
    run=run,
)
