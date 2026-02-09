"""Configuration for BBH sampling script (hint data generation)."""

import re

from inspect_ai.dataset import Sample

DEFAULT_INSTRUCTIONS = (
    "Solve the following problem carefully.\n\n"
    "Return your final answer on the last line in the format:\n"
    "ANSWER: <final answer>\n"
)


def get_dataset():
    """Load BBH dataset via inspect-evals Task."""
    from inspect_evals.bbh import bbh

    task = bbh()
    return task.dataset


def extract_answer(response: str) -> str:
    """Best-effort extraction for diverse BBH tasks.

    Strategy:
    - If there is an 'ANSWER:' line, use the last one.
    - Otherwise, use the last non-empty line.
    """
    matches = re.findall(r"(?im)^\s*answer\s*:\s*(.+)\s*$", response)
    if matches:
        return matches[-1].strip()

    for line in reversed(response.splitlines()):
        if line.strip():
            return line.strip()
    return ""


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Conservative grading: strip and compare to target."""
    return extracted_answer.strip() == str(target).strip()


def format_prompt(sample: Sample) -> str:
    """Format prompt with generic free-form instructions."""
    return DEFAULT_INSTRUCTIONS + sample.input


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}

