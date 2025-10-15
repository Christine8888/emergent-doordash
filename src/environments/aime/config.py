"""Configuration for AIME sampling script."""

from inspect_ai.dataset import Sample
from environments.aime.aime import get_aime_dataset
from environments.math.utils import extract_answer as math_extract_answer, grade_math_answer
from evals.solvers.math_solver import DEFAULT_INSTRUCTIONS, DEFAULT_EXAMPLE_TEMPLATE


def get_dataset():
    """Load AIME dataset from HuggingFace."""
    return get_aime_dataset()


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    return math_extract_answer(response)


async def grade_answer(response: str, target: str) -> bool:
    """Grade AIME answer using canonical math grader with sympy."""
    answer = extract_answer(response)
    return await grade_math_answer(
        answer=answer,
        target=target,
        exact_match=True,
        use_sympy=True,
    )


def format_prompt(sample: Sample) -> str:
    """Format AIME question using solver template."""
    current_task = DEFAULT_EXAMPLE_TEMPLATE.format(
        question=sample.input,
        solution=""
    )
    return DEFAULT_INSTRUCTIONS + "\n\n" + current_task


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}
