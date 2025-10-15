"""Configuration for GPQA sampling script."""

from inspect_ai.dataset import Sample
from environments.gpqa.gpqa import get_gpqa_dataset
from evals.solvers.mcq_utils import parse_answer, format_answer_options
from evals.solvers.mcq_solver import DEFAULT_INSTRUCTIONS, DEFAULT_EXAMPLE_TEMPLATE


def get_dataset():
    """Load GPQA dataset with shuffled choices."""
    return get_gpqa_dataset()


def extract_answer(response: str) -> str:
    """Extract answer letter from response."""
    return parse_answer(response, num_choices=4) or ""


async def grade_answer(response: str, target: str) -> bool:
    """Grade GPQA answer by comparing extracted letter to target."""
    answer = parse_answer(response, num_choices=4)
    return answer == target


def format_prompt(sample: Sample) -> str:
    """Format GPQA question with shuffled choices using solver template."""
    choices_text = format_answer_options(sample.choices)
    question_with_choices = f"{sample.input}\n\n{choices_text}"

    # Use the same template structure as the solver
    current_task = DEFAULT_EXAMPLE_TEMPLATE.format(
        question=question_with_choices,
        solution=""
    )
    return DEFAULT_INSTRUCTIONS + "\n\n" + current_task


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    choices_text = format_answer_options(sample.choices)
    question_with_choices = f"{sample.input}\n\n{choices_text}"

    return {
        "choices": sample.choices,
        "question_with_choices": question_with_choices,
    }
