"""Configuration for GPQA sampling script."""

from inspect_ai.dataset import Sample
from environments.gpqa.gpqa import get_gpqa_dataset, DEFAULT_INSTRUCTIONS
from environments.gpqa.utils import extract_answer as gpqa_extract_answer, grade_answer as gpqa_grade_answer


def get_dataset(shuffle_seed: int = 42):
    """Load GPQA dataset with shuffled choices.

    Args:
        shuffle_seed: Random seed for shuffling choices (default: 42).
    """
    return get_gpqa_dataset(shuffle_seed=shuffle_seed)


def extract_answer(response: str) -> str:
    """Extract answer letter from response."""
    return gpqa_extract_answer(response, num_choices=4)


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Grade GPQA answer by comparing extracted letter to target."""
    return await gpqa_grade_answer(extracted_answer, target)


def format_prompt(sample: Sample) -> str:
    """Format GPQA question.

    Question already has choices formatted, just add instructions.
    """
    return DEFAULT_INSTRUCTIONS + "\n\n" + sample.input


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}
