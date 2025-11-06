"""Configuration for HLE sampling script."""

from inspect_ai.dataset import Sample
from environments.hle.hle import get_hle_dataset, DEFAULT_INSTRUCTIONS
from environments.gpqa.utils import extract_answer as gpqa_extract_answer, grade_answer as gpqa_grade_answer


def get_dataset():
    """Load HLE dataset from HuggingFace."""
    return get_hle_dataset(split="test")


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    return gpqa_extract_answer(response, num_choices=4)


async def grade_answer(response: str, target: str) -> bool:
    """Grade HLE answer using letter matching."""
    extracted = extract_answer(response)
    return await gpqa_grade_answer(extracted, target)


def format_prompt(sample: Sample) -> str:
    """Format HLE question.

    The question already contains the answer choices,
    so we just add instructions to the model.
    """
    return DEFAULT_INSTRUCTIONS + "\n\n" + sample.input


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {
        "category": sample.metadata.get("category") if sample.metadata else None,
        "raw_subject": sample.metadata.get("raw_subject") if sample.metadata else None,
    }
