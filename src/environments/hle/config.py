"""Configuration for HLE sampling script."""

from inspect_ai.dataset import Sample
from environments.hle.hle import get_hle_dataset, DEFAULT_INSTRUCTIONS
from environments.hle.utils import extract_answer as hle_extract_answer, grade_hle_answer


def get_dataset():
    """Load HLE dataset from HuggingFace."""
    return get_hle_dataset(split="test")


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    return hle_extract_answer(response)


async def grade_answer(response: str, target: str) -> bool:
    """Grade HLE answer using letter matching."""
    return await grade_hle_answer(response, target)


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
