"""Configuration for ARC sampling script."""

from inspect_ai.dataset import Sample
from environments.arc.arc import get_arc_dataset
from environments.arc.utils import extract_answer as arc_extract_answer, grade_answer as arc_grade_answer


def get_dataset(test_case_seed: int = 42):
    """Load ARC dataset from local files.

    Args:
        test_case_seed: Seed for selecting test case when multiple exist (default: 42)
    """
    return get_arc_dataset(split="training", test_case_seed=test_case_seed)


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    return arc_extract_answer(response)


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Grade ARC answer using normalization-based exact match."""
    return await arc_grade_answer(extracted_answer, target)


def format_prompt(sample: Sample) -> str:
    """Format ARC question.

    The prompt is already fully constructed in sample.input,
    so we just return it directly.
    """
    return sample.input


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {
        "task_id": sample.metadata.get("task_id") if sample.metadata else None,
        "test_idx": sample.metadata.get("test_idx") if sample.metadata else None,
    }
