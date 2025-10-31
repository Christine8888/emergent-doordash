"""Configuration for ARC sampling script."""

from inspect_ai.dataset import Sample
from environments.arc.arc import get_arc_dataset
from environments.arc.utils import extract_answer, grade_arc_answer


def get_dataset():
    """Load ARC dataset from local files."""
    return get_arc_dataset(split="training")


def extract_answer(response: str) -> str:
    """Extract answer from response."""
    from environments.arc.utils import extract_answer as arc_extract_answer
    return arc_extract_answer(response)


async def grade_answer(response: str, target: str) -> bool:
    """Grade ARC answer using normalization-based exact match."""
    return await grade_arc_answer(response, target)


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
