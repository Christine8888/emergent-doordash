"""Configuration for HellaSwag sampling script (hint data generation)."""

from inspect_ai.dataset import Sample

from environments.gpqa.utils import extract_answer as mcq_extract_answer

NUM_CHOICES = 4

DEFAULT_INSTRUCTIONS = (
    "Answer the following multiple choice question. "
    "Think step by step before answering. "
    "The last line of your response should be of the following format: "
    "'ANSWER: $LETTER' (without quotes) where LETTER is one of the options."
)


def get_dataset():
    """Load HellaSwag dataset via inspect-evals Task."""
    from inspect_evals.hellaswag import hellaswag

    task = hellaswag()
    return task.dataset


def extract_answer(response: str) -> str:
    """Extract answer letter from response."""
    return mcq_extract_answer(response, num_choices=NUM_CHOICES)


def _normalize_target(target: str) -> str:
    t = str(target).strip()
    if len(t) == 1 and t.isdigit():
        idx = int(t)
        if 0 <= idx < NUM_CHOICES:
            return chr(ord("A") + idx)
        if 1 <= idx <= NUM_CHOICES:
            return chr(ord("A") + (idx - 1))
    if len(t) == 1 and t.isalpha():
        return t.upper()
    return t


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Grade by comparing extracted letter to target (with light normalization)."""
    if not extracted_answer:
        return False
    return extracted_answer.strip().upper() == _normalize_target(target)


def format_prompt(sample: Sample) -> str:
    """Format prompt with generic multiple-choice instructions."""
    return DEFAULT_INSTRUCTIONS + "\n\n" + sample.input


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}

