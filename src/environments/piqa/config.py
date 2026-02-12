"""Configuration for PIQA sampling script (hint data generation)."""

from inspect_ai.dataset import Sample

from environments.gpqa.utils import extract_answer as mcq_extract_answer

NUM_CHOICES = 2

DEFAULT_INSTRUCTIONS = (
    "Answer the following multiple choice question. "
    "Think step by step before answering. "
    "The last line of your response should be of the following format: "
    "'ANSWER: $LETTER' (without quotes) where LETTER is one of the options."
)


def get_dataset():
    """Load PIQA dataset via inspect-evals Task."""
    from inspect_evals.piqa import piqa

    task = piqa()
    return task.dataset


def extract_answer(response: str) -> str:
    """Extract answer letter from response (A/B)."""
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


def _strip_parens(s: str) -> str:
    """Strip surrounding parentheses: (a) -> a, a) -> a."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    elif s.endswith(")") and "(" not in s:
        s = s[:-1].strip()
    return s


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Grade by comparing extracted letter to target (with light normalization)."""
    if not extracted_answer:
        return False
    return _strip_parens(extracted_answer).upper() == _normalize_target(target)


def format_prompt(sample: Sample) -> str:
    """Format prompt with generic multiple-choice instructions (including choices)."""
    from hints.sample_utils import sample_input_to_str, format_choices_for_prompt

    input_text = sample_input_to_str(sample.input)
    choices_str = format_choices_for_prompt(getattr(sample, "choices", None))
    if choices_str:
        input_text = input_text + "\n\n" + choices_str
    return DEFAULT_INSTRUCTIONS + "\n\n" + input_text


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}

