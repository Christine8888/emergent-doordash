"""Configuration for BBH sampling script (hint data generation)."""

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

DEFAULT_INSTRUCTIONS = (
    "Solve the following problem carefully.\n\n"
    "Return your final answer on the last line in the format:\n"
    "ANSWER: <final answer>\n"
)


def _normalize_bbh_answer(answer: str) -> str:
    """Normalize a BBH answer for comparison.

    Handles both MCQ answers like (b), B), b and free-text answers.
    Strips surrounding parentheses, trailing ), and lowercases.
    """
    s = answer.strip()
    # Strip surrounding parentheses: (b) -> b
    if len(s) >= 2 and s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    # Strip trailing ) without leading (: B) -> B
    elif len(s) >= 2 and s.endswith(")") and "(" not in s:
        s = s[:-1].strip()
    return s.lower()


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
    """Grade by normalizing both sides (case-insensitive, strip parens)."""
    if not extracted_answer:
        return False
    return _normalize_bbh_answer(extracted_answer) == _normalize_bbh_answer(str(target))


def format_prompt(sample: Sample) -> str:
    """Format prompt with generic free-form instructions (including choices if present)."""
    from hints.sample_utils import sample_input_to_str, format_choices_for_prompt

    input_text = sample_input_to_str(sample.input)
    choices_str = format_choices_for_prompt(getattr(sample, "choices", None))
    if choices_str:
        input_text = input_text + "\n\n" + choices_str
    return DEFAULT_INSTRUCTIONS + input_text


def extract_sample_fields(sample: Sample) -> dict:
    """Extract additional fields for sample_to_dict."""
    return {}


@task
def bbh_task(sample_ids=None, solver=None):
    from inspect_evals.bbh.bbh import bbh_scorer
    dataset = get_dataset()
    if sample_ids is not None:
        sample_ids_str = {str(sid) for sid in sample_ids}
        dataset = dataset.filter(lambda s: str(s.id) in sample_ids_str)
    if solver is None:
        from evals.solvers import instructions, generate
        solver = [instructions(DEFAULT_INSTRUCTIONS), generate()]
    return Task(dataset=dataset, solver=solver, scorer=bbh_scorer())

