"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

from typing import Any
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Solver, TaskState

from environments.gpqa.utils import format_answer_options, extract_answer, grade_answer

LOCAL_DATA_DIR = Path(__file__).parent / "data"

# Default instructions for GPQA
DEFAULT_INSTRUCTIONS = (
    "Answer the following multiple choice question. "
    "The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of the options. "
    "Think step by step before answering."
)


def get_gpqa_dataset(shuffle_seed: int = 42, sample_ids: set[str] | None = None):
    """Load GPQA dataset with shuffled choices.

    Args:
        shuffle_seed: Random seed for shuffling choices (default: 42).
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
    """
    import functools

    # Create a version of record_to_sample with the seed bound
    record_to_sample_with_seed = functools.partial(record_to_sample, shuffle_seed=shuffle_seed)

    dataset = csv_dataset(
            csv_file=str(LOCAL_DATA_DIR / "gpqa_diamond.csv"),
            sample_fields=record_to_sample_with_seed,
    )

    # Filter to specific sample IDs if provided
    if sample_ids is not None:
        dataset = dataset.filter(
            name=f"{dataset.name}_filtered",
            predicate=lambda sample: sample.id in sample_ids
        )

    return dataset
@task
def gpqa_diamond(
    shuffle_seed: int | None = None,
    sample_ids: set[str] | None = None,
    solver: Solver | list[Solver] | None = None,
) -> Task:
    """
    Baseline GPQA Diamond task.

    This is the minimal task definition. For custom configurations (prefill, few-shot, etc.),
    use solver composition in your experiment file.

    Args:
        shuffle_seed: Random seed for shuffling choices (default: 42)
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
        solver: Custom solver or list of solvers. If None, uses basic generate() with instructions.

    Returns:
        Task configured for GPQA evaluation
    """
    # Load dataset
    dataset = get_gpqa_dataset(shuffle_seed=shuffle_seed or 42, sample_ids=sample_ids)

    # Use provided solver or create basic one
    if solver is None:
        from evals.solvers import format_prompt, generate_with_continuation
        solver = [
            format_prompt(instruction_template=DEFAULT_INSTRUCTIONS),
            generate_with_continuation()
        ]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=gpqa_scorer(),
    )


def record_to_sample(record: dict[str, Any], shuffle_seed: int = 42) -> Sample:
    """Convert GPQA CSV record to Sample with choices formatted into text.

    Args:
        record: CSV record
        shuffle_seed: Random seed for shuffling choices (default: 42).
    """
    import random

    # Collect all choices
    choices = [
        str(record["Correct Answer"]),
        str(record["Incorrect Answer 1"]),
        str(record["Incorrect Answer 2"]),
        str(record["Incorrect Answer 3"]),
    ]

    # Shuffle with seed for reproducibility
    rng = random.Random(shuffle_seed)
    choices_shuffled = choices.copy()
    rng.shuffle(choices_shuffled)

    # Find target letter based on where correct answer ended up
    target_idx = choices_shuffled.index(str(record["Correct Answer"]))
    target_letter = chr(ord('A') + target_idx)

    # Format choices as "A) ... B) ... C) ... D) ..."
    choices_text = format_answer_options(choices_shuffled)

    # Combine question with formatted choices
    question_with_choices = f"{record['Question']}\n\n{choices_text}"

    return Sample(
        id=record["Record ID"],
        input=question_with_choices,
        target=target_letter,
    )


def record_to_sample_prefill(record: dict[str, Any]) -> Sample:
    """Convert prefill JSONL record to Sample.

    Prefill records already have the question formatted with choices in the
    'question_with_choices' field.
    """
    return Sample(
        id=record["id"],
        input=record["question_with_choices"],  # Formatted question with choices
        target=record["target"],
    )


@scorer(metrics=[accuracy(), stderr()])
def gpqa_scorer() -> Scorer:
    """Score GPQA answers using letter extraction and comparison."""
    async def score(state: TaskState, target: Target) -> Score:
        extracted_letter = extract_answer(state.output.completion, num_choices=4)

        correct = await grade_answer(
            extracted_letter=extracted_letter,
            target=target.text,
        )

        if correct:
            return Score(
                value=CORRECT,
                answer=extracted_letter,
                explanation="Correct answer",
            )
        else:
            return Score(
                value=INCORRECT,
                answer=extracted_letter,
                explanation="Incorrect answer",
            )

    return score