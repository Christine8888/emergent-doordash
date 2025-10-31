"""
HLE: Humanity's Last Exam

Dataset: cais/hle
https://huggingface.co/datasets/cais/hle
"""
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset, json_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from evals.prefill import PrefillConfig
from evals.solvers.generic_solver import solver_with_prefill
from environments.hle.utils import grade_hle_answer

DATASET_PATH = "cais/hle"

# Default instructions for HLE
DEFAULT_INSTRUCTIONS = (
    "Answer the following multiple choice question. "
    "Think through the problem step by step, then provide your final answer as a single letter (A, B, C, or D). "
    "The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of the options."
)


def get_hle_dataset(split: str = "test", shuffle: bool = True) -> Dataset:
    """Load HLE dataset from HuggingFace, filtering for multiple choice without images.

    Args:
        split: Dataset split to use (default: "test", the only available split)
        shuffle: Whether to shuffle the dataset

    Returns:
        Dataset with HLE samples
    """
    dataset = hf_dataset(
        path=DATASET_PATH,
        split=split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )

    # Filter for multiple choice questions without images
    dataset = dataset.filter(
        name=f"{dataset.name}_mcq_no_images",
        predicate=lambda sample: (
            sample.metadata is not None
            and sample.metadata.get("answer_type") == "multipleChoice"
            and not sample.metadata.get("has_image", False)
        ),
    )

    return dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert HLE dataset record to Inspect Sample."""
    # Check if image field is non-empty
    has_image = bool(record.get("image", ""))

    return Sample(
        id=record["id"],
        input=record["question"],
        target=record["answer"],
        metadata={
            "answer_type": record["answer_type"],
            "category": record.get("category"),
            "raw_subject": record.get("raw_subject"),
            "has_image": has_image,
        },
    )


def record_to_sample_prefill(record: dict[str, Any]) -> Sample:
    """Convert prefill JSONL record to Inspect Sample.

    Used when loading dataset from prefill file.

    Args:
        record: Dictionary with keys: id, question, response, target

    Returns:
        Sample with question as input and response preserved in metadata
    """
    return Sample(
        id=record.get("id"),
        input=record["question"],
        target=record["target"],
        metadata={
            "answer_type": record.get("answer_type"),
            "category": record.get("category"),
            "raw_subject": record.get("raw_subject"),
            "response": record.get("response"),  # Store for prefill reference
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def hle_scorer() -> Scorer:
    """Score HLE multiple choice answers using letter extraction and comparison."""
    async def score(state: TaskState, target: Target) -> Score:
        correct = await grade_hle_answer(
            response=state.output.completion,
            target=target.text,
        )

        if correct:
            score = Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation="Correct answer",
            )
        else:
            score = Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation="Incorrect answer",
            )

        return score

    return score


@task
def hle(
    split: str = "test",
    prefill_config: PrefillConfig | None = None,
    timeout: int | None = None,
) -> Task:
    """
    Inspect Task implementation for the HLE benchmark (multiple choice only).

    Args:
        split: Dataset split to use (default: "test", the only available split)
        prefill_config: PrefillConfig for eval-time hints (optional)
        timeout: Timeout in seconds for generation (default: None)
    """
    # When using prefill data, load directly from the prefill JSONL file
    # This automatically filters to only tasks with pre-fills available
    if prefill_config:
        dataset = json_dataset(
            json_file=prefill_config.path,
            sample_fields=record_to_sample_prefill,
        )
    else:
        dataset = get_hle_dataset(split=split, shuffle=True)

    # Use generic solver with prefill support
    solver = solver_with_prefill(
        instruction_template=DEFAULT_INSTRUCTIONS,
        example_template=None,  # No example template - questions are standalone
        prefill_config=prefill_config,
        timeout=timeout,
    )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=hle_scorer(),
        config=GenerateConfig(temperature=1.0),
    )
