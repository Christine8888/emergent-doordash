"""
AIME: American Invitational Mathematics Examination (1983-2024)

Dataset: di-zhang-fdu/AIME_1983_2024
https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024
"""
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Solver, TaskState

from environments.math.math import DEFAULT_INSTRUCTIONS
from environments.math.utils import score_helper

DATASET_PATH = "di-zhang-fdu/AIME_1983_2024"


def get_aime_dataset(split: str = "train", shuffle: bool = True, sample_ids: set[str] | None = None):
    """Load AIME dataset from HuggingFace.

    Args:
        split: Dataset split to use
        shuffle: Whether to shuffle the dataset
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
    """
    dataset = hf_dataset(
        path=DATASET_PATH,
        split=split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )

    # Filter to specific sample IDs if provided
    if sample_ids is not None:
        dataset = dataset.filter(
            name=f"{dataset.name}",
            predicate=lambda sample: sample.id in sample_ids
        )

    return dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert AIME dataset record to Inspect Sample."""
    return Sample(
        id=record["ID"],
        input=record["Question"],
        target=str(record["Answer"]),
        metadata={
            "year": record["Year"],
            "problem_number": record["Problem Number"],
        },
    )


def record_to_sample_prefill(record: dict[str, Any]) -> Sample:
    """Map prefill JSONL records to inspect samples.

    This is used when loading the dataset directly from the prefill file,
    which automatically filters to only tasks with pre-fills available.

    Args:
        record: Dictionary with keys: id, question, response, target

    Returns:
        Sample with the question as input and response preserved in metadata
    """
    return Sample(
        id=record.get("id"),
        input=record["question"],
        target=str(record["target"]),
        metadata={
            "year": record.get("year"),
            "problem_number": record.get("problem_number"),
            "solution": record.get("response"),  # Store the full response for reference
        },
    )


@task
def aime(
    split: str = "train",
    sample_ids: set[str] | None = None,
    solver: Solver | list[Solver] | None = None,
) -> Task:
    """
    Baseline AIME task.

    This is the minimal task definition. For custom configurations (prefill, few-shot, etc.),
    use solver composition in your experiment file.

    Args:
        split: Dataset split to use (default: "train", the only available split)
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
        solver: Custom solver or list of solvers. If None, uses basic generate() with instructions.

    Returns:
        Task configured for AIME evaluation
    """
    # Load dataset
    dataset = get_aime_dataset(split=split, shuffle=True, sample_ids=sample_ids)

    # Use provided solver or create basic one
    if solver is None:
        from evals.solvers import instructions, generate
        solver = [
            instructions(DEFAULT_INSTRUCTIONS),
            generate()
        ]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=aime_scorer(),
        config=GenerateConfig(temperature=1.0),
    )


@scorer(metrics=[accuracy(), stderr()])
def aime_scorer() -> Scorer:
    """
    Score AIME answers using exact match with sympy.

    AIME answers are integers 0-999, so we use sympy-based exact matching
    from the MATH benchmark's scoring utilities.
    """
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            exact_match=True,
            use_sympy=True,
        )

    return score
