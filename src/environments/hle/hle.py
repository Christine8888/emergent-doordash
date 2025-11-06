"""
HLE: Humanity's Last Exam

Dataset: cais/hle
https://huggingface.co/datasets/cais/hle
"""
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Solver, TaskState

from environments.gpqa.utils import extract_answer, grade_answer

DATASET_PATH = "cais/hle"

# Default instructions for HLE
DEFAULT_INSTRUCTIONS = (
    "Answer the following multiple choice question. "
    "Think through the problem step by step, then provide your final answer as a single letter (A, B, C, or D). "
    "The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of the options."
)


def get_hle_dataset(split: str = "test", shuffle: bool = True, sample_ids: set[str] | None = None) -> Dataset:
    """Load HLE dataset from HuggingFace, filtering for multiple choice without images.

    Args:
        split: Dataset split to use (default: "test", the only available split)
        shuffle: Whether to shuffle the dataset
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)

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

    # Filter to specific sample IDs if provided
    if sample_ids is not None:
        dataset = dataset.filter(
            name=f"{dataset.name}_filtered",
            predicate=lambda sample: sample.id in sample_ids
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
        extracted_letter = extract_answer(state.output.completion, num_choices=4)

        correct = await grade_answer(
            extracted_letter=extracted_letter,
            target=target.text,
        )

        if correct:
            score = Score(
                value=CORRECT,
                answer=extracted_letter,
                explanation="Correct answer",
            )
        else:
            score = Score(
                value=INCORRECT,
                answer=extracted_letter,
                explanation="Incorrect answer",
            )

        return score

    return score


@task
def hle(
    split: str = "test",
    sample_ids: set[str] | None = None,
    solver: Solver | list[Solver] | None = None,
) -> Task:
    """
    Baseline HLE task (multiple choice only).

    This is the minimal task definition. For custom configurations (prefill, few-shot, etc.),
    use solver composition in your experiment file.

    Args:
        split: Dataset split to use (default: "test", the only available split)
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
        solver: Custom solver or list of solvers. If None, uses basic generate() with instructions.

    Returns:
        Task configured for HLE evaluation

    Example:
        # Baseline
        task = hle()

        # With prefill (in experiment file)
        from evals.solvers import format_prompt, add_prefill, generate_with_continuation
        from evals.prefill import PrefillConfig

        prefill_config = PrefillConfig(path="path/to/hints.jsonl", fraction=0.8)
        task = hle(
            sample_ids=prefill_config.get_available_ids(),
            solver=[
                format_prompt(instruction_template=DEFAULT_INSTRUCTIONS),
                add_prefill(prefill_config),
                generate_with_continuation(timeout=600)
            ]
        )
    """
    # Load dataset
    dataset = get_hle_dataset(split=split, shuffle=True, sample_ids=sample_ids)

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
        scorer=hle_scorer(),
        config=GenerateConfig(temperature=1.0),
    )
