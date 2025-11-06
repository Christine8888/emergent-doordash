"""
ARC-AGI: Abstraction and Reasoning Corpus

Dataset: fchollet/ARC-AGI
https://github.com/fchollet/ARC-AGI
"""
import json
import os
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample, json_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Solver, TaskState

from environments.arc.utils import construct_arc_prompt, format_grid, extract_answer, grade_answer

ARC_DATA_PATH = Path(__file__).parent / "ARC-AGI" / "data"


def get_arc_dataset(split: str = "training", shuffle: bool = True, test_case_seed: int = 42, sample_ids: set[str] | None = None) -> Dataset:
    """Load ARC dataset from local JSON files.

    For tasks with multiple test cases, randomly selects one using the seed.

    Args:
        split: Either "training" or "evaluation"
        shuffle: Whether to shuffle the dataset
        test_case_seed: Seed for selecting test case when multiple exist (default: 42)
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)

    Returns:
        Dataset with ARC samples (one per task)
    """
    import random

    data_dir = ARC_DATA_PATH / split
    samples = []
    rng = random.Random(test_case_seed)

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        task_id = json_file.stem

        # Skip if filtering and this task is not in the filter set
        if sample_ids is not None and task_id not in sample_ids:
            continue

        train_examples = data["train"]

        # Select one test case (randomly if multiple, otherwise the only one)
        test_cases = data["test"]
        test_case = rng.choice(test_cases) if len(test_cases) > 1 else test_cases[0]

        sample = Sample(
            id=task_id,
            input=construct_arc_prompt(train_examples, test_case["input"]),
            target=format_grid(test_case["output"]),
            metadata={
                "task_id": task_id,
                "train_examples": train_examples,
                "test_input": test_case["input"],
                "test_output": test_case["output"],
            },
        )
        samples.append(sample)

    dataset = MemoryDataset(samples=samples, name=f"arc_{split}")

    if shuffle:
        dataset.shuffle()

    return dataset


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
            "task_id": record.get("task_id"),
            "test_idx": record.get("test_idx"),
            "response": record.get("response"),  # Store for prefill reference
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def arc_scorer() -> Scorer:
    """Score ARC answers using exact match after normalization.

    Normalizes by removing all whitespace, punctuation, and letters,
    keeping only digits for comparison.
    """
    async def score(state: TaskState, target: Target) -> Score:
        extracted_answer = extract_answer(state.output.completion)

        correct = await grade_answer(
            extracted_answer=extracted_answer,
            target=target.text,
        )

        if correct:
            score = Score(
                value=CORRECT,
                answer=extracted_answer,
                explanation="Correct grid prediction",
            )
        else:
            score = Score(
                value=INCORRECT,
                answer=extracted_answer,
                explanation="Incorrect grid prediction",
            )

        return score

    return score


@task
def arc(
    split: str = "training",
    test_case_seed: int = 42,
    sample_ids: set[str] | None = None,
    solver: Solver | list[Solver] | None = None,
) -> Task:
    """
    Baseline ARC-AGI task.

    This is the minimal task definition. For custom configurations (prefill, few-shot, etc.),
    use solver composition in your experiment file.

    Args:
        split: Dataset split to use ("training" or "evaluation")
        test_case_seed: Seed for selecting test case when multiple exist (default: 42)
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
        solver: Custom solver or list of solvers. If None, uses basic generate().

    Returns:
        Task configured for ARC evaluation
    """
    # Load dataset
    dataset = get_arc_dataset(
        split=split,
        shuffle=True,
        test_case_seed=test_case_seed,
        sample_ids=sample_ids
    )

    # Use provided solver or create basic one
    # Note: ARC prompts are already fully constructed, so no instructions needed
    if solver is None:
        from inspect_ai.solver import generate
        solver = generate()

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=arc_scorer(),
        config=GenerateConfig(temperature=1.0),
    )
