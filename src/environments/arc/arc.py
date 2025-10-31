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
from inspect_ai.solver import TaskState, generate

from evals.prefill import PrefillConfig
from evals.solvers.generic_solver import solver_with_prefill
from environments.arc.utils import construct_arc_prompt, format_grid, grade_arc_answer

ARC_DATA_PATH = Path(__file__).parent / "ARC-AGI" / "data"


def get_arc_dataset(split: str = "training", shuffle: bool = True, test_case_seed: int = 42) -> Dataset:
    """Load ARC dataset from local JSON files.

    For tasks with multiple test cases, randomly selects one using the seed.

    Args:
        split: Either "training" or "evaluation"
        shuffle: Whether to shuffle the dataset
        test_case_seed: Seed for selecting test case when multiple exist (default: 42)

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
        correct = await grade_arc_answer(
            response=state.output.completion,
            target=target.text,
        )

        if correct:
            score = Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation="Correct grid prediction",
            )
        else:
            score = Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation="Incorrect grid prediction",
            )

        return score

    return score


@task
def arc(
    split: str = "training",
    prefill_config: PrefillConfig | None = None,
    test_case_seed: int = 42,
    timeout: int | None = None,
) -> Task:
    """
    Inspect Task implementation for the ARC-AGI benchmark.

    Args:
        split: Dataset split to use ("training" or "evaluation")
        prefill_config: PrefillConfig for eval-time hints (optional)
        test_case_seed: Seed for selecting test case when multiple exist (default: 42)
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
        dataset = get_arc_dataset(split=split, shuffle=True, test_case_seed=test_case_seed)

    # Use generic solver with prefill support (no templates needed for ARC)
    solver = solver_with_prefill(
        instruction_template=None,  # ARC prompts are already fully constructed
        example_template=None,
        prefill_config=prefill_config,
        timeout=timeout,
    )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=arc_scorer(),
        config=GenerateConfig(temperature=1.0),
    )
