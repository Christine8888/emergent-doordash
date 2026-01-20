"""
ARC-AGI: Abstraction and Reasoning Corpus

Dataset: fchollet/ARC-AGI
https://github.com/fchollet/ARC-AGI
"""
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Solver, TaskState

from environments.arc.utils import construct_arc_prompt, format_grid, extract_answer, grade_answer

ARC_DATA_PATH = Path(__file__).parent / "ARC-AGI" / "data"

# Default instructions for ARC
DEFAULT_INSTRUCTIONS = (
    "Find the common rule that maps an input grid to an output grid, given the examples below."
)


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
            )
        else:
            score = Score(
                value=INCORRECT,
                answer=extracted_answer,
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
    # Default solver: instructions then generate
    if solver is None:
        from evals.solvers import instructions, generate
        solver = [
            instructions(DEFAULT_INSTRUCTIONS),
            generate()
        ]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=arc_scorer(),
    )
