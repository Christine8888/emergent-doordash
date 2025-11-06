"""
Measuring Mathematical Problem Solving With the MATH Dataset

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora,
Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874

Based on:
1. https://github.com/openai/simple-evals/blob/main/math_eval.py
2. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math
3. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math
"""
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Solver, TaskState
from pathlib import Path

from environments.math.utils import (
    MathLevel,
    MathSubject,
    filter_dataset,
    record_to_sample,
    score_helper,
)

DATASET_PATH = "DigitalLearningGmbH/MATH-lighteval"
LOCAL_DATASET_DIR = Path(__file__).parent / "data"

# Default templates for math problems
DEFAULT_INSTRUCTIONS = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.
""".strip()

DEFAULT_EXAMPLE_TEMPLATE = """
PROBLEM:
{question}

SOLUTION:
{solution}
""".strip()


def get_math_dataset(
    split: str = "test",
    levels: list[MathLevel] | MathLevel = [],
    subjects: list[MathSubject] | MathSubject = [],
    shuffle: bool = True,
    sample_ids: set[str] | None = None,
):
    """
    Load MATH dataset from local JSONL files.

    Args:
        split: Dataset split to use ("test", "train", or "validation")
        levels: List of levels to filter on, 1 to 5
        subjects: List of subjects to filter on
        shuffle: Whether to shuffle the dataset
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)

    Returns:
        Inspect Dataset object
    """
    local_file = LOCAL_DATASET_DIR / f"math_{split}.jsonl"
    dataset = json_dataset(
        json_file=str(local_file),
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )
    # Subset the data based on levels and/or subjects
    dataset = filter_dataset(dataset=dataset, levels=levels, subjects=subjects)

    # Filter to specific sample IDs if provided
    if sample_ids is not None:
        dataset = dataset.filter(
            name=f"{dataset.name}_filtered",
            predicate=lambda sample: sample.id in sample_ids
        )

    return dataset


@task
def math(
    levels: list[MathLevel] | MathLevel = [],
    subjects: list[MathSubject] | MathSubject = [],
    split: str = "test",
    sample_ids: set[str] | None = None,
    solver: Solver | list[Solver] | None = None,
) -> Task:
    """
    Baseline MATH task.

    This is the minimal task definition. For custom configurations (prefill, few-shot, etc.),
    use solver composition in your experiment file.

    Args:
        levels: List of levels to filter on, 1 to 5
        subjects: List of subjects to filter on
        split: Dataset split to use ("test", "train", or "validation")
        sample_ids: Optional set of sample IDs to filter to (default: None = use all)
        solver: Custom solver or list of solvers. If None, uses basic generate() with instructions.

    Returns:
        Task configured for MATH evaluation
    """
    # Load dataset
    dataset = get_math_dataset(
        split=split,
        levels=levels,
        subjects=subjects,
        shuffle=True,
        sample_ids=sample_ids
    )

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
        scorer=expression_exact_match_sympy(),
        config=GenerateConfig(temperature=1.0),
    )

# Exact match using an LLM GRADER!!!! 
@scorer(metrics=[accuracy(), stderr()])
def expression_equivalance(model: str | Model | None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            model=model,
            exact_match=False,
        )

    return score


# Exact match using sympy based on: https://arxiv.org/pdf/2206.14858
@scorer(metrics=[accuracy(), stderr()])
def expression_exact_match_sympy() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            exact_match=True,
            use_sympy=True,
        )

    return score


# Exact match based on:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L36
@scorer(metrics=[accuracy(), stderr()])
def expression_exact_match() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            exact_match=True,
            use_sympy=False,
        )

    return score