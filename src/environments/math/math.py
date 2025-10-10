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
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig
from evals.solvers import math_solver
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from pathlib import Path

from environments.math.utils import (
    MathLevel,
    MathSubject,
    filter_dataset,
    record_to_sample,
    sample_to_fewshot,
    score_helper,
)

DATASET_PATH = "DigitalLearningGmbH/MATH-lighteval"
LOCAL_DATASET_DIR = Path(__file__).parent / "data"


@task
def math(
    levels: list[MathLevel] | MathLevel = [],
    subjects: list[MathSubject] | MathSubject = [],
    split: str = "test",
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: "FewShotConfig | None" = None,
    prefill_config: "PrefillConfig | None" = None,
    timeout: int | None = None,
) -> Task:
    """
    Inspect Task implementation for the MATH benchmark

    Args:
        levels (list[MathLevel]): List of levels to filter on, 1 to 5.
        subjects (list[MathSubject]): List of subjects to filter on.
        split (str): Dataset split to use ("test", "train", or "validation")
        instruction_template: Custom instruction template (overrides default)
        example_template: Custom example template (overrides default)
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig object for eval-time hints (test_hints.jsonl)
        timeout: Timeout in seconds for generation (default: None)
    """
    # Load from local JSONL file
    local_file = LOCAL_DATASET_DIR / f"math_{split}.jsonl"
    dataset = json_dataset(
        json_file=str(local_file),
        sample_fields=record_to_sample,
        shuffle=True,
    )
    # Subset the data based on levels and/or subjects
    dataset = filter_dataset(dataset=dataset, levels=levels, subjects=subjects)

    return Task(
        dataset=dataset,
        solver=math_solver(
            instruction_template=instruction_template,
            example_template=example_template,
            fewshot_config=fewshot_config,
            prefill_config=prefill_config,
            timeout=timeout,
            local_dataset_dir=LOCAL_DATASET_DIR,
            record_to_sample=record_to_sample,
            sample_to_fewshot=sample_to_fewshot,
        ),
        scorer=expression_exact_match_sympy(),
        config=GenerateConfig(temperature=0.5),
    )


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