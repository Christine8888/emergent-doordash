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

# Setup for problem + instructions for providing answer
USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


@task
def math(
    levels: list[MathLevel] | MathLevel = [],
    subjects: list[MathSubject] | MathSubject = [],
    fewshot: int = 0,
    fewshot_seed: int = 42,
    template: str | None = None,
    split: str = "test",
    fewshot_config: "PrefillConfig | None" = None,
    prefill_config: "PrefillConfig | None" = None,
) -> Task:
    """
    Inspect Task implementation for the MATH benchmark

    Args:
        levels (list[MathLevel]): List of levels to filter on, 1 to 5.
        subjects (list[MathSubject]): List of subjects to filter on.
        fewshot (int): The number of fewshots to include
        fewshot_seed (int): The seed used when selecting fewshots
        template (str): Custom prompt template (must include {prompt} placeholder)
        split (str): Dataset split to use ("test", "train", or "validation")
        fewshot_config: PrefillConfig for few-shot solutions (train_hints.jsonl)
        prefill_config: PrefillConfig object for eval-time hints (test_hints.jsonl)
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

    prompt_tmpl = template if template else USER_PROMPT_TEMPLATE

    return Task(
        dataset=dataset,
        solver=math_solver(
            template=prompt_tmpl,
            fewshot=fewshot,
            fewshot_seed=fewshot_seed,
            fewshot_config=fewshot_config,
            prefill_config=prefill_config,
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