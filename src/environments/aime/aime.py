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
from inspect_ai.solver import TaskState

from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig
from evals.solvers import math_solver
from environments.math.utils import score_helper

DATASET_PATH = "di-zhang-fdu/AIME_1983_2024"


def get_aime_dataset(split: str = "train", shuffle: bool = True):
    """Load AIME dataset from HuggingFace."""
    return hf_dataset(
        path=DATASET_PATH,
        split=split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )


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


@task
def aime(
    split: str = "train",
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
    prefill_config: PrefillConfig | None = None,
    timeout: int | None = None,
) -> Task:
    """
    Inspect Task implementation for the AIME benchmark.

    Args:
        split: Dataset split to use (default: "train", the only available split)
        instruction_template: Custom instruction template (overrides default)
        example_template: Custom example template (overrides default)
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig object for eval-time hints
        timeout: Timeout in seconds for generation (default: None)
    """
    dataset = get_aime_dataset(split=split, shuffle=True)

    return Task(
        dataset=dataset,
        solver=math_solver(
            instruction_template=instruction_template,
            example_template=example_template,
            fewshot_config=fewshot_config,
            prefill_config=prefill_config,
            timeout=timeout,
        ),
        scorer=aime_scorer(),
        config=GenerateConfig(temperature=0.5),
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
