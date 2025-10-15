"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

from typing import Any
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset, json_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from evals.solvers.mcq_solver import multiple_choice_prefill
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig

# default epochs to run eval for
DEFAULT_EPOCHS = 1
LOCAL_DATA_DIR = Path(__file__).parent / "data"


def get_gpqa_dataset():
    dataset = csv_dataset(
            csv_file=str(LOCAL_DATA_DIR / "gpqa_diamond.csv"),
            sample_fields=record_to_sample,
            shuffle_choices=True,
    )
    return dataset
@task
def gpqa_diamond(
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
    prefill_config: PrefillConfig | None = None,
    timeout: int | None = 600,
) -> Task:
    """GPQA Diamond evaluation task.

    Args:
        instruction_template: Custom instruction template (overrides default)
        example_template: Custom example template (overrides default)
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig object for vLLM prefill (optional)
        timeout: Timeout in seconds for generation (default: 600)
    """
    solver = multiple_choice_prefill(
        instruction_template=instruction_template,
        example_template=example_template,
        fewshot_config=fewshot_config,
        prefill_config=prefill_config,
        timeout=timeout
    )

    # Create dataset
    # When using prefill data, load directly from the prefill JSONL file to preserve
    # the (question, response, target, choices) tuple that cannot be separated
    if prefill_config:
        dataset = json_dataset(
            json_file=prefill_config.path,
            sample_fields=record_to_sample_prefill,
        )
    else:
        dataset = get_gpqa_dataset()

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=choice(),
        epochs=DEFAULT_EPOCHS,
    )


# map CSV records to inspect samples (note that target is always "A" in the
# dataset, we will shuffle the presentation of options to mitigate this)
def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )


# map prefill JSONL records to inspect samples
# The (question, target, choices) tuple is preserved from the prefill data
def record_to_sample_prefill(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],  # Just the question without choices - choices are in question_with_choices
        choices=record["choices"],  # Choices in the correct order from prefill data
        target=record["target"],    # Target letter that matches the choice ordering
        id=record["id"],
    )