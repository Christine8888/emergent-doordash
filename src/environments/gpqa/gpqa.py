"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

from typing import Any
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from evals.solvers import multiple_choice_prefill
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig

# default epochs to run eval for
DEFAULT_EPOCHS = 1
LOCAL_DATA_DIR = Path(__file__).parent / "data"


@task
def gpqa_diamond(
    template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
    prefill_config: PrefillConfig | None = None,
    timeout: int | None = 600,
) -> Task:
    """GPQA Diamond evaluation task.

    Args:
        template: Custom prompt template
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig object for vLLM prefill (optional)
        timeout: Timeout in seconds for generation (default: 600)
    """
    solver = multiple_choice_prefill(
        template=template,
        fewshot_config=fewshot_config,
        prefill_config=prefill_config,
        timeout=timeout
    )

    return Task(
        dataset=csv_dataset(
            csv_file=str(LOCAL_DATA_DIR / "gpqa_diamond.csv"),
            sample_fields=record_to_sample,
            shuffle_choices=True,
        ),
        solver=solver,
        scorer=choice(),
        epochs=DEFAULT_EPOCHS,
    )


# map records to inspect samples (note that target is always "A" in the,
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