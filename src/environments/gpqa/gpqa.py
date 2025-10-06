"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

# default epochs to run eval for
DEFAULT_EPOCHS = 1


@task
def gpqa_diamond(template: str | None = None) -> Task:
    solver = [multiple_choice(template=template)] if template else [multiple_choice(cot=True)]
    return Task(
        dataset=csv_dataset(
            csv_file="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
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