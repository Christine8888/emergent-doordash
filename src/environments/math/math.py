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
from src.evals.prefill import PrefillConfig
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import (
    Solver,
    TaskState,
    generate,
    prompt_template,
    system_message,
)

from inspect_evals.math.utils import (
    MathLevel,
    MathSubject,
    filter_dataset,
    record_to_sample,
    sample_to_fewshot,
    score_helper,
)

DATASET_PATH = "DigitalLearningGmbH/MATH-lighteval"

# Few-shot prompt template partially based on https://arxiv.org/pdf/2206.14858 - Appendix D.2
SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
You will be asked to solve a math problem. Some examples of problems and solutions are provided below.

{examples}
""".strip()

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
        prefill_config: PrefillConfig object for vLLM prefill (optional)
    """
    dataset = hf_dataset(
        path=DATASET_PATH,
        split=split,
        name="default",
        sample_fields=record_to_sample,
        auto_id=True,
        shuffle=True,
    )
    # Subset the data based on levels and/or subjects
    dataset = filter_dataset(dataset=dataset, levels=levels, subjects=subjects)

    prompt_tmpl = template if template else USER_PROMPT_TEMPLATE

    return Task(
        dataset=dataset,
        solver=math_solver(
            fewshot=fewshot,
            fewshot_seed=fewshot_seed,
            template=prompt_tmpl,
            prefill_config=prefill_config,
        ),
        scorer=[
            expression_exact_match(),
            expression_exact_match_sympy(),
        ],
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


def math_solver(
    fewshot: int,
    fewshot_seed: int,
    template: str,
    prefill_config: "PrefillConfig | None" = None,
) -> list[Solver]:
    """Build solver for MATH task.

    Arguments:
        fewshot (int): Number of few shot examples to use.
        fewshot_seed (int): Random seed for sampling few shot examples.
        template (str): Prompt template to use.
        prefill_config: PrefillConfig object for vLLM prefill (optional).
    """
    solver = [prompt_template(template)]

    # Add prefill if config provided
    if prefill_config:
        from src.evals.prefill import prefill
        solver.append(prefill(prefill_config))

    solver.append(generate())

    if fewshot:
        fewshot_samples = hf_dataset(
            path=DATASET_PATH,
            split="train",
            trust=True,
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
                    examples="\n\n".join(
                        [sample_to_fewshot(sample=sample) for sample in fewshot_samples]
                    )
                )
            ),
        )

    return solver