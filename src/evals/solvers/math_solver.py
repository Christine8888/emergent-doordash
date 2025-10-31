"""Math solver with optional prefill support for vLLM continuation."""

from pathlib import Path

from inspect_ai.solver import Solver, solver

from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig
from evals.solvers.generic_solver import solver_with_prefill

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


@solver
def math_solver(
    *,
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
    prefill_config: PrefillConfig | None = None,
    local_dataset_dir: Path | None = None,
    record_to_sample=None,
    sample_to_fewshot=None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Math solver with optional prefill support.

    This is a thin wrapper around solver_with_prefill() that provides
    math-specific default templates.

    Args:
        instruction_template: Custom instruction template (overrides default).
        example_template: Custom example template (overrides default).
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig for eval-time hints
        local_dataset_dir: Path to local dataset directory (deprecated)
        record_to_sample: Function to convert records to samples (deprecated)
        sample_to_fewshot: Function to convert samples to fewshot strings (deprecated)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    return solver_with_prefill(
        instruction_template=instruction_template or DEFAULT_INSTRUCTIONS,
        example_template=example_template or DEFAULT_EXAMPLE_TEMPLATE,
        fewshot_config=fewshot_config,
        prefill_config=prefill_config,
        max_tokens=max_tokens,
        timeout=timeout,
    )
