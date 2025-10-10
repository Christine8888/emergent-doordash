"""Math solver with optional prefill support for vLLM continuation."""

import logging
from pathlib import Path

from inspect_ai.dataset import json_dataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.util import resource

from evals.prefill import PrefillConfig, load_prefill_map
from evals.fewshot import FewShotConfig, load_fewshot_samples, create_fewshot_message
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

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

    This solver:
    1. Formats the problem using DEFAULT_EXAMPLE_TEMPLATE (or custom template)
    2. Optionally adds few-shot examples with the same format
    3. Optionally adds a prefill assistant message
    4. Calls generate() with continue_final_message=True if prefill was added

    Args:
        instruction_template: Custom instruction template (overrides default).
                             Used for both 0-shot and few-shot prompts.
        example_template: Custom example template (overrides default).
                         Used for both 0-shot and few-shot examples.
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig for eval-time hints (test_hints.jsonl)
        local_dataset_dir: Path to local dataset directory (deprecated)
        record_to_sample: Function to convert records to samples (deprecated)
        sample_to_fewshot: Function to convert samples to fewshot strings (deprecated)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """

    prefill_map = {}
    if prefill_config:
        prefill_map = load_prefill_map(prefill_config)

    # Load few-shot samples
    all_fewshot_samples = []
    if fewshot_config:
        all_fewshot_samples = load_fewshot_samples(fewshot_config)

    if instruction_template is None:
        instruction_template = DEFAULT_INSTRUCTIONS
    if example_template is None:
        example_template = DEFAULT_EXAMPLE_TEMPLATE

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Format current task as an incomplete example
        current_task = example_template.format(
            question=state.user_prompt.text,
            solution=""  # Empty - to be completed by the model
        )

        # Handle few-shot prompting
        if fewshot_config and all_fewshot_samples:
            user_content = create_fewshot_message(
                all_samples=all_fewshot_samples,
                config=fewshot_config,
                instruction_template=instruction_template,
                example_template=example_template,
                current_task=current_task,
                current_id=state.sample_id,
                seed=state.sample_id,
            )
            state.user_prompt.text = user_content
        else:
            # 0-shot: just instructions + current task
            user_content = instruction_template + "\n\n" + current_task
            state.user_prompt.text = user_content

        # Handle prefill if available
        continue_message = False
        if prefill_config and state.sample_id in prefill_map:
            full_response = prefill_map[state.sample_id]
            prefill_text = get_prefill_fraction(
                full_response,
                fraction=prefill_config.fraction
            )

            if prefill_text:
                state.messages.append(ChatMessageAssistant(content=prefill_text))
                continue_message = True

        # Set generation parameters
        gen_config = GenerateConfig(
            max_tokens=max_tokens,
            continue_final_message=continue_message,
            timeout=timeout
        )
        state = await generate(state, config=gen_config)

        return state

    return solve
