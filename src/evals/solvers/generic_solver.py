"""Generic solver with optional prefill support for vLLM continuation."""

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

from evals.example import Example
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig, create_fewshot_message
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)


@solver
def solver_with_prefill(
    *,
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
    prefill_config: PrefillConfig | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Generic solver with optional prefill and few-shot support.

    This solver:
    1. Formats the prompt using templates (if provided)
    2. Optionally adds few-shot examples with the same format
    3. Optionally adds a prefill assistant message
    4. Calls generate() with continue_final_message=True if prefill was added

    Args:
        instruction_template: Instructions for the task. If None, no instructions added.
        example_template: Template for formatting examples. Should have {question} and {solution} placeholders.
                         If None, uses the raw prompt as-is.
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig for eval-time hints
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """

    # Get cached prefill data if config provided
    prefill_data = {}
    if prefill_config:
        prefill_data = prefill_config.get_data()

    # Get cached few-shot data if config provided
    fewshot_data = {}
    if fewshot_config:
        fewshot_data = fewshot_config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Format current task
        if example_template:
            current_task = example_template.format(
                question=state.user_prompt.text,
                solution=""  # Empty - to be completed by the model
            )
        else:
            # No template - use raw prompt
            current_task = state.user_prompt.text

        # Handle few-shot prompting
        if fewshot_config and fewshot_data:
            user_content = create_fewshot_message(
                fewshot_data=fewshot_data,
                config=fewshot_config,
                instruction_template=instruction_template or "",
                example_template=example_template or "{question}\n{solution}",
                current_task=current_task,
                current_id=state.sample_id,
                seed=state.sample_id,
            )
            state.user_prompt.text = user_content
        else:
            # 0-shot: instructions + current task
            if instruction_template:
                user_content = instruction_template + "\n\n" + current_task
            else:
                user_content = current_task
            state.user_prompt.text = user_content

        # Handle prefill if available
        # Skip prefilling if fraction is 0.0 (useful for ablation studies)
        continue_message = False
        if state.sample_id in prefill_data and prefill_config.fraction > 0.0:
            full_response = prefill_data[state.sample_id].response
            prefill_text = get_prefill_fraction(
                full_response,
                fraction=prefill_config.fraction
            )
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
