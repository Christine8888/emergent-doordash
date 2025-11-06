"""Modular solver components for composing evaluation pipelines.

These components can be composed using Inspect's chain() function or list syntax:
    solver=[format_prompt(...), add_prefill(...), generate_with_continuation(...)]
"""

import logging
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, solver
from inspect_ai.solver import TaskState

from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig, create_fewshot_message

logger = logging.getLogger(__name__)


@solver
def format_prompt(
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
) -> Solver:
    """Format the prompt with instructions and optional few-shot examples.

    Args:
        instruction_template: Instructions to prepend to the task
        example_template: Template for formatting few-shot examples
        fewshot_config: Configuration for few-shot examples

    Returns:
        Solver that formats the prompt
    """
    # Load fewshot data if provided
    fewshot_data = fewshot_config.get_data() if fewshot_config else None

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get current task text
        current_task = state.user_prompt.text

        # Format with few-shot examples if provided
        if fewshot_config and fewshot_data:
            user_content = create_fewshot_message(
                fewshot_data=fewshot_data,
                config=fewshot_config,
                instruction_template=instruction_template or "",
                example_template=example_template or "{question}\n{solution}",
                current_task=current_task,
                current_id=state.sample_id,
                seed=state.sample_id,  # Use sample_id as seed for deterministic sampling
            )
            state.user_prompt.text = user_content
        else:
            # Just add instructions if no few-shot
            if instruction_template:
                state.user_prompt.text = instruction_template + "\n\n" + current_task

        return state

    return solve


@solver
def add_prefill(prefill_config: PrefillConfig) -> Solver:
    """Add prefill text as an assistant message.

    Args:
        prefill_config: Configuration for prefill (path, fraction)

    Returns:
        Solver that adds prefill for the current sample

    Raises:
        KeyError: If sample_id is not in prefill data (when fraction > 0.0)
    """
    # Load prefill data (sample_id -> prefill_text)
    prefill_data = prefill_config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Require sample to exist in prefill data
        if state.sample_id not in prefill_data:
            raise KeyError(
                f"Sample '{state.sample_id}' not found in prefill data. "
                f"Available samples should be filtered using prefill_config.get_available_ids()"
            )
        
        if prefill_config.fraction > 0.0:
            # Add prefill as assistant message
            prefill_text = prefill_data[state.sample_id]
            state.messages.append(ChatMessageAssistant(content=prefill_text))

        return state

    return solve


@solver
def generate_with_continuation(
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Generate with automatic continuation detection.

    If the last message is an assistant message, enables continue_final_message
    for vLLM to continue from that message.

    Args:
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds

    Returns:
        Solver that generates with appropriate configuration
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Auto-detect if we should continue from last message
        continue_message = (
            len(state.messages) > 0 and
            isinstance(state.messages[-1], ChatMessageAssistant)
        )

        # Configure generation
        gen_config = GenerateConfig(
            max_tokens=max_tokens,
            continue_final_message=continue_message,
            timeout=timeout
        )

        # Generate
        state = await generate(state, config=gen_config)

        return state

    return solve


@solver
def add_system_message(message: str) -> Solver:
    """Add a system message to the conversation.

    Args:
        message: System message content

    Returns:
        Solver that adds the system message
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content=message))
        return state

    return solve
