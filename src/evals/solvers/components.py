"""Modular solver components for composing evaluation pipelines.

These components can be composed in any order:
    solver=[instructions(...), fewshot(...), prefill(...), generate()]

Or:
    solver=[fewshot(...), prefill(...), instructions(...), generate()]
"""

import logging
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, solver
from inspect_ai.solver import TaskState

from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig, create_fewshot_message

logger = logging.getLogger(__name__)


@solver
def instructions(template: str) -> Solver:
    """Add instructions to the beginning of the user prompt.

    Args:
        template: Instruction text to prepend

    Returns:
        Solver that adds instructions
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = template + "\n\n" + state.user_prompt.text
        return state

    return solve


@solver
def fewshot(
    config: FewShotConfig,
    example_template: str = "{question}\n{response}"
) -> Solver:
    """Add few-shot examples to the user prompt.

    Args:
        config: FewShotConfig with path and sampling settings
        example_template: Template for formatting examples (default: "{question}\\n{response}")

    Returns:
        Solver that adds few-shot examples
    """
    # Load fewshot data
    fewshot_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get current task text
        current_task = state.user_prompt.text

        # Create fewshot message
        user_content = create_fewshot_message(
            fewshot_data=fewshot_data,
            config=config,
            instruction_template="",  # No instructions here, use instructions() solver
            example_template=example_template,
            current_task=current_task,
            current_id=state.sample_id,
            seed=state.sample_id,
        )

        state.user_prompt.text = user_content
        return state

    return solve


@solver
def prefill(config: PrefillConfig) -> Solver:
    """Add prefill text as an assistant message.

    Args:
        config: PrefillConfig with path and fraction settings

    Returns:
        Solver that adds prefill for the current sample

    Raises:
        KeyError: If sample_id is not in prefill data (when fraction > 0.0)
    """
    # Load prefill data (sample_id -> prefill_text)
    prefill_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Require sample to exist in prefill data
        if state.sample_id not in prefill_data:
            raise KeyError(
                f"Sample '{state.sample_id}' not found in prefill data. "
                f"Available samples should be filtered using config.get_available_ids()"
            )

        if config.fraction > 0.0:
            # Add prefill as assistant message
            prefill_text = prefill_data[state.sample_id]
            state.messages.append(ChatMessageAssistant(content=prefill_text))

        return state

    return solve


@solver
def system_message(message: str) -> Solver:
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


@solver
def generate(
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
    async def solve(state: TaskState, gen: Generate) -> TaskState:
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
        state = await gen(state, config=gen_config)

        return state

    return solve
