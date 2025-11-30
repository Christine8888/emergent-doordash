"""Modular solver components for composing evaluation pipelines.

These components can be composed in any order:
    solver=[instructions(...), fewshot(...), prefill(...), generate()]

Or:
    solver=[fewshot(...), prefill(...), instructions(...), generate()]
"""

import logging
import random
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, solver
from inspect_ai.solver import TaskState

from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig, format_fewshot_examples

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
    example_template: str = "{question}\n{response}",
) -> Solver:
    """Add few-shot examples to the user prompt.

    **How it works:**
    This solver APPENDS examples to the end of the current prompt text.
    Always excludes the current problem from few-shot selection.

    **Execution order matters:**
    - [instructions(), fewshot()] → [Instructions][Problem][Examples]
    - [fewshot(), instructions()] → [Examples][Instructions][Problem] ← weird!

    **Typical usage:**
    Put fewshot() AFTER instructions() to get natural order:
        solver = [
            instructions("Solve the problem."),
            fewshot(FewShotConfig(
                path="hints.jsonl",
                num_examples=3,
                prefix="Here are some examples:",
                suffix="Now solve:"
            )),
            prefill(config),
            generate()
        ]

    Args:
        config: FewShotConfig with path, num_examples, seed, prefix, suffix
        example_template: Format string with {question} and {response} (default: "{question}\\n{response}")

    Returns:
        Solver that appends few-shot examples

    Example final prompt structure:
        [Instructions from instructions()]

        [Current problem]

        [config.prefix if provided]

        [Example 1]
        [Example 2]

        [config.suffix if provided]
    """
    fewshot_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        examples_text = format_fewshot_examples(
            fewshot_data=fewshot_data,
            n_examples=config.num_examples,
            example_template=example_template,
            current_id=state.sample_id if config.exclude_current else None,
            seed=config.seed,
            prefix=config.prefix,
            suffix=config.suffix,
        )

        # APPEND examples to current prompt (not prepend!)
        if examples_text:
            state.user_prompt.text = state.user_prompt.text + "\n\n" + examples_text

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
    prefill_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if config.fraction > 0.0:
            if state.sample_id not in prefill_data:
                raise KeyError(
                    f"Sample '{state.sample_id}' not found in prefill data. "
                    f"Available samples should be filtered using config.get_available_ids()"
                )
            samples = prefill_data[state.sample_id]
            rng = random.Random(state.epoch)
            prefill_text = rng.choice(list(samples.values()))
            state.messages.append(ChatMessageAssistant(content=prefill_text))

        return state

    return solve


@solver
def intext(config: PrefillConfig, prefix: str = "Here is part of a hint that may be helpful to your solution:\n") -> Solver:
    """Add hint text inline to the user prompt.

    Similar to prefill() but appends hint text to the user prompt instead of
    adding an assistant message. The hint text is prefixed with a customizable
    prefix string.

    Args:
        config: PrefillConfig with path, fraction, mode, and mask_token settings
        prefix: Text to prepend to the hint (default: "Here is part of a hint that may be helpful to your solution:\\n")

    Returns:
        Solver that appends hint text to user prompt for the current sample

    Raises:
        KeyError: If sample_id is not in prefill data (when fraction > 0.0)
    """
    hint_data = config.get_data()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if config.fraction > 0.0:
            if state.sample_id not in hint_data:
                raise KeyError(
                    f"Sample '{state.sample_id}' not found in hint data. "
                    f"Available samples should be filtered using config.get_available_ids()"
                )
            samples = hint_data[state.sample_id]
            rng = random.Random(state.epoch)
            hint_text = rng.choice(list(samples.values()))
            state.user_prompt.text = state.user_prompt.text + "\n\n" + prefix + hint_text

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
