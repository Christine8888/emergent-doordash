"""Math solver with optional prefill support for vLLM continuation."""

import logging
from pathlib import Path

from inspect_ai.dataset import json_dataset
from inspect_ai.model import ChatMessageAssistant, GenerateConfig
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
    system_message,
)
from inspect_ai.util import resource

from evals.prefill import PrefillConfig, load_prefill_map
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

# Few-shot prompt template
SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
You will be asked to solve a math problem. Some examples of problems and solutions are provided below.

{examples}
""".strip()


@solver
def math_solver(
    *,
    template: str,
    fewshot: int = 0,
    fewshot_seed: int = 42,
    prefill_config: PrefillConfig | None = None,
    local_dataset_dir: Path | None = None,
    record_to_sample=None,
    sample_to_fewshot=None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Math solver with optional prefill support.

    This solver:
    1. Optionally adds few-shot examples
    2. Formats the math prompt
    3. Optionally adds a prefill assistant message
    4. Calls generate() with continue_final_message=True if prefill was added

    Args:
        template: Template for the question
        fewshot: Number of few shot examples to use
        fewshot_seed: Random seed for sampling few shot examples
        prefill_config: Optional prefill configuration
        local_dataset_dir: Path to local dataset directory (for fewshot examples)
        record_to_sample: Function to convert records to samples (for fewshot)
        sample_to_fewshot: Function to convert samples to fewshot strings (for fewshot)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    template_str = resource(template)

    # Load prefill map if config provided
    prefill_map = {}
    if prefill_config:
        prefill_map = load_prefill_map(prefill_config)

    # Load few-shot examples if requested
    fewshot_system_message = None
    if fewshot:
        if not all([local_dataset_dir, record_to_sample, sample_to_fewshot]):
            raise ValueError(
                "local_dataset_dir, record_to_sample, and sample_to_fewshot "
                "must be provided when using fewshot"
            )

        # Load fewshot examples from local file
        local_train_file = local_dataset_dir / "math_train.jsonl"
        fewshot_samples = json_dataset(
            json_file=str(local_train_file),
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        fewshot_system_message = SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
            examples="\n\n".join(
                [sample_to_fewshot(sample=sample) for sample in fewshot_samples]
            )
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Add few-shot system message if available
        if fewshot_system_message:
            state.messages.insert(0, system_message(fewshot_system_message)(state, generate))

        # Format the prompt using the template
        state.user_prompt.text = str(template_str).format(prompt=state.user_prompt.text)

        # Add prefill if available
        continue_message = False
        if prefill_config and state.sample_id in prefill_map:
            full_response = prefill_map[state.sample_id]
            prefill_text = get_prefill_fraction(
                full_response,
                fraction=prefill_config.fraction
            )

            if prefill_text:
                logger.info(
                    f"Sample {state.sample_id}: Adding prefill "
                    f"({len(prefill_text.split())} words, {prefill_config.fraction:.1%})"
                )

                # Add assistant message with prefill
                state.messages.append(ChatMessageAssistant(content=prefill_text))
                continue_message = True

        # Generate with or without continuation
        gen_config = GenerateConfig(
            max_tokens=max_tokens,
            continue_final_message=continue_message,
            timeout=timeout
        )
        state = await generate(state, config=gen_config)

        return state

    return solve
