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
from evals.fewshot import FewShotConfig, load_fewshot_samples, create_fewshot_system_message
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

DEFAULT_MATH_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

PROBLEM:
{prompt}

SOLUTION:
""".strip()


@solver
def math_solver(
    *,
    template: str | None = None,
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
    1. Optionally adds few-shot examples (different per task)
    2. Formats the math prompt
    3. Optionally adds a prefill assistant message
    4. Calls generate() with continue_final_message=True if prefill was added

    Args:
        template: Template for the question (defaults to DEFAULT_MATH_TEMPLATE)
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: PrefillConfig for eval-time hints (test_hints.jsonl)
        local_dataset_dir: Path to local dataset directory (deprecated)
        record_to_sample: Function to convert records to samples (deprecated)
        sample_to_fewshot: Function to convert samples to fewshot strings (deprecated)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    if template is None:
        template = DEFAULT_MATH_TEMPLATE

    template_str = resource(template)

    prefill_map = {}
    if prefill_config:
        prefill_map = load_prefill_map(prefill_config)

    all_fewshot_samples = []
    if fewshot_config:
        raw_samples = load_fewshot_samples(fewshot_config)
        for sample in raw_samples:
            formatted_sample = sample.copy()
            question = sample.get('question', '')
            # No special formatting for math problems
            formatted_example = f"{question}"
            formatted_sample['formatted_example'] = formatted_example
            all_fewshot_samples.append(formatted_sample)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Construct main prompt
        formatted_question = str(template_str).format(prompt=state.user_prompt.text)

        # Handle few-shot prompting if desired
        if fewshot_config and all_fewshot_samples:
            seed = state.sample_id if state.sample_id else None

            # Get few-shot examples
            import random

            rng = random.Random(hash(seed) if isinstance(seed, str) else seed)
            available_samples = all_fewshot_samples
            if fewshot_config.exclude_current and state.sample_id is not None:
                available_samples = [s for s in all_fewshot_samples
                                    if s.get(fewshot_config.id_field) != state.sample_id]

            selected_samples = rng.sample(available_samples,
                                         min(fewshot_config.num_examples, len(available_samples)))

            # Format examples
            examples_text = []
            for sample in selected_samples:
                try:
                    example = fewshot_config.example_template.format(**sample)
                    examples_text.append(example)
                except KeyError as e:
                    logger.warning(f"Missing field {e} in sample {sample.get(fewshot_config.id_field, 'unknown')}")
                    continue

            # Build user message: system instruction + examples + current question
            system_instruction = fewshot_config.system_template.replace("{examples}", "").strip()
            user_content = system_instruction + "\n\n" + "\n\n".join(examples_text) + "\n\n" + formatted_question
            state.user_prompt.text = user_content
        else:
            state.user_prompt.text = formatted_question

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
