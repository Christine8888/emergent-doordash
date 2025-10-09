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
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

# Few-shot prompt template
SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
You will be asked to solve a math problem. Some examples of problems and solutions are provided below.

{examples}
""".strip()


def create_fewshot_examples(
    train_samples: list[dict],
    num_examples: int,
    seed: int | str
) -> str:
    """Create few-shot examples from training samples.

    Args:
        train_samples: List of dicts with 'question', 'response', 'target'
        num_examples: Number of examples to include
        seed: Seed for random selection (can be int or string like sample_id)

    Returns:
        Formatted few-shot system message
    """
    import random

    rng = random.Random(hash(seed) if isinstance(seed, str) else seed)
    selected_samples = rng.sample(train_samples, min(num_examples, len(train_samples)))

    examples = []
    for sample in selected_samples:
        prob_str = f"PROBLEM:\n{sample['question']}"
        soln_str = f"SOLUTION:\n{sample['response']}"
        ans_str = f"ANSWER: {sample['target']}"
        example = f"{prob_str}\n\n{soln_str}\n{ans_str}"
        examples.append(example)

    return SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
        examples="\n\n".join(examples)
    )


@solver
def math_solver(
    *,
    template: str,
    fewshot: int = 0,
    fewshot_seed: int = 42,
    fewshot_config: PrefillConfig | None = None,
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
        template: Template for the question
        fewshot: Number of few shot examples to use
        fewshot_seed: Random seed for sampling few shot examples
        fewshot_config: PrefillConfig for few-shot solutions (train_hints.jsonl)
        prefill_config: PrefillConfig for eval-time hints (test_hints.jsonl)
        local_dataset_dir: Path to local dataset directory (for fewshot examples)
        record_to_sample: Function to convert records to samples (for fewshot)
        sample_to_fewshot: Function to convert samples to fewshot strings (for fewshot)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    template_str = resource(template)

    # Load prefill map if config provided (for eval-time hints)
    prefill_map = {}
    if prefill_config:
        prefill_map = load_prefill_map(prefill_config)

    # Load all train samples if using few-shot
    all_train_samples = []
    if fewshot:
        if fewshot_config:
            # Load from train_hints.jsonl with full solutions
            import json
            train_hints_file = Path(fewshot_config.path)
            with open(train_hints_file) as f:
                for line in f:
                    data = json.loads(line)
                    all_train_samples.append({
                        'id': data.get(fewshot_config.id_field),
                        'question': data.get('question'),
                        'response': data.get(fewshot_config.response_field),
                        'target': data.get('target')
                    })
        else:
            raise ValueError("fewshot_config must be provided when using fewshot")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Add few-shot system message if using few-shot (different per task)
        if fewshot and all_train_samples:
            seed = state.sample_id if state.sample_id else fewshot_seed
            fewshot_message = create_fewshot_examples(all_train_samples, fewshot, seed)
            state.messages.insert(0, ChatMessageSystem(content=fewshot_message))

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
