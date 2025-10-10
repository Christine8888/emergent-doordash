"""Multiple choice solver with prefill support for vLLM continuation."""

import logging
import re

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.util import resource

from evals.prefill import PrefillConfig, load_prefill_map
from evals.fewshot import FewShotConfig, load_fewshot_samples, create_fewshot_message
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

# Default templates for multiple choice problems
DEFAULT_INSTRUCTIONS = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of the options. Think step by step before answering.
""".strip()

DEFAULT_EXAMPLE_TEMPLATE = """
PROBLEM:
{question}

{choices}

SOLUTION:
{solution}
""".strip()


def answer_options(choices: list[str]) -> str:
    """Format choices as A) ... B) ... etc."""
    return "\n".join(
        [f"{answer_character(i)}) {choice}" for i, choice in enumerate(choices)]
    )


def prompt(question: str, choices: list[str], template: str) -> str:
    """Format the multiple choice prompt."""
    choices_text = answer_options(choices)
    letters = ",".join(answer_character(i) for i in range(len(choices)))

    return template.format(
        choices=choices_text,
        letters=letters,
        question=question,
    )


def parse_answer(completion: str, num_choices: int) -> str | None:
    """Extract single answer from completion (A, B, C, etc.)."""
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z])\s*(?:$|\n|\.)",
        completion,
        flags=re.MULTILINE,
    )

    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z])(?:[^\w]|\n|$|\.)",
            completion,
        )

    if match is None:
        return None

    matched = match.group(1).strip().upper()
    allowed_options = set(answer_character(i) for i in range(num_choices))

    if matched in allowed_options:
        return matched

    return None


@solver
def multiple_choice_prefill(
    *,
    instruction_template: str | None = None,
    example_template: str | None = None,
    fewshot_config: FewShotConfig | None = None,
    prefill_config: PrefillConfig | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Multiple choice solver with prefill support.

    This solver:
    1. Optionally adds few-shot examples (different per task, excludes current sample)
    2. Formats the multiple choice prompt
    3. Optionally adds a prefill assistant message
    4. Calls generate() with continue_final_message=True if prefill was added

    Args:
        instruction_template: Custom instruction template (overrides default).
                             Used for both 0-shot and few-shot prompts.
        example_template: Custom example template (overrides default).
                         Used for both 0-shot and few-shot examples.
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: Optional prefill configuration
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    # Load prefill map if config provided
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
        if not state.choices:
            raise ValueError("multiple_choice_prefill requires samples with choices")

        # Format current task as an incomplete example
        choices_list = [choice.value for choice in state.choices]
        choices_text = answer_options(choices_list)

        current_task = example_template.format(
            question=state.user_prompt.text,
            choices=choices_text,
            solution=""  # Empty - to be completed by the model
        )

        # Handle few-shot prompting
        if fewshot_config and all_fewshot_samples:
            # Define formatter for multiple choice specific formatting
            def format_mc_sample(sample_data: dict) -> dict:
                """Format choices list into A) B) C) format for template."""
                if 'choices' in sample_data and isinstance(sample_data['choices'], list):
                    sample_data['choices'] = answer_options(sample_data['choices'])
                return sample_data

            user_content = create_fewshot_message(
                all_samples=all_fewshot_samples,
                config=fewshot_config,
                instruction_template=instruction_template,
                example_template=example_template,
                current_task=current_task,
                current_id=state.sample_id,
                seed=state.sample_id,
                format_sample=format_mc_sample,
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

        # Parse answer and grade
        answer_letter = parse_answer(state.output.completion, len(state.choices))
        if answer_letter:
            answer_idx = answer_index(answer_letter)
            for i in range(len(state.choices)):
                state.choices.mark_choice(i, i == answer_idx)

        return state

    return solve
