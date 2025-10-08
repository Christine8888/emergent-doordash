"""Multiple choice solver with prefill support for vLLM continuation."""

import logging
import re

from inspect_ai.model import ChatMessageAssistant, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.util import resource

from evals.prefill import PrefillConfig, load_prefill_map
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
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
    template: str | None = None,
    prefill_config: PrefillConfig | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> Solver:
    """Multiple choice solver with prefill support.

    This solver:
    1. Formats the multiple choice prompt
    2. Optionally adds a prefill assistant message
    3. Calls generate() with continue_final_message=True if prefill was added

    Args:
        template: Template for the question (defaults to COT template)
        prefill_config: Optional prefill configuration
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    if template is None:
        template = SINGLE_ANSWER_TEMPLATE_COT

    template = resource(template)

    # Load prefill map if config provided
    prefill_map = {}
    if prefill_config:
        prefill_map = load_prefill_map(prefill_config)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("multiple_choice_prefill requires samples with choices")

        # Store original question for later
        original_question = state.user_prompt.text

        # Format the multiple choice prompt
        choices_list = [choice.value for choice in state.choices]
        state.user_prompt.text = prompt(
            question=state.user_prompt.text,
            choices=choices_list,
            template=str(template),
        )

        # Add prefill if available
        continue_message = False
        if prefill_config and state.sample_id in prefill_map:
            full_response = prefill_map[state.sample_id]
            prefill_text = get_prefill_fraction(
                full_response,
                fraction=prefill_config.fraction
            )

            if prefill_text:
                # Add non-empty prefill text
                state.messages.append(ChatMessageAssistant(content=prefill_text))
                continue_message = True

        # Generate with or without continuation
        gen_config = GenerateConfig(
            max_tokens=max_tokens,
            continue_final_message=continue_message,
            timeout=timeout
        )
        state = await generate(state, config=gen_config)

        # Parse the answer
        answer_letter = parse_answer(state.output.completion, len(state.choices))

        if answer_letter:
            # Mark the chosen answer
            answer_idx = answer_index(answer_letter)
            for i in range(len(state.choices)):
                state.choices.mark_choice(i, i == answer_idx)

        return state

    return solve
