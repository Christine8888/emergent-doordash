"""Multiple choice solver with prefill support for vLLM continuation."""

import logging
import re

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.util import resource

from evals.prefill import PrefillConfig, load_prefill_data
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


def _try_extract_answer_letter(completion: str) -> str | None:
    """Try to extract answer letter from completion using multiple patterns.

    Supports various answer formats:
    - ANSWER: X or Answer: X
    - Final Answer: X
    - ANSWER: [X] (with brackets)
    - Answer: X) (with parenthesis)
    - \\boxed{X} (LaTeX format)
    - X) (just letter with parenthesis, as fallback)

    Handles whitespace/newlines between "Answer" and the letter.
    Returns the LAST occurrence found (most recent answer).

    Returns the extracted letter (uppercase) or None if no match found.
    """
    # Define patterns to try (ordered by specificity)
    patterns = [
        # LaTeX boxed format: \boxed{A}
        r"\\boxed\s*\{\s*([A-Za-z])\s*\}",

        # Answer with brackets: ANSWER: [A] or Answer: [A]
        r"(?i)(?:final\s+)?answer\s*:\s*\[\s*([A-Za-z])\s*\]",

        # Answer with letter and parenthesis: Answer: C) or ANSWER: C)
        r"(?i)(?:final\s+)?answer\s*:\s*([A-Za-z])\s*\)",

        # Answer with just letter: ANSWER: C or Final Answer: A
        # Allow newlines and whitespace between "Answer:" and the letter
        r"(?i)(?:final\s+)?answer\s*:\s*\n?\s*([A-Za-z])",

        # Fallback: Just letter followed by parenthesis (e.g., "C)")
        # Use word boundary or start of line to avoid matching mid-word
        r"(?:^|\s)([A-Za-z])\s*\)",
    ]

    all_matches = []

    # Find all matches for all patterns
    for pattern in patterns:
        matches = re.finditer(pattern, completion, flags=re.MULTILINE | re.DOTALL)
        for match in matches:
            # Store (position, letter) to track order
            all_matches.append((match.start(), match.group(1).strip().upper()))

    # Return the last match (most recent answer)
    if all_matches:
        all_matches.sort(key=lambda x: x[0])  # Sort by position
        return all_matches[-1][1]  # Return the letter from last match

    return None


def _validate_answer_letter(letter: str, num_choices: int) -> bool:
    """Check if extracted letter is a valid choice (A, B, C, etc.)."""
    allowed_options = set(answer_character(i) for i in range(num_choices))
    return letter in allowed_options


def parse_answer(completion: str, num_choices: int) -> str | None:
    """Extract single answer from completion (A, B, C, etc.).

    Attempts to find "ANSWER: X" pattern in the completion and validates
    that the extracted letter corresponds to a valid choice.

    Args:
        completion: The model's completion text
        num_choices: Number of available choices

    Returns:
        The answer letter (A, B, C, etc.) or None if no valid answer found
    """
    letter = _try_extract_answer_letter(completion)

    if letter is None:
        return None

    if _validate_answer_letter(letter, num_choices):
        return letter

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

    Two modes:
    1. WITH prefill: Uses question_with_choices from the prefill JSONL file
    2. WITHOUT prefill: Uses normal formatting with choices (A, B, C, etc.)

    This solver:
    1. Optionally adds few-shot examples (different per task, excludes current sample)
    2. Formats the question (using prefill source or normal formatting)
    3. Optionally adds a prefill assistant message
    4. Calls generate() with continue_final_message=True if prefill was added

    Args:
        instruction_template: Custom instruction template (overrides default).
                             Used for both 0-shot and few-shot prompts.
        example_template: Custom example template (overrides default).
                         Used for both 0-shot and few-shot examples.
                         Should expect {question} and {solution} fields.
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: Optional prefill configuration. When provided, loads
                       question_with_choices from the JSONL file.
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    # Load prefill data if config provided
    question_map = {}
    prefill_map = {}
    if prefill_config:
        question_map, prefill_map = load_prefill_data(prefill_config)

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

        # Two paths for getting question_with_choices (question + formatted choices, no instructions):
        # 1. WITH prefill: Use pre-formatted question_with_choices from JSONL
        # 2. WITHOUT prefill: Format question with choices normally
        use_preformatted = question_map and state.sample_id in question_map

        if use_preformatted:
            # Path 1: Use pre-formatted question_with_choices from prefill JSONL
            question_with_choices = question_map[state.sample_id]
        else:
            # Path 2: Format normally with choices (A) B) C) ...)
            choices_list = [choice.value for choice in state.choices]
            choices_text = answer_options(choices_list)
            question_with_choices = f"{state.user_prompt.text}\n\n{choices_text}"

        # Save question_with_choices (question + choices, no instructions) to metadata
        if state.metadata is None:
            state.metadata = {}
        state.metadata["question_with_choices"] = question_with_choices

        # Both paths: wrap in example template (PROBLEM: ... SOLUTION: ...)
        current_task = example_template.format(
            question=question_with_choices,
            solution=""
        )

        # Both paths: add instructions (0-shot or few-shot)
        if fewshot_config and all_fewshot_samples:
            user_content = create_fewshot_message(
                all_samples=all_fewshot_samples,
                config=fewshot_config,
                instruction_template=instruction_template,
                example_template=example_template,
                current_task=current_task,
                current_id=state.sample_id,
                seed=state.sample_id,
                format_sample=None,  # No formatting needed - questions already have choices
            )
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
