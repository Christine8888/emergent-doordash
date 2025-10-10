"""Multiple choice solver with prefill support for vLLM continuation."""

import logging
import re

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.util import resource

from evals.prefill import PrefillConfig, load_prefill_map
from evals.fewshot import FewShotConfig, load_fewshot_samples, create_fewshot_system_message
from evals.hint import get_prefill_fraction

logger = logging.getLogger(__name__)

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

PROBLEM:
{question}

{choices}

SOLUTION:
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
        template: Template for the question (defaults to COT template)
        fewshot_config: FewShotConfig for few-shot examples
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

    # Load and format few-shot samples
    all_fewshot_samples = []
    if fewshot_config:
        raw_samples = load_fewshot_samples(fewshot_config)

        # Format each sample to match the prompt format
        for sample in raw_samples:
            formatted_sample = sample.copy()
            # Format question with choices to match the PROBLEM: format
            question = sample.get('question', '')
            choices = sample.get('choices', [])
            formatted_example = f"{question}\n{answer_options(choices)}"
            formatted_sample['formatted_example'] = formatted_example
            all_fewshot_samples.append(formatted_sample)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("multiple_choice_prefill requires samples with choices")

        # Construct main prompt
        choices_list = [choice.value for choice in state.choices]
        formatted_question = prompt(
            question=state.user_prompt.text,
            choices=choices_list,
            template=str(template),
        )

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

        # Parse answer and grade
        answer_letter = parse_answer(state.output.completion, len(state.choices))
        if answer_letter:
            answer_idx = answer_index(answer_letter)
            for i in range(len(state.choices)):
                state.choices.mark_choice(i, i == answer_idx)

        return state

    return solve
