"""Multiple choice solver with prefill support for vLLM continuation."""

import logging
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import resource
from inspect_ai._util.answer import answer_index

from evals.example import Example
from evals.prefill import PrefillConfig
from evals.fewshot import FewShotConfig, create_fewshot_message
from evals.hint import get_prefill_fraction
from evals.solvers.mcq_utils import parse_answer, format_answer_options


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
    1. Formats the question with choices (A, B, C, etc.)
    2. Optionally adds few-shot examples (different per task, excludes current sample)
    3. Optionally adds a prefill assistant message from prefill data
    4. Calls generate() with continue_final_message=True if prefill was added

    When using prefill:
    - Dataset should be loaded from prefill JSONL with choices in correct order
    - No shuffling should occur (handled in task definition)
    - The (choices, target) tuple is preserved from the prefill data

    Args:
        instruction_template: Custom instruction template (overrides default).
                             Used for both 0-shot and few-shot prompts.
        example_template: Custom example template (overrides default).
                         Used for both 0-shot and few-shot examples.
                         Should expect {question} and {solution} fields.
        fewshot_config: FewShotConfig for few-shot examples
        prefill_config: Optional prefill configuration. When provided, loads
                       response text from the JSONL file for prefilling.
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds for generation (default: None)
    """
    # Get cached prefill data if config provided
    prefill_data = {}
    if prefill_config:
        prefill_data = prefill_config.get_data()

    # Get cached few-shot data if config provided
    fewshot_data = {}
    if fewshot_config:
        fewshot_data = fewshot_config.get_data()

    if instruction_template is None:
        instruction_template = DEFAULT_INSTRUCTIONS
    if example_template is None:
        example_template = DEFAULT_EXAMPLE_TEMPLATE

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("multiple_choice_prefill requires samples with choices")

        # Format question with choices
        # When using prefill, choices are already in the correct order (no shuffling)
        # so the target matches the choice ordering
        choices_list = [choice.value for choice in state.choices]
        choices_text = format_answer_options(choices_list)
        question_with_choices = f"{state.user_prompt.text}\n\n{choices_text}"

        # Record question & ordering of choices
        state.metadata["question_with_choices"] = question_with_choices

        current_task = example_template.format(
            question=question_with_choices,
            solution=""
        )

        # Add instructions (0-shot or few-shot)
        if fewshot_config and fewshot_data:
            user_content = create_fewshot_message(
                fewshot_data=fewshot_data,
                config=fewshot_config,
                instruction_template=instruction_template,
                example_template=example_template,
                current_task=current_task,
                current_id=state.sample_id,
                seed=state.sample_id,
            )
        else:
            user_content = instruction_template + "\n\n" + current_task

        state.user_prompt.text = user_content

        # Handle prefill if available
        # Skip prefilling if fraction is 0.0 (useful for ablation studies)
        continue_message = False
        if state.sample_id in prefill_data and prefill_config.fraction > 0.0:
            full_response = prefill_data[state.sample_id].response
            prefill_text = get_prefill_fraction(
                full_response,
                fraction=prefill_config.fraction
            )
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