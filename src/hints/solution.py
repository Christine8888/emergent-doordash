#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from inspect_ai.model import ChatMessageUser

from hints.sample_utils import (
    collect_samples,
    create_base_parser,
    sample_input_to_str,
    format_choices_for_prompt,
    append_choices_to_messages,
)
from inspect_ai.dataset import Sample
import re


def extract_solution(x: str) -> str:
    match = re.search(r"<solution>(.*?)</solution>", x, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_answer(response: str) -> str:
    """Extract answer from response using <answer> </answer> tags, to verify the answer is correct."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else response


SOLUTION_INSTRUCTIONS = (
    "\nFirst, include your detailed and careful reasoning in <thinking> </thinking> tags. "
    "Then, once you are completely confident in your answer, write your final answer in <answer> </answer> tags. "
    "If the question is multiple choice, the answer should ONLY be the letter of the correct answer and nothing else.\n"
    "Finally, given your confident final answer, write a detailed explanation of your answer in <solution> </solution> tags. "
    "This solution should 1) be anywhere from 250 to 1000 words 2) end with ANSWER: [the correct answer to the actual question, "
    "which matches the answer you wrote in <answer> </answer> tags] on a new line 3) be detailed enough that someone with sufficient "
    "background can easily follow the correct solution, 4) fully sequential, so that the actual answer is not revealed until after ANSWER: , "
    "and so that the process of solving the problem can be followed in complete detail step by step. The goal of this solution is *pedagogical*. "
    "We will provide some fraction of this solution as a hint to a student who is trying to solve the problem. "
    "Therefore, make sure the solution is fully detailed and easy to follow, but never reveals the final answer until after ANSWER:.\n"
)


def format_solution_prompt(sample: Sample) -> str | tuple[list | str, str]:
    """Format prompt for solution hint generation. Returns (model_input, prompt) when sample.input is a list (matches baseline)."""
    prefix = "Solve the following problem by thinking carefully step by step.\n\n"
    choices = getattr(sample, "choices", None)

    if isinstance(sample.input, list):
        # Append choices to last user message so the model sees the answer options
        input_msgs = append_choices_to_messages(sample.input, choices)
        model_input = list(input_msgs) + [ChatMessageUser(content=prefix + SOLUTION_INSTRUCTIONS)]
        prompt = prefix + sample_input_to_str(input_msgs) + SOLUTION_INSTRUCTIONS
        return (model_input, prompt)

    # String input — append choices inline if present
    input_text = sample_input_to_str(sample.input)
    choices_str = format_choices_for_prompt(choices)
    if choices_str:
        input_text = input_text + "\n\n" + choices_str
    prompt = prefix + input_text + SOLUTION_INSTRUCTIONS
    return prompt


async def main():
    parser = create_base_parser("Sample eval problems until all have correct solutions")
    args, _ = parser.parse_known_args()

    await collect_samples(
        args=args,
        response_to_hint=extract_solution,  # hint = response for CoT
        format_fn=format_solution_prompt,  # use eval_config.format_prompt
        extract_fn=extract_answer,  # use eval_config.extract_answer
    )


if __name__ == "__main__":
    asyncio.run(main())