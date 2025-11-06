import re


def format_grid(grid: list[list[int]]) -> str:
    """Format a 2D grid as a space-separated string."""
    return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])


def construct_arc_prompt(train_examples: list[dict], test_input: list[list[int]]) -> str:
    """Construct ARC prompt from training examples and test input.

    Args:
        train_examples: List of dicts with 'input' and 'output' keys
        test_input: 2D grid (list of lists) for the test case

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    for i, example in enumerate(train_examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append("")
        prompt_parts.append("INPUT:")
        prompt_parts.append(format_grid(example["input"]))
        prompt_parts.append("")
        prompt_parts.append("ANSWER:")
        prompt_parts.append(format_grid(example["output"]))
        prompt_parts.append("")

    prompt_parts.append("Below is a test input grid. Predict the corresponding output grid by applying the rule you found.")
    prompt_parts.append("Think through the pattern step by step, then provide your final answer.")
    prompt_parts.append("Your final answer should be formatted as:")
    prompt_parts.append("ANSWER:")
    prompt_parts.append("[the output grid with numbers separated by spaces]")
    prompt_parts.append("")
    prompt_parts.append("INPUT:")
    prompt_parts.append(format_grid(test_input))

    return "\n".join(prompt_parts)


def extract_answer(completion: str) -> str:
    """Extract the ANSWER: portion from model completion.

    Finds the last occurrence of ANSWER: pattern and returns everything after it.
    """
    pattern = r'(?i)(?:^|\n)ANSWER\s*:\s*(.+)'
    matches = list(re.finditer(pattern, completion, re.DOTALL))

    if matches:
        answer = matches[-1].group(1).strip()
        return answer

    return completion.strip()


def normalize_for_grading(text: str) -> str:
    """Normalize text by keeping only digits and their relative positions.

    Removes all whitespace, punctuation, and letters, keeping only numbers.
    """
    normalized = re.sub(r'[^0-9]', '', text)
    return normalized


async def grade_answer(extracted_answer: str, target: str) -> bool:
    """Grade ARC answer by normalizing both extracted answer and target.

    Args:
        extracted_answer: Extracted answer from model's response
        target: Target output grid as formatted string

    Returns:
        True if normalized extracted answer matches normalized target
    """
    norm_response = normalize_for_grading(extracted_answer)
    norm_target = normalize_for_grading(target)

    return norm_response == norm_target
