import re
from inspect_ai._util.answer import answer_character, answer_index

def format_answer_options(choices: list[str]) -> str:
    """Format choices as A) ... B) ... etc."""
    return "\n".join(
        [f"{answer_character(i)}) {choice}" for i, choice in enumerate(choices)]
    )


def prompt(question: str, choices: list[str], template: str) -> str:
    """Format the multiple choice prompt."""
    choices_text = format_answer_options(choices)
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