"""Utility functions for GPQA multiple choice extraction and grading."""

import re
from inspect_ai._util.answer import answer_character


def format_answer_options(choices: list[str]) -> str:
    """Format choices as A) ... B) ... etc."""
    return "\n".join(
        [f"{answer_character(i)}) {choice}" for i, choice in enumerate(choices)]
    )


def extract_answer(completion: str, num_choices: int = 4) -> str:
    """Extract single answer letter from completion (A, B, C, etc.).

    Supports various answer formats:
    - ANSWER: X or Answer: X
    - Final Answer: X
    - ANSWER: [X] (with brackets)
    - Answer: X) (with parenthesis)
    - \\boxed{X} (LaTeX format)
    - X) (just letter with parenthesis, as fallback)
    - Single letter (final fallback)

    Handles whitespace/newlines between "Answer" and the letter.
    Returns the LAST occurrence found (most recent answer).

    Args:
        completion: The model's completion text
        num_choices: Number of available choices (default: 4)

    Returns:
        The answer letter (A, B, C, etc.) or empty string if no valid answer found
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

    # Normalize: uppercase, strip whitespace, remove $ signs
    normalized = completion.strip().upper()
    normalized = re.sub(r'\$', '', normalized)

    # Find all matches for all patterns
    for pattern in patterns:
        matches = re.finditer(pattern, normalized, flags=re.MULTILINE | re.DOTALL)
        for match in matches:
            letter = match.group(1).strip().upper()
            letter = re.sub(r'\$', '', letter)
            # Store (position, letter) to track order
            all_matches.append((match.start(), letter))

    # Return the last match (most recent answer)
    if all_matches:
        all_matches.sort(key=lambda x: x[0])  # Sort by position
        last_letter = all_matches[-1][1]  # Return the letter from last match

        # Validate the letter is within valid range
        if _validate_answer_letter(last_letter, num_choices):
            return last_letter

    # Final fallback: check if the entire completion (stripped) is just a single letter
    stripped = completion.strip()
    if len(stripped) == 1 and stripped.isalpha():
        letter = stripped.upper()
        if _validate_answer_letter(letter, num_choices):
            return letter

    return ""


def _validate_answer_letter(letter: str, num_choices: int) -> bool:
    """Check if extracted letter is a valid choice (A, B, C, etc.)."""
    allowed_options = set(answer_character(i) for i in range(num_choices))
    return letter in allowed_options


async def grade_answer(extracted_letter: str, target: str) -> bool:
    """Grade GPQA answer by comparing extracted letter to target.

    Args:
        extracted_letter: Extracted answer letter (single letter or empty string)
        target: Target answer (single letter)

    Returns:
        True if extracted letter matches target (case-insensitive)
    """
    if not extracted_letter:
        return False

    return extracted_letter.upper() == target.upper()
