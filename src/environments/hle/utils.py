import re


def extract_answer(completion: str) -> str:
    """Extract the answer letter from model completion.

    Supports all the same patterns as mcq_utils:
    - LaTeX boxed format: \boxed{A}
    - Answer with brackets: ANSWER: [A]
    - Answer with letter and parenthesis: Answer: C)
    - Answer with just letter: ANSWER: C (allows newlines between)
    - Fallback: Just letter followed by parenthesis (e.g., "C)")
    - Final fallback: If completion is just a single letter (optionally with whitespace)

    Returns the LAST occurrence found (most recent answer).
    """
    # Define patterns to try (ordered by specificity, same as mcq_utils)
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

    # Find all matches for all patterns in original completion
    for pattern in patterns:
        matches = re.finditer(pattern, completion, flags=re.MULTILINE | re.DOTALL)
        for match in matches:
            # Normalize the extracted letter: uppercase, strip whitespace, remove $ signs
            letter = match.group(1).strip().upper()
            letter = re.sub(r'\$', '', letter)
            # Store (position, letter) to track order
            all_matches.append((match.start(), letter))

    # Return the last match (most recent answer)
    if all_matches:
        all_matches.sort(key=lambda x: x[0])  # Sort by position
        return all_matches[-1][1]  # Return the letter from last match

    # Final fallback: check if the entire completion (stripped) is just a single letter
    stripped = completion.strip()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()

    return ""


async def grade_hle_answer(response: str, target: str) -> bool:
    """Grade HLE multiple choice answer.

    Args:
        response: Model's full response
        target: Target answer (single letter)

    Returns:
        True if extracted letter matches target (case-insensitive)
    """
    extracted = extract_answer(response)
    if not extracted:
        return False

    return extracted.upper() == target.upper()
