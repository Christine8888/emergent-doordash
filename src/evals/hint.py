import re


def get_prefill_fraction(reasoning: str, fraction: float = 0.5, stop_string: str = "ANSWER:") -> str:
    """Extract a fraction of the reasoning text for prefilling.

    Args:
        reasoning: The full reasoning text to extract from
        fraction: Fraction of words to include (must be > 0.0)
        stop_string: String to stop at if encountered

    Returns:
        Non-empty prefill text

    Raises:
        ValueError: If reasoning is empty, fraction is 0.0, or extracted text is empty
    """
    if not reasoning or not reasoning.strip():
        raise ValueError("Cannot create prefill from empty reasoning text")

    if fraction <= 0.0:
        raise ValueError(f"Fraction must be > 0.0, got {fraction}")

    # Split on whitespace while capturing it
    tokens = re.split(r'(\s+)', reasoning)

    # Filter to just words (non-whitespace tokens)
    words = [t for t in tokens]
    num_words = int(len(words) * fraction)

    if num_words == 0:
        raise ValueError(f"Fraction {fraction} results in 0 words from {len(words)} total words")

    result = []
    word_count = 0
    for token in tokens:
        if word_count >= num_words:
            break
        word_count += 1
        result.append(token)

        # add stop string to result but break afterwards
        if token == stop_string:
            break

    prefill_text = "".join(result).strip()

    if not prefill_text:
        raise ValueError("Extracted prefill text is empty")

    return prefill_text
