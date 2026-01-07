"""Extract model size from model names."""

import re

# Pattern: -[digits, possibly with decimal][b or B]
# Examples: -0.6B, -32B, -8B, -4b, -1.5B
_SIZE_PATTERN = re.compile(r"-(\d+\.?\d*)[bB]")


def size(model: str) -> float:
    """Extract model size in billions from model name.

    Looks for pattern: -[number][B or b]

    Examples:
        size("Qwen/Qwen3-0.6B") -> 0.6
        size("Qwen2.5-32B-Instruct") -> 32.0
        size("meta-llama/Llama-3.1-8B-Instruct") -> 8.0
        size("google/gemma-3-4b-it") -> 4.0

    Returns:
        Model size in billions, or 0.0 if not found.
    """
    match = _SIZE_PATTERN.search(model)
    if match:
        return float(match.group(1))
    return 0.0


def clean_name(model: str) -> str:
    """Clean up model names for display by extracting size portion.

    Examples:
        clean_name("Qwen2.5-0.5B-Instruct") -> "0.5B"
        clean_name("Qwen2.5-7B-Instruct") -> "7B"
        clean_name("meta-llama/Llama-3.1-8B-Instruct") -> "8B"
    """
    match = _SIZE_PATTERN.search(model)
    if match:
        return match.group(1) + "B"
    return model
