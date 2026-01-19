"""Extract model size and ECI from model names."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Pattern: -[digits, possibly with decimal][b or B]
# Examples: -0.6B, -32B, -8B, -4b, -1.5B
_SIZE_PATTERN = re.compile(r"-(\d+\.?\d*)[bB]")

# Cached ECI data
_ECI_CACHE: dict[str, float] | None = None


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


def _load_eci_cache() -> dict[str, float]:
    """Load and cache ECI values from fitted results."""
    global _ECI_CACHE
    if _ECI_CACHE is not None:
        return _ECI_CACHE

    _ECI_CACHE = {}

    # Try fitted values first
    fitted_path = Path(__file__).parent.parent.parent / "christine_experiments/20260107/eci_model_capabilities.csv"
    if fitted_path.exists():
        import pandas as pd
        df = pd.read_csv(fitted_path)
        for _, row in df.iterrows():
            _ECI_CACHE[row["model"]] = float(row["Cm"])
        return _ECI_CACHE

    # Fall back to Epoch's data
    try:
        from .eci import load_epoch_eci
        _ECI_CACHE = load_epoch_eci()
    except Exception as e:
        logger.warning(f"Could not load ECI data: {e}")

    return _ECI_CACHE


def model_eci(model: str) -> float | None:
    """Get ECI (Epoch Capabilities Index) for a model.

    Uses fitted ECI from christine_experiments/20260107/ if available,
    otherwise falls back to Epoch's pre-computed values.

    Examples:
        model_eci("Qwen2.5-7B-Instruct") -> 121.2
        model_eci("claude-3-5-sonnet-20240620") -> 130.0

    Returns:
        ECI score, or None if not found.
    """
    cache = _load_eci_cache()

    # Exact match
    if model in cache:
        return cache[model]

    # Partial match (model name contained in version string)
    model_lower = model.lower()
    for version, score in cache.items():
        if model_lower in version.lower():
            logger.warning(f"ECI partial match: '{model}' -> '{version}' = {score:.1f}")
            return score

    logger.warning(f"No ECI found for model: {model}")
    return None
