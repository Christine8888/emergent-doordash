"""Prefill utilities for vLLM-based evals with assistant continuation."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from evals.example import Example

logger = logging.getLogger(__name__)


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

    tokens = re.split(r'(\s+)', reasoning)
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
        # this is so we use hint fraction = 1.0 and still stop before the actual answer
        if token == stop_string:
            break

    prefill_text = "".join(result).strip()

    if not prefill_text:
        raise ValueError("Extracted prefill text is empty")

    return prefill_text


@dataclass
class PrefillConfig:
    """Configuration for prefill solver.

    JSONL files must have standardized fields:
    - id: sample identifier (required)
    - question: the question text (required)
    - target: the target answer (required)
    - response: the full response from the model (required)
    - hint: the hint to use for prefill (required)

    Args:
        path: Path to JSONL file containing prefill data
        fraction: Fraction of words to include from hint (0.0 to 1.0)
    """

    path: str
    fraction: float = 0.5

    def __post_init__(self):
        """Load prefill data immediately after initialization."""
        self._data = load_prefill_data(self)

    def get_data(self) -> dict[str, str]:
        """Get prefill data.

        Returns:
            Dictionary mapping sample IDs to prefill text (truncated by fraction)
        """
        return self._data

    def get_available_ids(self) -> set[str]:
        """Get set of sample IDs that have prefill data available.

        Returns:
            Set of sample IDs
        """
        return set(self._data.keys())


def load_prefill_data(config: PrefillConfig) -> dict[str, str]:
    """Load prefill data from JSONL file and compute prefill text.

    Args:
        config: PrefillConfig with path and fraction

    Returns:
        Dictionary mapping sample IDs to prefill text (truncated by fraction)

    Raises:
        FileNotFoundError: If prefill file doesn't exist
        KeyError: If required fields are missing
        ValueError: If data is invalid or prefill text is empty
    """
    prefill_data = {}
    prefill_file = Path(config.path)

    if not prefill_file.exists():
        raise FileNotFoundError(f"Prefill file not found: {config.path}")

    with open(prefill_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                example = Example.from_dict(data)

                if example.hint:
                    assert isinstance(example.hint, str), "Hint must be a string"
                    # Only compute fraction if > 0.0, otherwise store full hint for validation
                    if config.fraction > 0.0:
                        prefill_text = get_prefill_fraction(example.hint, fraction=config.fraction)
                    else:
                        prefill_text = example.hint  # Full hint, but won't be used by solver
                    prefill_data[example.id] = prefill_text

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Line {line_num}: {e}")

    logger.info(f"Loaded {len(prefill_data)} hints with fraction={config.fraction} from {config.path}")
    return prefill_data
