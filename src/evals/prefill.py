"""Prefill utilities for vLLM-based evals with assistant continuation."""

import json
import logging
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from evals.example import Example

logger = logging.getLogger(__name__)


def _truncate_at_stop_string(text: str, stop_string: str) -> str:
    """Truncate text before stop_string, warning if not found."""
    if stop_string not in text:
        logger.warning(f"stop_string '{stop_string}' not found in text")
        return text
    return text[:text.index(stop_string)].strip()


def _split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    """Split text into tokens preserving whitespace, return tokens and word indices."""
    tokens = re.split(r'(\s+)', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip()]
    return tokens, word_indices


def get_prefill_fraction(text: str, fraction: float = 0.5, stop_string: str = "ANSWER:") -> str:
    """Extract the first fraction of words from text truncated at stop_string."""
    text = _truncate_at_stop_string(text, stop_string)
    tokens, word_indices = _split_preserving_whitespace(text)

    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return text

    last_word_idx = word_indices[num_words - 1]
    return "".join(tokens[:last_word_idx + 1]).strip()


def get_masked_text(text: str, fraction: float = 0.5, mask_token: str = "[MASK]", stop_string: str = "ANSWER:") -> str:
    """Mask random words, showing only a fraction of them. Truncates at stop_string."""
    text = _truncate_at_stop_string(text, stop_string)
    tokens, word_indices = _split_preserving_whitespace(text)

    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask == 0:
        return text

    mask_indices = set(random.sample(word_indices, num_to_mask))
    return "".join(mask_token if i in mask_indices else t for i, t in enumerate(tokens)).strip()


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
        fraction: Fraction of words to show from hint (0.0 to 1.0)
        mode: How to apply fraction - "sequential" shows first fraction*words,
              "masked" shows fraction*words at random positions (masks the rest)
        mask_token: Token to use for masking in masked mode
        stop_string: String to stop at - inclusive for sequential, exclusive for masked
    """

    path: str
    fraction: float = 0.5
    mode: str = "sequential"
    mask_token: str = "[MASK]"
    stop_string: str = "ANSWER:"

    def __post_init__(self):
        """Load prefill data immediately after initialization."""
        if self.mode not in ["sequential", "masked"]:
            raise ValueError(f"Mode must be 'sequential' or 'masked', got '{self.mode}'")
        self._data = load_prefill_data(self)

    def get_data(self) -> dict[str, dict[int, str]]:
        """Get prefill data.

        Returns:
            Dictionary mapping sample IDs to dict of sample_idx -> prefill text
        """
        return self._data

    def get_available_ids(self) -> set[str]:
        """Get set of sample IDs that have prefill data available.

        Returns:
            Set of sample IDs
        """
        return set(self._data.keys())


def load_prefill_data(config: PrefillConfig) -> dict[str, dict[int, str]]:
    """Load prefill data from JSONL file and compute prefill text.

    Args:
        config: PrefillConfig with path and fraction

    Returns:
        Dictionary mapping sample IDs to dict of sample_idx -> prefill text.
        Returns empty dict if fraction is 0.
    """
    if config.fraction == 0.0:
        return {}

    prefill_data: dict[str, dict[int, str]] = {}
    prefill_file = Path(config.path)

    if not prefill_file.exists():
        raise FileNotFoundError(f"Prefill file not found: {config.path}")

    with open(prefill_file) as f:
        for line in f:
            data = json.loads(line)
            example = Example.from_dict(data)

            if not example.has_valid_hint():
                continue

            if config.mode == "sequential":
                prefill_text = get_prefill_fraction(example.hint, fraction=config.fraction, stop_string=config.stop_string)
            else:
                prefill_text = get_masked_text(example.hint, fraction=config.fraction, mask_token=config.mask_token, stop_string=config.stop_string)

            if example.id not in prefill_data:
                prefill_data[example.id] = {}
            prefill_data[example.id][example.sample_idx] = prefill_text

    total_samples = sum(len(samples) for samples in prefill_data.values())
    sample_counts = [len(samples) for samples in prefill_data.values()]
    if sample_counts:
        min_samples = min(sample_counts)
        max_samples = max(sample_counts)
        median_samples = statistics.median(sample_counts)
        logger.info(f"Loaded {total_samples} hints for {len(prefill_data)} questions (samples per question: min={min_samples}, max={max_samples}, median={median_samples}) with fraction={config.fraction}, mode={config.mode} from {config.path}")

        first_id = next(iter(prefill_data))
        first_sample = next(iter(prefill_data[first_id].values()))
        logger.info(f"Example hint (id={first_id}, {len(first_sample)} chars):\n{first_sample}")

    return prefill_data
