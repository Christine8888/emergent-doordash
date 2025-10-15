"""Prefill utilities for vLLM-based evals with assistant continuation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from evals.example import Example

logger = logging.getLogger(__name__)

# Module-level cache for prefill data
_PREFILL_DATA_CACHE: dict[str, dict[str, Example]] = {}


@dataclass
class PrefillConfig:
    """Configuration for prefill solver.

    Caches loaded data to avoid reloading for every solver invocation.

    Args:
        path: Path to JSONL file containing prefill data
        fraction: Fraction of words to include from response (0.0 to 1.0)
        id_field: Field name containing the sample ID
        question_field: Field name containing the question (e.g., question with choices)
        response_field: Field name containing the response
        target_field: Field name containing the target answer (optional)
    """

    path: str
    fraction: float = 0.5
    id_field: str = "id"
    question_field: str = "question"
    response_field: str = "response"
    target_field: str = "target"

    def get_data(self) -> dict[str, Example]:
        """Get cached prefill data, loading if necessary.

        Returns:
            Dictionary mapping sample IDs to Example objects
        """
        # Use path as cache key
        if self.path not in _PREFILL_DATA_CACHE:
            logger.info(f"Loading prefill data from {self.path}")
            _PREFILL_DATA_CACHE[self.path] = load_prefill_data(self)

        return _PREFILL_DATA_CACHE[self.path]

    def get_available_ids(self) -> set[str]:
        """Get set of sample IDs that have prefill data available.

        Returns:
            Set of sample IDs
        """
        prefill_data = self.get_data()
        return set(prefill_data.keys())


def load_prefill_map(config: PrefillConfig) -> dict[str, str]:
    """Load prefill data from JSONL file into a dictionary.

    Args:
        config: PrefillConfig with path and field names

    Returns:
        Dictionary mapping sample IDs to response texts
    """
    prefill_map = {}
    prefill_file = Path(config.path)

    if not prefill_file.exists():
        raise FileNotFoundError(f"Prefill file not found: {config.path}")

    with open(prefill_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                sample_id = data.get(config.id_field)
                response_text = data.get(config.response_field)

                if sample_id and response_text:
                    prefill_map[sample_id] = response_text
                else:
                    logger.warning(f"Line {line_num}: No sample ID or response text found in {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(prefill_map)} prefill entries from {config.path}")
    return prefill_map


def load_prefill_data(config: PrefillConfig) -> dict[str, Example]:
    """Load prefill data from JSONL file.

    Args:
        config: PrefillConfig with path and field names

    Returns:
        Dictionary mapping sample IDs to Example objects
    """
    prefill_data = {}
    prefill_file = Path(config.path)

    if not prefill_file.exists():
        raise FileNotFoundError(f"Prefill file not found: {config.path}")

    with open(prefill_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                sample_id = data.get(config.id_field)
                question_text = data.get(config.question_field)
                response_text = data.get(config.response_field)
                target_text = data.get(config.target_field)  # Optional

                if sample_id and question_text and response_text:
                    prefill_data[sample_id] = Example(
                        question=question_text,
                        response=response_text,
                        target=target_text
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(prefill_data)} prefill entries from {config.path}")
    return prefill_data
