"""Prefill utilities for vLLM-based evals with assistant continuation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PrefillConfig:
    """Configuration for prefill solver.

    Args:
        path: Path to JSONL file containing prefill data
        fraction: Fraction of words to include from response (0.0 to 1.0)
        response_field: Field name containing the full response text
        id_field: Field name containing the sample ID
        question_field: Field name containing the question with formatted choices (optional)
    """

    path: str
    fraction: float = 0.5
    response_field: str = "response"
    id_field: str = "id"
    question_field: str = "question_with_choices"


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
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(prefill_map)} prefill entries from {config.path}")
    return prefill_map


def load_prefill_data(config: PrefillConfig) -> tuple[dict[str, str], dict[str, str]]:
    """Load both question_with_choices and prefill responses from JSONL file.

    Args:
        config: PrefillConfig with path and field names

    Returns:
        Tuple of (question_map, prefill_map) where:
        - question_map: Dictionary mapping sample IDs to question_with_choices
        - prefill_map: Dictionary mapping sample IDs to response texts
    """
    question_map = {}
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
                question_text = data.get(config.question_field)

                if sample_id and response_text:
                    prefill_map[sample_id] = response_text

                if sample_id and question_text:
                    question_map[sample_id] = question_text
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(prefill_map)} prefill entries and {len(question_map)} questions from {config.path}")
    return question_map, prefill_map
