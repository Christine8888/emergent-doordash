"""Few-shot example utilities for solver prompts."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class FewShotConfig:
    """Configuration for few-shot examples.

    Args:
        path: Path to JSONL file containing few-shot data
        id_field: Field name containing the sample ID
        response_field: Field name containing the full response text
        num_examples: Number of examples to include (default: 5)
        seed: Random seed for sampling examples (default: 42)
        system_template: Template for the system message containing examples.
                        Must contain {examples} placeholder.
        example_template: Template for individual examples. Can reference any
                         fields from the JSONL data (e.g., {question}, {response}, {target}).
        exclude_current: Whether to exclude the current sample from few-shot selection (default: True)
        additional_fields: List of additional field names to extract from JSONL (default: [])
    """

    path: str
    id_field: str = "id"
    response_field: str = "response"
    num_examples: int = 5
    seed: int = 42
    system_template: str = "You will be asked to solve a problem. Some examples of problems and solutions are provided below.\n\n{examples}"
    example_template: str = "PROBLEM:\n{question}\n\nSOLUTION:\n{response}\nANSWER: {target}"
    exclude_current: bool = True
    additional_fields: list[str] | None = None


def load_fewshot_samples(config: FewShotConfig) -> list[dict]:
    """Load few-shot samples from JSONL file.

    Args:
        config: FewShotConfig with path and field names

    Returns:
        List of dictionaries containing sample data
    """
    samples = []
    fewshot_file = Path(config.path)

    if not fewshot_file.exists():
        raise FileNotFoundError(f"Few-shot file not found: {config.path}")

    # Determine which fields to extract
    fields_to_extract = [config.id_field, config.response_field]
    if config.additional_fields:
        fields_to_extract.extend(config.additional_fields)

    with open(fewshot_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                sample = {}

                # Extract all specified fields
                for field in fields_to_extract:
                    if field in data:
                        sample[field] = data[field]

                # Also extract all fields for template flexibility
                sample.update(data)

                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(samples)} few-shot samples from {config.path}")
    return samples


def create_fewshot_system_message(
    all_samples: list[dict],
    config: FewShotConfig,
    current_id: str | None = None,
    seed: int | str | None = None,
) -> str:
    """Create a few-shot system message from samples.

    Args:
        all_samples: List of sample dictionaries
        config: FewShotConfig with templates and settings
        current_id: ID of current sample to exclude (if config.exclude_current is True)
        seed: Random seed for sampling (overrides config.seed if provided)

    Returns:
        Formatted system message with few-shot examples
    """
    import random

    # Filter out current sample if needed
    available_samples = all_samples
    if config.exclude_current and current_id is not None:
        available_samples = [s for s in all_samples if s.get(config.id_field) != current_id]

    if len(available_samples) < config.num_examples:
        logger.warning(
            f"Requested {config.num_examples} few-shot examples, but only {len(available_samples)} "
            f"available{f' after excluding current sample {current_id}' if config.exclude_current else ''}"
        )

    # Sample examples
    effective_seed = seed if seed is not None else config.seed
    rng = random.Random(hash(effective_seed) if isinstance(effective_seed, str) else effective_seed)
    selected_samples = rng.sample(available_samples, min(config.num_examples, len(available_samples)))

    # Format each example using the template
    examples = []
    for sample in selected_samples:
        try:
            example = config.example_template.format(**sample)
            examples.append(example)
        except KeyError as e:
            logger.warning(f"Missing field {e} in sample {sample.get(config.id_field, 'unknown')}")
            continue

    # Format system message
    examples_text = "\n\n".join(examples)
    return config.system_template.format(examples=examples_text)
