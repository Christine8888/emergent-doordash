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
        exclude_current: Whether to exclude the current sample from few-shot selection (default: True)
    """

    path: str
    id_field: str = "id"
    response_field: str = "response"
    num_examples: int = 5
    seed: int = 42
    exclude_current: bool = True


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

    with open(fewshot_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Load all fields from JSONL for template flexibility
                samples.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(samples)} few-shot samples from {config.path}")
    return samples


def create_fewshot_message(
    all_samples: list[dict],
    config: FewShotConfig,
    instruction_template: str,
    example_template: str,
    current_task: str,
    current_id: str | None = None,
    seed: int | str | None = None,
    format_sample: Callable[[dict], dict] | None = None,
) -> str:
    """Create a few-shot message with instructions, examples, and current task.

    Args:
        all_samples: List of sample dictionaries
        config: FewShotConfig with data loading settings
        instruction_template: Template for instructions
        example_template: Template for formatting examples
        current_task: Formatted current task to append
        current_id: ID of current sample to exclude (if config.exclude_current is True)
        seed: Random seed for sampling (overrides config.seed if provided)
        format_sample: Optional function to format sample data before template formatting.
                      Takes a sample dict and returns a formatted dict.

    Returns:
        Formatted message: instructions + examples + current_task
    """
    import random

    # Filter out current sample if requested
    available_samples = all_samples
    if config.exclude_current and current_id is not None:
        available_samples = [s for s in all_samples if s.get(config.id_field) != current_id]

    if len(available_samples) < config.num_examples:
        logger.warning(
            f"Requested {config.num_examples} few-shot examples, but only {len(available_samples)} "
            f"available{f' after excluding current sample {current_id}' if config.exclude_current else ''}"
        )

    # Sample examples for few-shot prompting
    effective_seed = seed if seed is not None else config.seed
    rng = random.Random(hash(effective_seed) if isinstance(effective_seed, str) else effective_seed)
    selected_samples = rng.sample(available_samples, min(config.num_examples, len(available_samples)))

    # Format each example using the template
    examples_text = []
    for sample in selected_samples:
        try:
            # Get 'solution' field
            sample_data = sample.copy()
            sample_data['solution'] = sample.get(config.response_field, '')

            # Apply custom formatting
            if format_sample:
                sample_data = format_sample(sample_data)

            example = example_template.format(**sample_data)
            examples_text.append(example)
        except KeyError as e:
            logger.warning(f"Missing field {e} in sample {sample.get(config.id_field, 'unknown')}")
            continue

    return instruction_template + "\n\n" + "\n\n".join(examples_text) + "\n\n" + current_task
