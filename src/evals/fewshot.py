"""Few-shot example utilities for solver prompts."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evals.example import Example

logger = logging.getLogger(__name__)

# Module-level cache for few-shot data
_FEWSHOT_DATA_CACHE: dict[str, dict[str, Example]] = {}


@dataclass
class FewShotConfig:
    """Configuration for few-shot examples.

    Caches loaded data to avoid reloading for every solver invocation.

    Args:
        path: Path to JSONL file containing few-shot data
        id_field: Field name containing the sample ID
        question_field: Field name containing the question (e.g., question with choices)
        response_field: Field name containing the response
        num_examples: Number of examples to include (default: 5)
        seed: Random seed for sampling examples (default: 42)
        exclude_current: Whether to exclude the current sample from few-shot selection (default: True)
    """

    path: str
    id_field: str = "id"
    question_field: str = "question_with_choices"
    response_field: str = "response"
    num_examples: int = 5
    seed: int = 42
    exclude_current: bool = True

    def get_data(self) -> dict[str, Example]:
        """Get cached few-shot data, loading if necessary.

        Returns:
            Dictionary mapping sample IDs to Example objects
        """
        # Use path as cache key
        if self.path not in _FEWSHOT_DATA_CACHE:
            logger.info(f"Loading few-shot data from {self.path}")
            _FEWSHOT_DATA_CACHE[self.path] = load_fewshot_data(self)

        return _FEWSHOT_DATA_CACHE[self.path]


def load_fewshot_data(config: FewShotConfig) -> dict[str, Example]:
    """Load few-shot data from JSONL file.

    Args:
        config: FewShotConfig with path and field names

    Returns:
        Dictionary mapping sample IDs to Example objects
    """
    fewshot_data = {}
    fewshot_file = Path(config.path)

    if not fewshot_file.exists():
        raise FileNotFoundError(f"Few-shot file not found: {config.path}")

    with open(fewshot_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                sample_id = data.get(config.id_field)
                question_text = data.get(config.question_field)
                response_text = data.get(config.response_field)

                if sample_id and question_text and response_text:
                    fewshot_data[sample_id] = Example(
                        question=question_text,
                        response=response_text
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    logger.info(f"Loaded {len(fewshot_data)} few-shot examples from {config.path}")
    return fewshot_data


def create_fewshot_message(
    fewshot_data: dict[str, Example],
    config: FewShotConfig,
    instruction_template: str,
    example_template: str,
    current_task: str,
    current_id: str | None = None,
    seed: int | str | None = None,
) -> str:
    """Create a few-shot message with instructions, examples, and current task.

    Args:
        fewshot_data: Dictionary mapping sample IDs to Example objects
        config: FewShotConfig with data loading settings
        instruction_template: Template for instructions
        example_template: Template for formatting examples (expects {question} and {solution})
        current_task: Formatted current task to append
        current_id: ID of current sample to exclude (if config.exclude_current is True)
        seed: Random seed for sampling (overrides config.seed if provided)

    Returns:
        Formatted message: instructions + examples + current_task
    """
    import random

    # Filter out current sample if requested
    available_ids = list(fewshot_data.keys())
    if config.exclude_current and current_id is not None:
        available_ids = [id for id in available_ids if id != current_id]

    if len(available_ids) < config.num_examples:
        logger.warning(
            f"Requested {config.num_examples} few-shot examples, but only {len(available_ids)} "
            f"available{f' after excluding current sample {current_id}' if config.exclude_current else ''}"
        )

    # Sample examples for few-shot prompting
    effective_seed = seed if seed is not None else config.seed
    rng = random.Random(hash(effective_seed) if isinstance(effective_seed, str) else effective_seed)
    selected_ids = rng.sample(available_ids, min(config.num_examples, len(available_ids)))

    # Format each example using the template
    examples_text = []
    for sample_id in selected_ids:
        example_obj = fewshot_data[sample_id]
        try:
            example = example_template.format(
                question=example_obj.question,
                solution=example_obj.response
            )
            examples_text.append(example)
        except KeyError as e:
            logger.warning(f"Missing field {e} in template for sample {sample_id}")
            continue

    return instruction_template + "\n\n" + "\n\n".join(examples_text) + "\n\n" + current_task
