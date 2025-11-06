"""Few-shot example utilities for solver prompts."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from evals.example import Example

logger = logging.getLogger(__name__)


@dataclass
class FewShotConfig:
    """Configuration for few-shot examples.

    JSONL files must have standardized fields:
    - id: sample identifier (required)
    - question: the question text (required)
    - response: the solution/response (required for few-shot)
    - target: the target answer (optional)

    Args:
        path: Path to JSONL file containing few-shot data
        num_examples: Number of examples to include (default: 5)
        seed: Random seed for sampling examples (default: 42)
        exclude_current: Whether to exclude the current sample from few-shot selection (default: True)
        prefix: Optional text to put before examples (default: "")
        suffix: Optional text to put after examples (default: "")
    """

    path: str
    num_examples: int = 5
    seed: int = 42
    exclude_current: bool = True
    prefix: str = ""
    suffix: str = ""

    def __post_init__(self):
        """Load few-shot data immediately after initialization."""
        self._data = load_fewshot_data(self)

    def get_data(self) -> dict[str, Example]:
        """Get few-shot data.

        Returns:
            Dictionary mapping sample IDs to Example objects
        """
        return self._data


def load_fewshot_data(config: FewShotConfig) -> dict[str, Example]:
    """Load few-shot data from JSONL file.

    Args:
        config: FewShotConfig with path

    Returns:
        Dictionary mapping sample IDs to Example objects

    Raises:
        FileNotFoundError: If few-shot file doesn't exist
        KeyError: If required fields are missing
        ValueError: If data is invalid
    """
    fewshot_data = {}
    fewshot_file = Path(config.path)

    if not fewshot_file.exists():
        raise FileNotFoundError(f"Few-shot file not found: {config.path}")

    logger.info(f"Loading few-shot data from {config.path}")

    with open(fewshot_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                example = Example.from_dict(data)

                if example.response:
                    fewshot_data[example.id] = example
                else:
                    logger.warning(f"Line {line_num}: Missing 'response' field, required for few-shot")

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Line {line_num}: {e}")

    logger.info(f"Loaded {len(fewshot_data)} few-shot examples from {config.path}")
    return fewshot_data


def format_fewshot_examples(
    fewshot_data: dict[str, Example],
    n_examples: int,
    example_template: str,
    current_id: str | None = None,
    seed: int | str = 42,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Format few-shot examples for prompting.

    Always excludes the current sample from few-shot selection to avoid leakage.

    Args:
        fewshot_data: Dictionary mapping sample IDs to Example objects
        n_examples: Number of examples to sample
        example_template: Template for formatting examples (expects {question} and {response})
        current_id: ID of current sample to exclude from few-shot selection
        seed: Random seed for deterministic sampling (default: 42)
        prefix: Optional text to put before the examples (default: "")
        suffix: Optional text to put after the examples (default: "")

    Returns:
        Formatted few-shot text: prefix + examples + suffix (or empty string if no examples)

    Example:
        >>> examples = format_fewshot_examples(
        ...     fewshot_data=data,
        ...     n_examples=2,
        ...     example_template="Q: {question}\\nA: {response}",
        ...     prefix="Here are some examples:",
        ...     suffix="Now solve the problem:"
        ... )
        # Returns:
        # Here are some examples:
        #
        # Q: What is 1+1?
        # A: 2
        #
        # Q: What is 2+2?
        # A: 4
        #
        # Now solve the problem:
    """
    import random

    # Always exclude current sample to avoid leakage
    available_ids = list(fewshot_data.keys())
    if current_id is not None:
        available_ids = [id for id in available_ids if id != current_id]

    if not available_ids:
        logger.warning("No examples available for few-shot (all filtered out)")
        return ""

    if len(available_ids) < n_examples:
        logger.warning(
            f"Requested {n_examples} few-shot examples, but only {len(available_ids)} "
            f"available after excluding current sample"
        )

    rng = random.Random(hash(seed) if isinstance(seed, str) else seed)
    selected_ids = rng.sample(available_ids, min(n_examples, len(available_ids)))

    examples_text = []
    for sample_id in selected_ids:
        example_obj = fewshot_data[sample_id]
        try:
            example = example_template.format(
                question=example_obj.question,
                response=example_obj.response
            )
            examples_text.append(example)
        except KeyError as e:
            logger.warning(f"Missing field {e} in template for sample {sample_id}")
            continue

    if not examples_text:
        return ""

    # Build final text
    result = "\n\n".join(examples_text)
    if prefix:
        result = prefix + "\n\n" + result
    if suffix:
        result = result + "\n\n" + suffix

    return result
