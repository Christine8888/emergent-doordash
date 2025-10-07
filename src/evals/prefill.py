"""Prefill solver for vLLM-based evals with assistant continuation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Solver, TaskState, solver

from .hint import get_prefill_fraction

logger = logging.getLogger(__name__)


@dataclass
class PrefillConfig:
    """Configuration for prefill solver.

    Args:
        path: Path to JSONL file containing prefill data
        fraction: Fraction of words to include from response (0.0 to 1.0)
        response_field: Field name containing the full response text
        id_field: Field name containing the sample ID
    """

    path: str
    fraction: float = 0.5
    response_field: str = "response"
    id_field: str = "id"


@solver
def prefill(config: PrefillConfig):
    """Apply prefill to vLLM generation by continuing from a partial assistant response.

    Args:
        config: PrefillConfig object with prefill settings

    Example JSONL format:
        {"id": "sample_001", "response": "Let me solve this step by step..."}
        {"id": "sample_002", "response": "The answer is..."}

    Usage:
        from src.evals.prefill import PrefillConfig, prefill

        solver = [
            prompt_template(template),
            prefill(PrefillConfig(path="path/to/prefills.jsonl", fraction=0.3)),
            generate()
        ]
    """
    # Load prefill data once
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

                if sample_id is None:
                    logger.warning(
                        f"Line {line_num}: Missing '{config.id_field}' field, skipping"
                    )
                    continue

                if response_text is None:
                    logger.warning(
                        f"Line {line_num}: Missing '{config.response_field}' field for ID '{sample_id}', skipping"
                    )
                    continue

                prefill_map[sample_id] = response_text

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON, skipping - {e}")

    logger.info(f"Loaded {len(prefill_map)} prefill entries from {config.path}")

    async def solve(state: TaskState, generate: Solver):
        # Get prefill text for this sample
        if state.sample_id and state.sample_id in prefill_map:
            full_response = prefill_map[state.sample_id]
            prefill_text = get_prefill_fraction(full_response, fraction=config.fraction)

            # Inject vLLM prefill into config
            if state.config is None:
                state.config = GenerateConfig()

            extra = state.config.extra_body or {}
            extra["continue_final_message"] = prefill_text
            state.config.extra_body = extra

            logger.debug(
                f"Sample {state.sample_id}: Applying prefill "
                f"({len(prefill_text.split())} words, {config.fraction:.1%} of {len(full_response.split())} words)"
            )
        elif state.sample_id:
            logger.warning(
                f"Sample {state.sample_id}: No prefill found in {config.path}, skipping prefill"
            )
        else:
            logger.warning("No sample_id in state, skipping prefill")

        return await generate(state)

    return solve
