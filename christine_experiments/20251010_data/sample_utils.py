"""Shared utilities for sample_until_done scripts."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from inspect_ai.model import get_model, ChatMessageUser, GenerateConfig
from tqdm.asyncio import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common arguments for sampling scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL for solved problems")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-5-20250929", help="Model ID")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=25, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries per problem")
    return parser


async def sample_solution(prompt: str, model_id: str, temperature: float, max_tokens: int, max_connections: int) -> str:
    """Sample a solution from the model."""
    messages = [ChatMessageUser(content=prompt)]
    config = GenerateConfig(temperature=temperature, max_tokens=max_tokens, max_connections=max_connections)

    async with get_model(model_id, config=config) as model:
        response = await model.generate(input=messages)

    return response.completion


def load_solved_ids(output_path: Path) -> set[str]:
    """Load already solved IDs from output file."""
    solved_ids = set()
    if output_path.exists():
        logger.info("Checking for existing solutions...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    solved_ids.add(data['id'])
                except:
                    pass
        logger.info(f"Found {len(solved_ids)} existing solutions\n")
    return solved_ids


def sample_to_dict(
    sample_id: str,
    full_prompt: str,
    target: str,
    model: str,
    response: str,
    question: str | None = None,
    **extra_fields
) -> dict:
    """Convert sample to output dict format.

    Args:
        sample_id: Sample ID
        full_prompt: Full formatted prompt sent to model
        target: Target answer
        model: Model ID
        response: Model response
        question: Raw question text (defaults to full_prompt if not provided)
        **extra_fields: Additional fields to include (e.g., choices, question_with_choices)
    """
    result = {
        "id": sample_id,
        "full_prompt": full_prompt,
        "question": question or full_prompt,
        "target": target,
        "model": model,
        "response": response,
        "score": "C",
    }

    # Add any extra fields
    result.update(extra_fields)

    return result


async def run_sampling_loop(
    tasks: list,
    output_path: Path,
) -> None:
    """Run async sampling loop and write results as they complete.

    Args:
        tasks: List of coroutines to run
        output_path: Path to write results
    """
    file_lock = asyncio.Lock()

    for coro in asyncio.as_completed(tasks):
        result = await coro

        if result:
            async with file_lock:
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
