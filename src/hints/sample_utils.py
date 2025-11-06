"""Shared utilities for sampling scripts."""

import asyncio
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
from inspect_ai.model import get_model, ChatMessageUser, GenerateConfig
from tqdm.asyncio import tqdm

from evals.example import Example
from utils.setup import setup_env, setup_logging

setup_env()
logger = setup_logging()


def create_base_parser(description: str) -> ArgumentParser:
    """Create argument parser with common arguments for sampling scripts."""
    parser = ArgumentParser(description=description)
    parser.add_argument("--eval", type=str, required=True,
                        choices=["gpqa", "aime", "math", "hle", "arc"],
                        help="Eval name")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str,
                        default="anthropic/claude-sonnet-4-5-20250929",
                        help="Model ID")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=25,
                        help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Max retries per problem")
    parser.add_argument("--prompt-suffix", type=str, default=None,
                        help="Additional text to append to prompt")
    return parser


def load_solved_ids(output_path: Path) -> set[str]:
    """Load IDs that already have solutions."""
    if not output_path.exists():
        return set()

    solved_ids = set()
    with open(output_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                solved_ids.add(data["id"])
            except:
                pass
    return solved_ids


async def sample_solution(
    prompt: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
) -> str:
    """Sample a single solution from the model."""
    messages = [ChatMessageUser(content=prompt)]
    config = GenerateConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        max_connections=max_connections
    )

    async with get_model(model_id, config=config) as model:
        response = await model.generate(input=messages)

    return response.completion


async def sample_until_correct(
    sample_id: str,
    question: str,
    prompt: str,
    target: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    max_retries: int,
    sem: asyncio.Semaphore,
    pbar: tqdm,
    eval_config,
    extra_fields: dict,
    response_to_hint: Callable[[str], any],
    extract_fn: Callable[[str], str] | None = None,
) -> dict | None:
    """Sample until correct answer or max_retries exceeded.

    Args:
        sample_id: Sample identifier
        question: The question text
        prompt: Full prompt to send to model
        target: Target answer
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        max_connections: Max concurrent connections
        max_retries: Max attempts per problem
        sem: Semaphore for concurrency control
        pbar: Progress bar
        eval_config: Eval config module with grade_answer and extract_answer
        extra_fields: Additional fields to include in output
        response_to_hint: Function to convert response to hint (e.g., identity for CoT)
        extract_fn: Optional function to extract answer (overrides eval_config.extract_answer)

    Returns:
        Dictionary with Example fields plus extra_fields, or None if failed
    """
    # Use provided extract_fn or fall back to eval_config.extract_answer
    extract = extract_fn if extract_fn is not None else eval_config.extract_answer

    async with sem:
        for attempt in range(max_retries):
            try:
                response = await sample_solution(
                    prompt, model_id, temperature, max_tokens, max_connections
                )

                # Extract answer from response
                extracted = extract(response)

                # Grade the extracted answer
                correct = await eval_config.grade_answer(extracted, target)

                if correct:
                    # Create Example with hint from response_to_hint function
                    example = Example(
                        id=sample_id,
                        question=question,
                        target=target,
                        response=response,
                        hint=response_to_hint(response),
                        prompt=prompt,
                    )
                    # Add extra fields to dict
                    result = example.to_dict()
                    result.update(extra_fields)

                    pbar.update(1)
                    return result
                else:
                    logger.debug(f"  {sample_id}: got {extracted}, want {target}")

            except Exception as e:
                logger.error(f"  {sample_id}: {e}")

        pbar.update(1)
        return None


async def run_sampling_loop(tasks: list, output_path: Path):
    """Run all sampling tasks and write results as they complete."""
    file_lock = asyncio.Lock()

    for coro in asyncio.as_completed(tasks):
        result = await coro

        if result:
            async with file_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")


async def collect_samples(
    args,
    response_to_hint: Callable[[str], any],
    format_fn: Callable[[any], str] | None = None,
    extract_fn: Callable[[str], str] | None = None,
):
    """Collect samples by sampling until all have correct solutions.

    Args:
        args: Parsed command-line arguments (from create_base_parser)
        response_to_hint: Function to convert response to hint
        format_fn: Optional function to format prompt (overrides eval_config.format_prompt)
        extract_fn: Optional function to extract answer (overrides eval_config.extract_answer)
    """
    import importlib
    from utils.setup import setup_logging

    logger = setup_logging()

    # Load eval config
    eval_module = importlib.import_module(f"environments.{args.eval}.config")
    eval_config = eval_module

    # Add eval-specific args if available
    if hasattr(eval_config, "add_cli_args"):
        # Re-parse with eval-specific args
        parser = create_base_parser("Sample eval problems until all have correct solutions")
        eval_config.add_cli_args(parser)
        args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading {args.eval.upper()} dataset...")
    dataset_kwargs = {}
    if hasattr(eval_config, "get_dataset_kwargs"):
        dataset_kwargs = eval_config.get_dataset_kwargs(args)
    dataset = eval_config.get_dataset(**dataset_kwargs)
    all_samples = list(dataset)

    logger.info(f"Loaded {len(all_samples)} problems")

    # Setup output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solved_ids = load_solved_ids(output_path)
    samples_to_solve = [s for s in all_samples if s.id not in solved_ids]

    logger.info(f"Processing {len(samples_to_solve)} problems "
                f"(skipping {len(solved_ids)} existing)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max concurrent: {args.max_concurrent}, "
                f"Max retries: {args.max_retries}")
    if args.prompt_suffix:
        logger.info(f"Prompt suffix: {args.prompt_suffix}")
    logger.info(f"Output: {output_path}\n")

    # Use provided format_fn or fall back to eval_config.format_prompt
    format_prompt = format_fn if format_fn is not None else eval_config.format_prompt

    # Create tasks
    sem = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=len(samples_to_solve), desc="Solving")

    tasks = []
    for sample in samples_to_solve:
        prompt = format_prompt(sample)

        if args.prompt_suffix:
            prompt = prompt + "\n\n" + args.prompt_suffix

        # Get extra fields from eval config
        extra_fields = {}
        if hasattr(eval_config, "extract_sample_fields"):
            extra_fields = eval_config.extract_sample_fields(sample)

        tasks.append(
            sample_until_correct(
                sample_id=sample.id,
                question=sample.input,
                prompt=prompt,
                target=sample.target,
                model_id=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_connections=args.max_concurrent,
                max_retries=args.max_retries,
                sem=sem,
                pbar=pbar,
                eval_config=eval_config,
                extra_fields=extra_fields,
                response_to_hint=response_to_hint,
                extract_fn=extract_fn,
            )
        )

    await run_sampling_loop(tasks, output_path)
    pbar.close()
    logger.info("Done!")
