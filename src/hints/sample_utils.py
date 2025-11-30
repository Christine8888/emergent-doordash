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
    parser.add_argument("--max-tokens", type=int, default=32000,
                        help="Max tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=25,
                        help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Max retries per problem")
    parser.add_argument("--n-per-question", type=int, default=1,
                        help="Number of correct samples to collect per question")
    parser.add_argument("--rationalize", action="store_true",
                        help="Add answer hint to prompt (rationalize mode)")
    parser.add_argument("--prompt-suffix", type=str, default=None,
                        help="Additional text to append to prompt")
    return parser


def load_solved_ids(output_path: Path) -> dict[str, set[int]]:
    """Load IDs and track which sample_idx values exist per question.

    Returns:
        Dictionary mapping question ID to set of existing sample_idx values
    """
    if not output_path.exists():
        return {}

    solved_indices = {}
    with open(output_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                # Only track if hint field exists and is non-empty
                if data.get("hint") and data["hint"].strip():
                    qid = data["id"]
                    sample_idx = data.get("sample_idx", 0)

                    if qid not in solved_indices:
                        solved_indices[qid] = set()
                    solved_indices[qid].add(sample_idx)
            except:
                pass
    return solved_indices


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
    sample_idx: int = 0,
    rationalize: bool = False,
) -> dict | None:
    """Sample until correct answer or max_retries exceeded.

    Args:
        sample_id: Sample identifier
        question: The question text
        prompt: Full prompt to send to model (may be modified if rationalize=True)
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
        sample_idx: Index for this sample (for multiple samples per question)
        rationalize: If True, prepend answer hint to prompt

    Returns:
        Dictionary with Example fields plus extra_fields, or None if failed
    """
    # Use provided extract_fn or fall back to eval_config.extract_answer
    extract = extract_fn if extract_fn is not None else eval_config.extract_answer

    # Add rationalize hint to prompt if enabled
    if rationalize:
        rationalize_hint = f"HINT: The answer is {target}. Please still produce a fully natural chain of thought and solution that does not reveal the answer early, but that reaches this answer after actually properly solving the problem in full detail. Do not mention this hint anywhere in your response.\n\n"
        prompt += "\n\n" + rationalize_hint

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
                    hint = response_to_hint(response)

                    # Only save if hint is non-empty
                    if not hint or not hint.strip():
                        logger.warning(f"  {sample_id}: Answer correct but hint extraction failed, retrying")
                        continue

                    metadata = {"model": model_id}
                    if rationalize:
                        metadata["rationalize"] = True

                    example = Example(
                        id=sample_id,
                        question=question,
                        target=target,
                        response=response,
                        hint=hint,
                        sample_idx=sample_idx,
                        prompt=prompt,
                        metadata=metadata,
                    )
                    # Add extra fields to dict
                    result = example.to_dict()
                    result.update(extra_fields)

                    pbar.update(1)
                    return result
                else:
                    logger.info(f"  {sample_id} [attempt {attempt + 1}/{max_retries}]: ANSWER: {extracted} | TARGET: {target}")

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

    # Load existing sample indices per question
    solved_indices = load_solved_ids(output_path)

    # Use provided format_fn or fall back to eval_config.format_prompt
    format_prompt = format_fn if format_fn is not None else eval_config.format_prompt

    # Create tasks for all samples that need more solutions
    # Each question may need multiple samples (up to n_per_question)
    tasks = []
    total_samples_needed = 0

    for sample in all_samples:
        existing_indices = solved_indices.get(sample.id, set())

        # Find missing sample indices (fill gaps and add new ones)
        missing_indices = []
        for idx in range(args.n_per_question):
            if idx not in existing_indices:
                missing_indices.append(idx)

        if missing_indices:
            total_samples_needed += len(missing_indices)

            # Create tasks for each missing sample index
            for sample_idx in missing_indices:
                prompt = format_prompt(sample)

                if args.prompt_suffix:
                    prompt = prompt + "\n\n" + args.prompt_suffix

                # Get extra fields from eval config
                extra_fields = {}
                if hasattr(eval_config, "extract_sample_fields"):
                    extra_fields = eval_config.extract_sample_fields(sample)

                tasks.append((sample, prompt, extra_fields, sample_idx))

    logger.info(f"Processing {total_samples_needed} samples across {len(all_samples)} problems")
    logger.info(f"Target: {args.n_per_question} sample(s) per question")
    if args.rationalize:
        logger.info(f"Rationalize mode: ENABLED")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max concurrent: {args.max_concurrent}, "
                f"Max retries: {args.max_retries}")
    if args.prompt_suffix:
        logger.info(f"Prompt suffix: {args.prompt_suffix}")
    logger.info(f"Output: {output_path}\n")

    # Create async tasks
    sem = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=total_samples_needed, desc="Solving")

    async_tasks = []
    for sample, prompt, extra_fields, sample_idx in tasks:
        async_tasks.append(
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
                sample_idx=sample_idx,
                rationalize=args.rationalize,
            )
        )

    await run_sampling_loop(async_tasks, output_path)
    pbar.close()
    logger.info("Done!")
