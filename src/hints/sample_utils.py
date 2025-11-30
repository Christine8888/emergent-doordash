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


def load_solved_counts(output_path: Path) -> dict[str, int]:
    """Load count of existing samples per question.

    Returns:
        Dictionary mapping question ID to count of existing samples
    """
    if not output_path.exists():
        return {}

    solved_counts = {}
    with open(output_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("hint") and data["hint"].strip():
                    qid = data["id"]
                    solved_counts[qid] = solved_counts.get(qid, 0) + 1
            except:
                pass
    return solved_counts


def log_sample_statistics(solved_counts: dict[str, int], n_total_problems: int):
    """Log statistics about sample counts per problem."""
    from collections import Counter

    if not solved_counts:
        logger.info("No existing samples found")
        return

    count_distribution = Counter(solved_counts.values())

    logger.info(f"Sample statistics ({len(solved_counts)}/{n_total_problems} problems have samples):")
    for count in sorted(count_distribution.keys()):
        n_problems = count_distribution[count]
        logger.info(f"  {count} sample(s): {n_problems} problems")

    logger.info(f"  Total samples: {sum(solved_counts.values())}")


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


async def sample_problem(
    sample_id: str,
    question: str,
    prompt: str,
    target: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    max_attempts: int,
    n_target: int,
    start_idx: int,
    sem: asyncio.Semaphore,
    pbar: tqdm,
    eval_config,
    extra_fields: dict,
    response_to_hint: Callable[[str], any],
    extract_fn: Callable[[str], str] | None = None,
    rationalize: bool = False,
) -> list[dict]:
    """Sample a problem up to max_attempts times, collecting up to n_target correct samples.

    Args:
        sample_id: Sample identifier
        question: The question text
        prompt: Full prompt to send to model (may be modified if rationalize=True)
        target: Target answer
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        max_connections: Max concurrent connections
        max_attempts: Max total attempts for this problem
        n_target: Number of correct samples to collect
        start_idx: Starting sample_idx (based on existing count)
        sem: Semaphore for concurrency control
        pbar: Progress bar
        eval_config: Eval config module with grade_answer and extract_answer
        extra_fields: Additional fields to include in output
        response_to_hint: Function to convert response to hint (e.g., identity for CoT)
        extract_fn: Optional function to extract answer (overrides eval_config.extract_answer)
        rationalize: If True, prepend answer hint to prompt

    Returns:
        List of dictionaries with Example fields plus extra_fields
    """
    extract = extract_fn if extract_fn is not None else eval_config.extract_answer

    if rationalize:
        rationalize_hint = f"HINT: The answer is {target}. Please still produce a fully natural chain of thought and solution that does not reveal the answer early, but that reaches this answer after actually properly solving the problem in full detail. Do not mention this hint anywhere in your response.\n\n"
        prompt += "\n\n" + rationalize_hint

    results = []
    current_idx = start_idx

    async with sem:
        for attempt in range(max_attempts):
            if len(results) >= n_target:
                break

            try:
                response = await sample_solution(
                    prompt, model_id, temperature, max_tokens, max_connections
                )

                extracted = extract(response)
                correct = await eval_config.grade_answer(extracted, target)

                if correct:
                    hint = response_to_hint(response)

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
                        sample_idx=current_idx,
                        prompt=prompt,
                        metadata=metadata,
                    )
                    result = example.to_dict()
                    result.update(extra_fields)

                    results.append(result)
                    current_idx += 1
                    pbar.update(1)
                    logger.info(f"  {sample_id}[{current_idx - 1}]: collected ({len(results)}/{n_target})")
                else:
                    logger.info(f"  {sample_id} [attempt {attempt + 1}/{max_attempts}]: ANSWER: {extracted} | TARGET: {target}")

            except Exception as e:
                logger.error(f"  {sample_id}: {e}")

    return results


async def run_sampling_loop(tasks: list, output_path: Path):
    """Run all sampling tasks and write results as they complete."""
    file_lock = asyncio.Lock()

    for coro in asyncio.as_completed(tasks):
        results = await coro

        if results:
            async with file_lock:
                with open(output_path, "a") as f:
                    for result in results:
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

    # Load existing sample counts per question
    solved_counts = load_solved_counts(output_path)
    log_sample_statistics(solved_counts, len(all_samples))

    # Use provided format_fn or fall back to eval_config.format_prompt
    format_prompt = format_fn if format_fn is not None else eval_config.format_prompt

    # Create tasks for problems that need more samples
    tasks = []
    total_samples_needed = 0

    for sample in all_samples:
        existing_count = solved_counts.get(sample.id, 0)
        n_needed = args.n_per_question - existing_count

        if n_needed > 0:
            total_samples_needed += n_needed

            prompt = format_prompt(sample)
            if args.prompt_suffix:
                prompt = prompt + "\n\n" + args.prompt_suffix

            extra_fields = {}
            if hasattr(eval_config, "extract_sample_fields"):
                extra_fields = eval_config.extract_sample_fields(sample)

            tasks.append((sample, prompt, extra_fields, existing_count, n_needed))

    logger.info(f"Processing {total_samples_needed} samples across {len(tasks)} problems")
    logger.info(f"Target: {args.n_per_question} sample(s) per question")
    if args.rationalize:
        logger.info(f"Rationalize mode: ENABLED")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max concurrent: {args.max_concurrent}, "
                f"Max attempts per problem: {args.max_retries}")
    if args.prompt_suffix:
        logger.info(f"Prompt suffix: {args.prompt_suffix}")
    logger.info(f"Output: {output_path}\n")

    # Create async tasks
    sem = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=total_samples_needed, desc="Solving")

    async_tasks = []
    for sample, prompt, extra_fields, start_idx, n_needed in tasks:
        async_tasks.append(
            sample_problem(
                sample_id=sample.id,
                question=sample.input,
                prompt=prompt,
                target=sample.target,
                model_id=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_connections=args.max_concurrent,
                max_attempts=args.max_retries,
                n_target=n_needed,
                start_idx=start_idx,
                sem=sem,
                pbar=pbar,
                eval_config=eval_config,
                extra_fields=extra_fields,
                response_to_hint=response_to_hint,
                extract_fn=extract_fn,
                rationalize=args.rationalize,
            )
        )

    await run_sampling_loop(async_tasks, output_path)
    pbar.close()
    logger.info("Done!")
