#!/usr/bin/env python3
"""
Unified script for sampling eval problems until all have correct solutions.

Usage:
    python sample_until_done.py --eval gpqa --output-file results.jsonl
    python sample_until_done.py --eval aime --output-file results.jsonl
    python sample_until_done.py --eval math --output-file results.jsonl --split train
"""
import asyncio
import importlib
from pathlib import Path
from tqdm.asyncio import tqdm

from sample_utils import (
    sample_solution,
    load_solved_ids,
    run_sampling_loop,
    logger,
    create_base_parser,
    sample_to_dict,
)


async def sample_until_correct(
    sample_id: str,
    sample_input: str,
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
    sample_fields: dict,
) -> dict | None:
    """Sample until correct answer or max_retries exceeded."""
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await sample_solution(
                    prompt, model_id, temperature, max_tokens, max_connections
                )
                correct = await eval_config.grade_answer(response, target)

                if correct:
                    result = sample_to_dict(
                        sample_id, prompt, target, model_id, response,
                        question=sample_input, **sample_fields
                    )
                    pbar.update(1)
                    return result
                else:
                    extracted = eval_config.extract_answer(response)
                    logger.info(f"  EXTRACTED: {extracted}")
                    logger.info(f"  TARGET: {target}\n")

            except Exception as e:
                logger.error({e})

        pbar.update(1)
        return None


async def main():
    parser = create_base_parser(
        "Sample eval problems until all have correct solutions"
    )
    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        choices=["gpqa", "aime", "math"],
        help="Eval name (gpqa, aime, math)",
    )
    args, unknown_args = parser.parse_known_args()

    # Dynamically import eval config
    eval_module = importlib.import_module(f"environments.{args.eval}.config")
    eval_config = eval_module

    # Add eval-specific CLI arguments
    if hasattr(eval_config, "add_cli_args"):
        eval_config.add_cli_args(parser)
        args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading {args.eval.upper()} dataset...")
    dataset_kwargs = {}
    if hasattr(eval_config, "get_dataset_kwargs"):
        dataset_kwargs = eval_config.get_dataset_kwargs(args)
    dataset = eval_config.get_dataset(**dataset_kwargs)
    all_samples = list(dataset)

    logger.info(f"Loaded {len(all_samples)} problems from dataset")

    # Check output file for already solved IDs
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solved_ids = load_solved_ids(output_path)
    samples_to_solve = [s for s in all_samples if s.id not in solved_ids]

    logger.info(
        f"Processing {len(samples_to_solve)} problems (skipping {len(all_samples) - len(samples_to_solve)} existing)"
    )
    logger.info(f"Model: {args.model}")
    logger.info(
        f"Max concurrent: {args.max_concurrent}, Max retries: {args.max_retries}"
    )
    logger.info(f"Output: {output_path}\n")

    sem = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=len(samples_to_solve), desc="Solving")

    tasks = []
    for sample in samples_to_solve:
        prompt = eval_config.format_prompt(sample)
        sample_fields = eval_config.extract_sample_fields(sample)

        tasks.append(
            sample_until_correct(
                sample_id=sample.id,
                sample_input=sample.input,
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
                sample_fields=sample_fields,
            )
        )

    await run_sampling_loop(tasks, output_path)
    pbar.close()


if __name__ == "__main__":
    asyncio.run(main())
