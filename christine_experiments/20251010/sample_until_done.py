#!/usr/bin/env python3
"""Sample problems until all have correct solutions."""

import asyncio
import argparse
import json
from pathlib import Path
from tqdm.asyncio import tqdm
from inspect_ai.model import get_model, ChatMessageUser, GenerateConfig
from dotenv import load_dotenv
import logging

from environments.math.math import get_math_dataset
from environments.math.utils import extract_answer, is_equiv_sympy, normalize_final_answer

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO) 
load_dotenv(Path(__file__).parent.parent.parent / ".env")


async def sample_solution(prompt: str, model_id: str, temperature: float, max_tokens: int, max_connections: int) -> str:
    """Sample a solution from the model."""
    messages = [ChatMessageUser(content=prompt)]
    config = GenerateConfig(temperature=temperature, max_tokens=max_tokens, max_connections=max_connections)

    async with get_model(model_id, config=config) as model:
        response = await model.generate(input=messages)

    return response.completion


async def grade_answer(response: str, target: str) -> tuple[bool, str]:
    """Extract and grade answer using sympy exact match. Returns (is_correct, extracted_answer)."""
    extracted = extract_answer(response)
    if not extracted:
        return False, ""

    norm_answer = await normalize_final_answer(extracted)
    norm_target = await normalize_final_answer(target)

    correct = await is_equiv_sympy(norm_answer, norm_target)
    return correct, extracted


def sample_to_dict(sample_id: str, input_text: str, target: str, model: str, response: str) -> dict:
    """Convert sample to output dict format matching input JSONL."""
    return {
        "id": sample_id,
        "full_prompt": input_text,
        "question_with_choices": input_text,  # Same for math
        "question": input_text,
        "model": model,
        "response": response,
        "target": target,
        "choices": None,
        "score": "C",
    }


async def sample_until_correct(
    sample_id: str,
    input_text: str,
    target: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    max_retries: int,
    sem: asyncio.Semaphore,
    pbar: tqdm,
) -> dict | None:
    """Sample until correct answer or max_retries exceeded. Returns result dict or None."""
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await sample_solution(input_text, model_id, temperature, max_tokens, max_connections)
                correct, extracted = await grade_answer(response, target)

                if correct:
                    result = sample_to_dict(sample_id, input_text, target, model_id, response)
                    pbar.update(1)
                    return result
                else:
                    logger.info(f"\n[{sample_id}] Attempt {attempt + 1}/{max_retries} - INCORRECT")
                    logger.info(f"  ANSWER: {extracted}")
                    logger.info(f"  TARGET: {target}")

            except Exception as e:
                logger.error(f"\n[{sample_id}] Error on attempt {attempt + 1}: {e}")

        # Max retries exceeded
        logger.info(f"\n[{sample_id}] Max retries exceeded, discarding")
        pbar.update(1)
        return None


async def main():
    parser = argparse.ArgumentParser(description="Sample problems until all have correct solutions")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL for solved problems")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/test/validation)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-5-20250929", help="Model ID")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=25, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries per problem")
    parser.add_argument("--buffer-size", type=int, default=1, help="Write buffer size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems to process")

    args = parser.parse_args()

    # Load math dataset
    logger.info(f"Loading MATH dataset (split={args.split})...")
    dataset = get_math_dataset(split=args.split, shuffle=False)

    # Convert to list for easier manipulation
    all_samples = list(dataset)
    logger.info(all_samples[0])
    if args.limit:
        all_samples = all_samples[:args.limit]

    logger.info(f"Loaded {len(all_samples)} problems from dataset")

    # Check output file for already solved IDs
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    samples_to_solve = [s for s in all_samples if s.id not in solved_ids]

    logger.info(f"Processing {len(samples_to_solve)} problems (skipping {len(all_samples) - len(samples_to_solve)} existing)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    logger.info(f"Max concurrent: {args.max_concurrent}, Max retries: {args.max_retries}")
    logger.info(f"Output: {output_path}\n")

    # Shared state
    buffer = []
    buffer_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=len(samples_to_solve), desc="Solving")

    # Process all problems concurrently
    tasks = []
    for sample in samples_to_solve:
        task = sample_until_correct(
            sample_id=sample.id,
            input_text=sample.input,
            target=sample.target,
            model_id=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_connections=args.max_concurrent,
            max_retries=args.max_retries,
            sem=sem,
            pbar=pbar,
        )
        tasks.append(task)

    # Gather results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro

        if result:
            async with buffer_lock:
                buffer.append(result)

                # Flush buffer when full
                if len(buffer) >= args.buffer_size:
                    with open(output_path, 'a') as f:
                        for item in buffer:
                            f.write(json.dumps(item) + '\n')
                    buffer.clear()

    # Write remaining buffer
    if buffer:
        with open(output_path, 'a') as f:
            for item in buffer:
                f.write(json.dumps(item) + '\n')

    pbar.close()
    logger.info(f"\n✓ Completed. Solved problems appended to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
