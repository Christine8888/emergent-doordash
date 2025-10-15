#!/usr/bin/env python3
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm

from sample_utils import sample_solution, load_solved_ids, run_sampling_loop, logger, create_base_parser, sample_to_dict
from inspect_ai.dataset import hf_dataset
from environments.aime.aime import record_to_sample
from environments.math.utils import extract_answer, normalize_final_answer, is_equiv_sympy

INSTRUCTIONS = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.
""".strip()


def get_aime_dataset():
    """Load AIME dataset from HuggingFace."""
    dataset = hf_dataset(
        path="di-zhang-fdu/AIME_1983_2024",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
    )
    return dataset


async def grade_answer(response: str, target: str) -> bool:
    """Grade AIME answer using sympy-based exact matching."""
    extracted = extract_answer(response)
    if not extracted:
        return False

    norm_answer = await normalize_final_answer(extracted)
    norm_target = await normalize_final_answer(target)

    # Try sympy first
    correct = await is_equiv_sympy(norm_answer, norm_target)

    return correct


def format_prompt(question: str) -> str:
    """Format AIME question with instructions."""
    return f"{INSTRUCTIONS}\n\nPROBLEM:\n{question}\n\nSOLUTION:"


async def sample_until_correct(
    sample_id: str,
    question: str,
    target: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    max_retries: int,
    sem: asyncio.Semaphore,
    pbar: tqdm,
) -> dict | None:
    """Sample until correct answer or max_retries exceeded."""
    async with sem:
        prompt = format_prompt(question)

        for attempt in range(max_retries):
            try:
                response = await sample_solution(prompt, model_id, temperature, max_tokens, max_connections)
                correct = await grade_answer(response, target)

                if correct:
                    result = sample_to_dict(sample_id, prompt, target, model_id, response, question=question)
                    pbar.update(1)
                    return result
                else:
                    extracted = extract_answer(response)
                    logger.info(f"  EXTRACTED: {extracted}")
                    logger.info(f"  RESPONSE: {response}")
                    logger.info(f"  TARGET: {target}")

            except Exception as e:
                logger.error({e})

        pbar.update(1)
        return None


async def main():
    parser = create_base_parser("Sample AIME problems until all have correct solutions")
    args = parser.parse_args()

    logger.info("Loading AIME dataset...")
    dataset = get_aime_dataset()
    all_samples = list(dataset)

    logger.info(f"Loaded {len(all_samples)} problems from dataset")

    # Check output file for already solved IDs
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solved_ids = load_solved_ids(output_path)
    samples_to_solve = [s for s in all_samples if s.id not in solved_ids]

    logger.info(f"Processing {len(samples_to_solve)} problems (skipping {len(all_samples) - len(samples_to_solve)} existing)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max concurrent: {args.max_concurrent}, Max retries: {args.max_retries}")
    logger.info(f"Output: {output_path}\n")

    sem = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=len(samples_to_solve), desc="Solving")

    tasks = [
        sample_until_correct(
            sample_id=sample.id,
            question=sample.input,
            target=sample.target,
            model_id=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_connections=args.max_concurrent,
            max_retries=args.max_retries,
            sem=sem,
            pbar=pbar,
        )
        for sample in samples_to_solve
    ]

    await run_sampling_loop(tasks, output_path)
    pbar.close()


if __name__ == "__main__":
    asyncio.run(main())
