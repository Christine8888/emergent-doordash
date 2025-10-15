#!/usr/bin/env python3
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from sample_utils import sample_solution, load_solved_ids, run_sampling_loop, logger, create_base_parser, sample_to_dict
from environments.math.math import get_math_dataset
from environments.math.utils import extract_answer, is_equiv_sympy, normalize_final_answer


async def grade_answer(response: str, target: str) -> tuple[bool, str]:
    """Extract and grade answer using sympy exact match. Returns (is_correct, extracted_answer)."""
    extracted = extract_answer(response)
    if not extracted:
        return False, ""

    norm_answer = await normalize_final_answer(extracted)
    norm_target = await normalize_final_answer(target)

    correct = await is_equiv_sympy(norm_answer, norm_target)
    return correct, extracted


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
                    logger.info(f"  ANSWER: {extracted}")
                    logger.info(f"  TARGET: {target}")

            except Exception as e:
                logger.error({e})

        pbar.update(1)
        return None


async def main():
    parser = create_base_parser("Sample MATH problems until all have correct solutions")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/test/validation)")
    args = parser.parse_args()

    # Load math dataset
    logger.info(f"Loading MATH dataset (split={args.split})...")
    dataset = get_math_dataset(split=args.split, shuffle=False)

    # Convert to list for easier manipulation
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
        for sample in samples_to_solve
    ]

    await run_sampling_loop(tasks, output_path)
    pbar.close()


if __name__ == "__main__":
    asyncio.run(main())