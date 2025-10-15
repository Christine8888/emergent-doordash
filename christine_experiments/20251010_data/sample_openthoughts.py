#!/usr/bin/env python3
"""Sample OpenThoughts-114k dataset with Claude 4.5 Sonnet using Inspect AI."""

import asyncio
import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm.asyncio import tqdm
from inspect_ai.model import get_model, ChatMessageUser, GenerateConfig
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


async def process_example(example: dict, idx: int, model_id: str, temperature: float, max_tokens: int, max_connections: int, max_retries: int, output_path: Path, buffer: list, buffer_lock: asyncio.Lock, buffer_size: int, pbar: tqdm, sem: asyncio.Semaphore) -> bool:
    """Process a single example from the dataset."""
    async with sem:
        try:
            # Extract first user message (skip system prompt)
            user_messages = [msg for msg in example['conversations'] if msg['from'] == 'user']
            if not user_messages:
                pbar.update(1)
                return False

            user_prompt = user_messages[0]['value']
            sample_id = f"openthought_{idx}"

            # Generate completion - each call gets its own model instance via context manager
            messages = [ChatMessageUser(content=user_prompt)]
            config = GenerateConfig(
                temperature=temperature,
                max_tokens=max_tokens,
                max_connections=max_connections,
                max_retries=max_retries,
            )

            async with get_model(model_id, config=config) as model:
                response = await model.generate(input=messages)

            # Format result
            result = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response.completion},
                ],
                "metadata": {
                    "sample_id": sample_id,
                    "model": model_id,
                    "temperature": temperature,
                }
            }

            # Thread-safe buffer append
            async with buffer_lock:
                buffer.append(result)

                # Flush buffer if full
                if len(buffer) >= buffer_size:
                    with open(output_path, 'a') as f:
                        for item in buffer:
                            f.write(json.dumps(item) + '\n')
                    buffer.clear()

            pbar.update(1)
            return True

        except Exception as e:
            print(f"\nError processing example {idx}: {e}")
            pbar.update(1)
            return False


async def main():
    parser = argparse.ArgumentParser(description="Sample OpenThoughts with a model")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-5-20250929", help="Model ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    parser.add_argument("--max-concurrent", type=int, default=25, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Max API retries per request")
    parser.add_argument("--buffer-size", type=int, default=500, help="Write buffer size")

    args = parser.parse_args()

    # Load dataset
    print("Loading OpenThoughts-114k...")
    ds = load_dataset('open-thoughts/OpenThoughts-114k', split='train')

    # Setup output file and check for existing IDs
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if output_path.exists():
        print("Checking for existing sample IDs...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data['metadata']['sample_id'])
                except:
                    pass
        print(f"Found {len(existing_ids)} existing samples, skipping those\n")

    # Count total to process
    total_examples = len(ds) if args.limit is None else min(args.limit, len(ds))
    examples_to_process = [idx for idx in range(total_examples) if f"openthought_{idx}" not in existing_ids]

    print(f"Processing {len(examples_to_process)} examples (skipping {total_examples - len(examples_to_process)} existing)")
    print(f"Model: {args.model}, Temp: {args.temperature}, Max tokens: {args.max_tokens}")
    print(f"Max concurrent: {args.max_concurrent}, Max retries: {args.max_retries}, Buffer: {args.buffer_size}")
    print(f"Output: {args.output_file}\n")

    # Shared buffer and synchronization primitives
    buffer = []
    buffer_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.max_concurrent)

    pbar = tqdm(total=len(examples_to_process), desc="Sampling")

    # Create task for each example - semaphore limits concurrent tasks
    tasks = []
    for idx in examples_to_process:
        task = process_example(
            example=ds[idx],
            idx=idx,
            model_id=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_connections=args.max_concurrent,
            max_retries=args.max_retries,
            output_path=output_path,
            buffer=buffer,
            buffer_lock=buffer_lock,
            buffer_size=args.buffer_size,
            pbar=pbar,
            sem=sem,
        )
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Write remaining buffer
    if buffer:
        with open(output_path, 'a') as f:
            for item in buffer:
                f.write(json.dumps(item) + '\n')

    pbar.close()
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
