import random
import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import anthropic

ANTHROPIC_CLIENT = anthropic.Anthropic()

ANTHROPIC_MODELS: List[str] = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]

input_token_costs = {
    "claude-sonnet-4-6": 3.0 / 1_000_000,
    "claude-opus-4-6": 5.0 / 1_000_000,
}

output_token_costs = {
    "claude-sonnet-4-6": 15.0 / 1_000_000,
    "claude-opus-4-6": 25.0 / 1_000_000,
}


@dataclass
class QueryResult:
    response_text: str
    input_token_count: int
    output_token_count: int
    is_error: bool
    cost: float = 0.0


def cost_calculator(model: str, input_token_count: int, output_token_count: int) -> float:
    if model not in input_token_costs or model not in output_token_costs:
        return 0.0
    return input_token_costs[model] * input_token_count + output_token_costs[model] * output_token_count


def parse_anthropic_response(response: anthropic.types.Message, model: str) -> QueryResult:
    texts: List[str] = []
    for block in response.content:
        if hasattr(block, "text") and isinstance(block.text, str):
            texts.append(block.text)

    input_token_count = response.usage.input_tokens
    output_token_count = response.usage.output_tokens

    return QueryResult(
        response_text="".join(texts),
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        is_error=False,
        cost=cost_calculator(model, input_token_count, output_token_count),
    )


def single_query_anthropic(prompt: str, model: str, max_tokens: int = 16000) -> QueryResult:
    messages = [{"role": "user", "content": prompt}]
    with ANTHROPIC_CLIENT.messages.stream(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    ) as stream:
        for _ in stream.text_stream:
            pass
        final_message = stream.get_final_message()
    return parse_anthropic_response(final_message, model)


def query_anthropic_with_retry(prompt: str, model: str, max_tokens: int = 16000) -> QueryResult:
    max_retries = 5
    base_delay = 5.0

    for attempt in range(max_retries):
        try:
            return single_query_anthropic(prompt, model, max_tokens)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + (random.random() * 0.5)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Giving up.")
                return QueryResult(
                    response_text=f'Error: Max retries reached for prompt starting with "{prompt[:50]}..."',
                    input_token_count=0,
                    output_token_count=0,
                    is_error=True,
                )

    raise ValueError("You should never get here.")


def query_model_anthropic_batch(
    prompts: List[str],
    model: str,
    max_tokens: int,
    poll_interval_seconds: int = 30,
) -> List[QueryResult]:
    """Submit prompts to Anthropic's Message Batches API and poll until complete.

    50% cheaper than standard API calls and avoids streaming timeout issues.
    Results are returned in the same order as the input prompts.
    """
    requests = [
        {
            "custom_id": f"prompt-{i}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for i, prompt in enumerate(prompts)
    ]

    batch = ANTHROPIC_CLIENT.messages.batches.create(requests=requests)
    print(f"[ANTHROPIC BATCH] submitted batch_id={batch.id}, n={len(prompts)}")

    while True:
        batch = ANTHROPIC_CLIENT.messages.batches.retrieve(batch.id)
        print(
            f"[ANTHROPIC BATCH] status={batch.processing_status}, "
            f"succeeded={batch.request_counts.succeeded}, "
            f"errored={batch.request_counts.errored}, "
            f"processing={batch.request_counts.processing}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval_seconds)

    id_to_result: dict = {}
    for result in ANTHROPIC_CLIENT.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            id_to_result[result.custom_id] = parse_anthropic_response(
                result.result.message, model
            )
        else:
            id_to_result[result.custom_id] = QueryResult(
                response_text=f"Error: {result.result.error}",
                input_token_count=0,
                output_token_count=0,
                is_error=True,
            )

    return [id_to_result[f"prompt-{i}"] for i in range(len(prompts))]
