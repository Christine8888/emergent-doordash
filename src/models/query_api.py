import random  # Import random for jitter calculation
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, cast

import anthropic
import openai
import tqdm
from google import genai
from google.genai import types
import dotenv
from dataclasses import dataclass

@dataclass
class QueryResult:
    response_text: str
    input_token_count: int
    output_token_count: int
    is_error: bool
    cost: float = 0.0  # We default to 0 as we normally run locally


def export():
    dotenv.load_dotenv(override=True)

# Load up environment variables
export()

DEBUG_QUERIES = True  # Set to False to disable printing outgoing request payloads

GEMINI_CLIENT = genai.Client(vertexai=True, project="hai-gcp-new-models", location="us-central1")

OPENAI_CLIENT = openai.OpenAI()

ANTHROPIC_CLIENT = anthropic.Anthropic()

GOOGLE_MODELS: List[str] = [
    "gemini-2.0-flash-001",
]

OPENAI_MODELS: List[str] = [
    "gpt-4.1-mini-2025-04-14",
    "gpt-3.5-turbo-0125",
    "gpt-4-0613",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "o1-preview-2024-09-12",
    "o1-2024-12-17",
    "gpt-4.1-2025-04-14",
    "o3-2025-04-16",
    "gpt-5-2025-08-07",
    "gpt-5-nano-2025-08-07", # for testign code; low cost
]

ANTHROPIC_MODELS: List[str] = [
    "claude-3-5-sonnet-latest",
    "claude-3-opus-latest",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
]

input_token_costs = {
    "gpt-4.1-mini-2025-04-14": 0.4 / 1_000_000,
    "claude-3-5-sonnet-latest": 3.0 / 1_000_000,
    "claude-3-opus-latest": 15.0 / 1_000_000,
    "claude-sonnet-4-5-20250929": 3 / 1_000_000,
    "claude-opus-4-1-20250805": 15 / 1_000_000,
    "gpt-3.5-turbo-0125": 0.5 / 1_000_000,
    "gpt-4-0613": 30 / 1_000_000,
    "gpt-4o-2024-05-13": 5 / 1_000_000,
    "gpt-4-turbo-2024-04-09": 10 / 1_000_000,
    "o1-preview-2024-09-12": 15 / 1_000_000,
    "o1-2024-12-17": 15 / 1_000_000,
    "gpt-4.1-2025-04-14": 2 / 1_000_000,
    "o3-2025-04-16": 2 / 1_000_000,
    "gpt-5-2025-08-07": 1.25 / 1_000_000,
    "gpt-5-nano-2025-08-07": 0.05 / 1_000_000,
}

output_token_costs = {
    "gpt-4.1-mini-2025-04-14": 0.1 / 1_000_000,
    "claude-3-5-sonnet-latest": 15.0 / 1_000_000,
    "claude-3-opus-latest": 75.0 / 1_000_000,
    "claude-sonnet-4-5-20250929": 15.0 / 1_000_000,
    "claude-opus-4-1-20250805": 75.0 / 1_000_000,
    "gpt-3.5-turbo-0125": 1.5 / 1_000_000,
    "gpt-4-0613": 60 / 1_000_000,
    "gpt-4o-2024-05-13": 15 / 1_000_000,
    "gpt-4-turbo-2024-04-09": 30 / 1_000_000,
    "o1-preview-2024-09-12": 60 / 1_000_000,
    "o1-2024-12-17": 60 / 1_000_000,
    "gpt-4.1-2025-04-14": 8 / 1_000_000,
    "o3-2025-04-16": 8 / 1_000_000,
    "gpt-5-2025-08-07": 10 / 1_000_000,
    "gpt-5-nano-2025-08-07": 0.4 / 1_000_000,
}


def cost_calculator(model: str, input_token_count: int, output_token_count: int) -> float:
    if model not in input_token_costs or model not in output_token_costs:
        raise ValueError(f"Model {model} not supported for cost caclulation.")

    input_cost = input_token_costs[model] * input_token_count
    output_cost = output_token_costs[model] * output_token_count

    return input_cost + output_cost


def parse_google_response(response: types.GenerateContentResponse, model: str) -> QueryResult:
    if response.text is None:
        response.text = ""

    input_token_count = response.usage_metadata.prompt_token_count
    output_token_count = response.usage_metadata.total_token_count - input_token_count

    output = QueryResult(
        response_text=response.text,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        is_error=False,
        cost=cost_calculator(model, input_token_count, output_token_count),
    )

    return output


def parse_openai_response(response: openai.ChatCompletion, model: str) -> QueryResult:
    choices = response.choices
    assert len(choices) == 1

    choice = choices[0]
    usage = response.usage
    input_token_count = usage.prompt_tokens
    output_token_count = usage.total_tokens - input_token_count

    outputs_str: str = choice.message.content

    return QueryResult(
        response_text=outputs_str,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        is_error=False,
        cost=cost_calculator(model, input_token_count, output_token_count),
    )


def parse_anthropic_response(response: anthropic.types.Message, model: str) -> QueryResult:
    texts: List[str] = []
    for block in response.content:
        if hasattr(block, "text") and isinstance(block.text, str):
            texts.append(block.text)
    response_text = "".join(texts)

    input_token_count = response.usage.input_tokens
    output_token_count = response.usage.output_tokens

    return QueryResult(
        response_text=response_text,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        is_error=False,
        cost=cost_calculator(model, input_token_count, output_token_count),
    )


def single_query_google(prompt: str, model: str) -> QueryResult:
    response = GEMINI_CLIENT.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
            max_output_tokens=16000,
        ),
    )
    return parse_google_response(response, model)


def single_query_openai(prompt: str, model: str) -> QueryResult:
    messages = [{"role": "user", "content": prompt}]
    if DEBUG_QUERIES:
        print(f"[OPENAI] model={model}")
        print(f"[OPENAI] messages={messages}")
    start_time = time.time()
    batch_response = OPENAI_CLIENT.chat.completions.create(model=model, messages=messages)
    elapsed_ms = (time.time() - start_time) * 1000.0
    if DEBUG_QUERIES:
        print(f"[OPENAI] latency_ms={elapsed_ms:.0f}")

    return parse_openai_response(batch_response, model)


def single_query_anthropic(prompt: str, model: str) -> QueryResult:
    messages = [{"role": "user", "content": prompt}]
    if DEBUG_QUERIES:
        print(f"[ANTHROPIC] model={model}")
        print(f"[ANTHROPIC] messages={messages}")
        print(f"[ANTHROPIC] max_tokens=16000 (streaming)")
    start_time = time.time()
    with ANTHROPIC_CLIENT.messages.stream(
        model=model,
        messages=messages,
        max_tokens=16000,
    ) as stream:
        for _ in stream.text_stream:
            pass
        final_message = stream.get_final_message()
    elapsed_ms = (time.time() - start_time) * 1000.0
    if DEBUG_QUERIES:
        print(f"[ANTHROPIC] latency_ms={elapsed_ms:.0f}")
    return parse_anthropic_response(final_message, model)


def query_model_api(prompt: str, model: str) -> QueryResult:
    max_retries = 5
    base_delay = 5.0

    if model in GOOGLE_MODELS:
        query_fn = single_query_google
    elif model in OPENAI_MODELS:
        query_fn = single_query_openai
    elif model in ANTHROPIC_MODELS:
        query_fn = single_query_anthropic
    else:
        raise ValueError(f"Model {model} not supported")

    for attempt in range(max_retries):
        try:
            response: QueryResult = query_fn(prompt, model)
            return response  # Success, return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed {e}")
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2**attempt) + (random.random() * 0.5)  # Add jitter
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


def query_model_batch_api(
    prompts: List[str],
    model: str,
    num_workers: int = 16,
) -> List[QueryResult]:
    export()
    # Initialize results list with placeholders
    results: List[QueryResult | None] = [None] * len(prompts)
    lock = threading.Lock()  # Lock for thread-safe updates (might not be strictly needed now but good practice)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks, storing future and original index
        futures = {executor.submit(query_model_api, prompt, model): index for index, prompt in enumerate(prompts)}

        # Process results as they complete, using tqdm for progress
        for future in tqdm.tqdm(as_completed(futures), total=len(prompts), desc="Processing samples"):
            original_index = futures[future]
            original_prompt = prompts[original_index]  # Get original prompt for error message if needed
            try:
                result = future.result()
                if result:
                    # Place result in the correct position
                    with lock:
                        results[original_index] = result
                else:
                    # Handle case where query_model returned None (max retries reached)
                    error_result = QueryResult(
                        response_text=f'Error: Max retries reached for prompt starting with "{original_prompt[:50]}..."',
                        input_token_count=0,
                        output_token_count=0,
                        is_error=True,
                    )
                    with lock:
                        results[original_index] = error_result

            except Exception as exc:
                # Print part of the prompt string to identify it
                print(f'Prompt starting with "{original_prompt[:50]}..." generated an exception: {exc}')

                # Create a dictionary to store the error result
                error_result = QueryResult(
                    response_text=f"Error: {exc}",
                    input_token_count=0,  # Indicate no successful API call
                    output_token_count=0,  # Indicate no successful API call
                    is_error=True,
                )
                # Place error result in the correct position
                results[original_index] = error_result

    assert all(x is not None for x in results)

    return cast(List[QueryResult], results)
