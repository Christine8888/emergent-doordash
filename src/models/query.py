import random  # Import random for jitter calculation
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, cast
from enum import Enum

import anthropic
import openai
import tqdm
from google import genai
from google.genai import types
import dotenv
from dataclasses import dataclass
import torch
from vllm import LLM, SamplingParams

class ModelType(Enum):
    GEMINI = "gemini"
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    model_name: str
    model_type: ModelType
    dtype: str = "bfloat16"
    temperature: float = 1.0
    max_tokens: Optional[int] = 8192
    chat: bool = False
    system_prompt: str = "You are a helpful assistant."
    gen_batch_size: int = 128
    # ^ This is used in server based generation. It is attached to the model config because different
    # models have different generation speeds (due to CoT lenght) so this seems like the right place to put it
    use_system_prompt: bool = False


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


def query_model_local(prompt: str, model_config: ModelConfig) -> QueryResult:
    """Query a local model using vLLM.
    
    Args:
        prompt: The prompt to send to the model
        model_config: ModelConfig with model configuration
    
    Returns:
        QueryResult with the model's response
    """
    results = query_model_local_batch([prompt], model_config)
    return results[0]


def query_model_local_batch(prompts: List[str], model_config: ModelConfig) -> List[QueryResult]:
    """Query a local model using vLLM with batching.
    
    Args:
        prompts: List of prompts to send to the model
        model_config: ModelConfig with model configuration
    
    Returns:
        List of QueryResult with the model's responses
    """
    if DEBUG_QUERIES:
        print(f"[LOCAL] model={model_config.model_name}")
        print(f"[LOCAL] batch_size={len(prompts)}")
    
    start_time = time.time()
    
    # Set up vLLM kwargs
    vllm_kwargs = {}
    if "gemma-2" in model_config.model_name:
        vllm_kwargs["enforce_eager"] = True
    
    # Adjust memory utilization for large models
    gpu_memory_utilization = 0.9
    if "DeepSeek-V3.1" in model_config.model_name:
        # Lower memory utilization for DeepSeek-V3.1 to avoid OOM
        gpu_memory_utilization = 0.7
    
    # Load model
    vllm_model = LLM(
        model=model_config.model_name,
        dtype=model_config.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        **vllm_kwargs,
    )
    
    # Set up sampling parameters
    if model_config.max_tokens is not None:
        sampling_params = SamplingParams(
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
    else:
        sampling_params = SamplingParams(
            temperature=model_config.temperature,
        )
    
    # Generate responses
    if model_config.chat:
        if model_config.use_system_prompt:
            conversations = [
                [
                    {"role": "system", "content": model_config.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
        else:
            conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        results = vllm_model.chat(conversations, sampling_params=sampling_params)
    else:
        results = vllm_model.generate(prompts, sampling_params=sampling_params)
    
    # Extract results
    query_results = []
    for result in results:
        input_token_count = len(result.prompt_token_ids)
        output_token_count = len(result.outputs[0].token_ids)
        response_text = result.outputs[0].text
        
        query_results.append(QueryResult(
            response_text=response_text,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            is_error=False,
            cost=0.0,  # Local models have no cost
        ))
    
    elapsed_ms = (time.time() - start_time) * 1000.0
    if DEBUG_QUERIES:
        print(f"[LOCAL] latency_ms={elapsed_ms:.0f}, avg_per_item_ms={elapsed_ms/len(prompts):.0f}")
    
    # Properly clean up vLLM model and GPU memory
    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return query_results


def query_model(prompt: str, model_config: ModelConfig) -> QueryResult:
    """General query function that routes to the appropriate query method based on model type.
    
    Args:
        prompt: The prompt to send to the model
        model_config: ModelConfig with model configuration
    
    Returns:
        QueryResult with the model's response
    """
    if model_config.model_type == ModelType.LOCAL:
        return query_model_local(prompt, model_config)
    else:
        # For API models (GEMINI, OPENAI, ANTHROPIC), use the API query function
        return query_model_api(prompt, model_config.model_name)


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


def query_model_batch(prompts: List[str], model_config: ModelConfig) -> List[QueryResult]:
    """Batch query function that routes to the appropriate batch query method based on model type.
    
    Args:
        prompts: List of prompts to send to the model
        model_config: ModelConfig with model configuration
    
    Returns:
        List of QueryResult with the model's responses
    """
    if model_config.model_type == ModelType.LOCAL:
        return query_model_local_batch(prompts, model_config)
    else:
        # For API models, use concurrent requests
        return query_model_api_batch(prompts, model_config.model_name)


def query_model_api_batch(prompts: List[str], model: str) -> List[QueryResult]:
    """Batch query API models using concurrent requests.
    
    Args:
        prompts: List of prompts to send to the model
        model: Model name
    
    Returns:
        List of QueryResult with the model's responses
    """
    if DEBUG_QUERIES:
        print(f"[API] model={model}, batch_size={len(prompts)}")
    
    results = []
    with ThreadPoolExecutor(max_workers=min(len(prompts), 10)) as executor:
        # Submit all queries
        future_to_prompt = {
            executor.submit(query_model_api, prompt, model): prompt 
            for prompt in prompts
        }
        
        # Collect results in order
        prompt_to_result = {}
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            result = future.result()
            prompt_to_result[prompt] = result
        
        # Return results in the same order as prompts
        for prompt in prompts:
            results.append(prompt_to_result[prompt])
    
    return results
