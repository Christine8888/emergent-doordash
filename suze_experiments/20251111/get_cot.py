"""
Purpose: extract some CoT for different models on AIME 2025 dataset to see 
which one would be nice to use
"""
import os
# Set PyTorch memory allocation config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import numpy as np
import torch

from src.data.datasets import AIME2025, Dataset, ModelAnswer
from src.models.query import query_model_batch, ModelConfig, ModelType, OPENAI_MODELS, ANTHROPIC_MODELS, GOOGLE_MODELS

NUM_SAMPLES = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # python suze_experiments/20251111/get_cot.py
    save_dir = ""
    save_path = os.path.join(save_dir, "aime_cot.json")
    
    # Load from existing file if it exists, otherwise load from HuggingFace
    if os.path.exists(save_path):
        print(f"Loading existing dataset from {save_path}")
        cot_dataset: Dataset = AIME2025.load_from_file(save_path)
    else:
        print(f"File {save_path} not found, loading from HuggingFace")
        cot_dataset: Dataset = AIME2025.load_from_huggingface()

    # query a model with one specific question
    # models = ["o3-2025-04-16", "gpt-5-2025-08-07", "claude-opus-4-1-20250805", "claude-sonnet-4-5-20250929", "Qwen/Qwen-7B-Chat", "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3.1", "moonshotai/Kimi-K2-Thinking"]
    # models = ["gpt-5-nano-2025-08-07"] # for testing; low cost
    models = ["claude-sonnet-4-5-20250929"]

    cot_samples = random.sample(cot_dataset.data, NUM_SAMPLES)
    
    # Loop over models first, then batch all samples for each model
    for model_name in models:
        # Determine model type based on model name
        if model_name in GOOGLE_MODELS:
            model_type = ModelType.GEMINI
        elif model_name in OPENAI_MODELS:
            model_type = ModelType.OPENAI
        elif model_name in ANTHROPIC_MODELS:
            model_type = ModelType.ANTHROPIC
        else:
            # must be a local model
            model_type = ModelType.LOCAL
        
        # Create model config
        model_config = ModelConfig(
            model_name=model_name,
            model_type=model_type,
        )
        
        # Collect prompts and corresponding samples for this model
        # Only include samples that don't already have a response from this model with the same prompt
        prompts = []
        samples_to_process = []
        for cot_sample in cot_samples:
            prompt = cot_sample.question
            # Check if this model already has a response for this question with this exact prompt
            has_existing_response = any(
                response.model == model_name and response.prompt == prompt
                for response in cot_sample.ground_truth_cot_responses
            )
            if not has_existing_response:
                prompts.append(prompt)
                samples_to_process.append(cot_sample)
        
        if not prompts:
            print(f"Model {model_name} already has responses for all {len(cot_samples)} samples, skipping...")
            continue
        
        print(f"Sending batch of {len(prompts)} requests to model {model_name} (skipping {len(cot_samples) - len(prompts)} already completed)...")
        query_results = query_model_batch(prompts, model_config)
        print(f"Got {len(query_results)} responses for model {model_name}")
        
        # Process results for each sample
        for cot_sample, query_result in zip(samples_to_process, query_results):
            extracted_answer = cot_dataset.extract_answer(query_result.response_text)
            is_correct = cot_dataset.is_correct(extracted_answer, cot_sample.ground_truth_answer)
            cot_sample.ground_truth_cot_responses.append(
                ModelAnswer(
                    model=model_name, 
                    cot=query_result.response_text,
                    extracted_answer=extracted_answer,
                    is_correct=is_correct,
                    prompt=cot_sample.question
                )
            )
        
        # Save immediately after processing each model's responses
        print(f"Saving results to {save_path}...")
        cot_dataset.save_to_file(save_path)
        print(f"Saved results for model {model_name}")
        
        # Additional cleanup for local models to free GPU memory
        if model_type == ModelType.LOCAL:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"Cleaned up GPU memory after {model_name}")


    # TODO read from file and compare outputs from different models
    # TODO query more models for CoT and compare
    # TODO try different hinting strategies with these CoT
    # (see aime_cot.json; not saved to github!)




    


if __name__ == "__main__":
    set_seed(42)
    main()
