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
import re
import hashlib
from typing import List

from src.data.datasets import AIME2025, Dataset, ModelAnswer, ModelHints, HintsDatum, HintsDataset
from src.models.query import query_model_batch, query_model, ModelConfig, ModelType, OPENAI_MODELS, ANTHROPIC_MODELS, GOOGLE_MODELS

NUM_SAMPLES = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


HINTS_PROMPT_TEMPLATE = """
You previously solved this problem:

Problem:
{question}

Your solution:
{cot}

Create 10 independent hints. Each hint must:
- Stand alone and be clear without reading any other hint.
- Contain unique information that does not overlap with the others.
- Be roughly equal in usefulness.
- Avoid revealing the full solution.

Format:
1. Hint one
2. Hint two
...
10. Hint ten

Provide exactly 10 hints:
"""


def get_prompt_template_hash() -> str:
    """Get hash of the current prompt template to detect changes."""
    return hashlib.md5(HINTS_PROMPT_TEMPLATE.encode()).hexdigest()[:8]


def create_hints_prompt(question: str, cot: str) -> str:
    return HINTS_PROMPT_TEMPLATE.format(question=question, cot=cot)


def extract_hints_from_response(response_text: str) -> List[str]:
    """Extract hints from the model's response."""
    hints = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to match numbered list patterns (1., 2., etc. or 1), 2), etc.)
        match = re.match(r'^\d+[.)]\s*(.+)$', line)
        if match:
            hints.append(match.group(1).strip())
        elif len(hints) < 10:
            # If no number pattern, but we haven't reached 10 hints yet, add it
            hints.append(line)
    
    # If we got fewer than 10 hints, pad with empty strings or use what we have
    while len(hints) < 10:
        hints.append("")
    
    # Return exactly 10 hints
    return hints[:10]


def main():
    # python suze_experiments/20251111/get_cot.py
    save_dir = ""
    save_path = os.path.join(save_dir, "aime_cot.json")
    hints_path = os.path.join(save_dir, "aime_hints.json")
    
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
    # models = ["o3-2025-04-16", "gpt-5-2025-08-07", "claude-opus-4-1-20250805", "claude-sonnet-4-5-20250929"]
    models = ["gpt-5-2025-08-07"]

    # cot_samples = random.sample(cot_dataset.data, NUM_SAMPLES)
    cot_samples = cot_dataset.data # use all data
    
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
                for response in cot_sample.cot_responses
            )
            if not has_existing_response:
                prompts.append(prompt)
                samples_to_process.append(cot_sample)
        
        # Load existing hints if file exists
        if os.path.exists(hints_path):
            print(f"Loading existing hints from {hints_path}")
            hints_dataset = HintsDataset.load_from_file(hints_path)
        else:
            print(f"Creating new hints dataset")
            hints_dataset = HintsDataset()
        
        if not prompts:
            print(f"Model {model_name} already has responses for all {len(cot_samples)} samples, checking for hints...")
        else:
            print(f"Sending batch of {len(prompts)} requests to model {model_name} (skipping {len(cot_samples) - len(prompts)} already completed)...")
            query_results = query_model_batch(prompts, model_config)
            print(f"Got {len(query_results)} responses for model {model_name}")
            
            # Process results for each sample
            for cot_sample, query_result in zip(samples_to_process, query_results):
                extracted_answer = cot_dataset.extract_answer(query_result.response_text)
                is_correct = cot_dataset.is_correct(extracted_answer, cot_sample.ground_truth_answer)
                model_answer = ModelAnswer(
                    model=model_name, 
                    cot=query_result.response_text,
                    extracted_answer=extracted_answer,
                    is_correct=is_correct,
                    prompt=cot_sample.question
                )
                cot_sample.cot_responses.append(model_answer)
                
                # If correct, ask for hints
                if is_correct:
                    # Find or create HintsDatum for this problem
                    hints_datum = None
                    for datum in hints_dataset.data:
                        if datum.id == cot_sample.id:
                            hints_datum = datum
                            break
                    
                    if hints_datum is None:
                        # Create new HintsDatum
                        hints_datum = HintsDatum(
                            id=cot_sample.id,
                            question=cot_sample.question,
                            hints=[]
                        )
                        hints_dataset.data.append(hints_datum)
                    
                    # Check if hints already exist for this model with the same prompt template
                    current_template_hash = get_prompt_template_hash()
                    existing_model_hints_idx = None
                    model_exists_with_different_template = False
                    for idx, model_hints in enumerate(hints_datum.hints):
                        if model_hints.model == model_name:
                            # Check if prompt template matches
                            if model_hints.prompt_template_hash == current_template_hash:
                                existing_model_hints_idx = idx
                            else:
                                model_exists_with_different_template = True
                                existing_model_hints_idx = idx  # Store index for replacement
                            break
                    
                    # Only generate hints if they don't exist with the same template
                    if existing_model_hints_idx is None or model_exists_with_different_template:
                        if model_exists_with_different_template:
                            print(f"Model {model_name} got correct answer for problem {cot_sample.id}, regenerating hints (prompt template changed)...")
                        else:
                            print(f"Model {model_name} got correct answer for problem {cot_sample.id}, requesting hints...")
                        hints_prompt = create_hints_prompt(cot_sample.question, query_result.response_text)
                        hints_result = query_model(hints_prompt, model_config)
                        hints_list = extract_hints_from_response(hints_result.response_text)
                        
                        model_hints = ModelHints(
                            model=model_name,
                            hints=hints_list,
                            model_cot=query_result.response_text,
                            prompt_template_hash=current_template_hash,
                            prompt=hints_prompt
                        )
                        
                        if existing_model_hints_idx is not None:
                            # Replace hints generated with different template
                            hints_datum.hints[existing_model_hints_idx] = model_hints
                            print(f"Regenerated hints for problem {cot_sample.id}, model {model_name}")
                        else:
                            # Add new hints
                            hints_datum.hints.append(model_hints)
                            print(f"Added hints for problem {cot_sample.id}, model {model_name}")
                        
                        # Save hints immediately (will be sorted at the end)
                        hints_dataset.save_to_file(hints_path)
                        print(f"Saved hints to {hints_path}")
                    else:
                        print(f"Hints already exist for problem {cot_sample.id}, model {model_name} with same prompt template, skipping...")
                    
                    # Save hints immediately (will be sorted at the end)
                    hints_dataset.save_to_file(hints_path)
                    print(f"Saved hints to {hints_path}")
            
            # Save immediately after processing each model's responses
            print(f"Saving results to {save_path}...")
            cot_dataset.save_to_file(save_path)
            print(f"Saved results for model {model_name}")
        
        # Check all samples (including existing ones) for correct answers that need hints
        # Only generate hints for models that originally gave the correct answer
        print(f"Checking for correct answers that need hints for model {model_name}...")
        for cot_sample in cot_samples:
            # Find all correct responses from this model (only process if is_correct == True)
            for response in cot_sample.cot_responses:
                if response.model == model_name and response.is_correct:
                    # Find or create HintsDatum for this problem
                    hints_datum = None
                    for datum in hints_dataset.data:
                        if datum.id == cot_sample.id:
                            hints_datum = datum
                            break
                    
                    # Check if hints already exist for this model with the same prompt template
                    current_template_hash = get_prompt_template_hash()
                    has_existing_hints = False
                    existing_model_hints_idx = None
                    if hints_datum is not None:
                        for idx, model_hints in enumerate(hints_datum.hints):
                            if model_hints.model == model_name:
                                # Check if prompt template matches
                                if model_hints.prompt_template_hash == current_template_hash:
                                    has_existing_hints = True
                                existing_model_hints_idx = idx
                                break
                    
                    if not has_existing_hints:
                        if existing_model_hints_idx is not None:
                            print(f"Found correct answer for problem {cot_sample.id} with different prompt template, regenerating hints...")
                        else:
                            print(f"Found correct answer for problem {cot_sample.id} without hints, generating hints...")
                        hints_prompt = create_hints_prompt(cot_sample.question, response.cot)
                        hints_result = query_model(hints_prompt, model_config)
                        hints_list = extract_hints_from_response(hints_result.response_text)
                        
                        if hints_datum is None:
                            # Create new HintsDatum
                            hints_datum = HintsDatum(
                                id=cot_sample.id,
                                question=cot_sample.question,
                                hints=[]
                            )
                            hints_dataset.data.append(hints_datum)
                        
                        model_hints = ModelHints(
                            model=model_name,
                            hints=hints_list,
                            model_cot=response.cot,
                            prompt_template_hash=current_template_hash,
                            prompt=hints_prompt
                        )
                        
                        if existing_model_hints_idx is not None:
                            # Replace hints generated with different template
                            hints_datum.hints[existing_model_hints_idx] = model_hints
                            print(f"Regenerated hints for problem {cot_sample.id}, model {model_name} (prompt template changed)")
                        else:
                            # Add new hints
                            hints_datum.hints.append(model_hints)
                            print(f"Added hints for problem {cot_sample.id}, model {model_name}")
                        
                        # Save hints immediately (will be sorted at the end)
                        hints_dataset.save_to_file(hints_path)
                        print(f"Saved hints to {hints_path}")
        
        # Sort all hints by question order at the end (matching cot_dataset order)
        print(f"Sorting hints to match cot_dataset order...")
        # Create a mapping from problem_id to its index in cot_dataset
        problem_id_to_index = {datum.id: idx for idx, datum in enumerate(cot_dataset.data)}
        hints_dataset.data.sort(key=lambda d: problem_id_to_index.get(d.id, len(cot_dataset.data)))
        hints_dataset.save_to_file(hints_path)
        
        # Additional cleanup for local models to free GPU memory
        if model_type == ModelType.LOCAL:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"Cleaned up GPU memory after {model_name}")


    # (see aime_cot.json; not saved to github!)


# TODO next: run with gpt5 hints over all problems. First add cost calculator
# TODO: once we have all the hints, eval

    


if __name__ == "__main__":
    set_seed(42)
    main()
