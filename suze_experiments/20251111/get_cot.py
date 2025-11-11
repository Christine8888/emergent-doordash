"""
Purpose: extract some CoT for different models on AIME 2025 dataset to see 
which one would be nice to use
"""
import random
import numpy as np
import torch
import random
import os

from src.data.datasets import AIME2025, Dataset, ModelAnswer
from src.models.query_api import query_model_api


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cot_dataset: Dataset = AIME2025.load_from_huggingface()
    
    save_dir = ""
    save_path = os.path.join(save_dir, "aime_cot.json")

    # query a model with one specific question
    models = ["o3-2025-04-16", "gpt-5-2025-08-07", "claude-opus-4-1", "claude-sonnet-4-5"]
    # models = ["gpt-5-nano-2025-08-07"] # for testing; low cost

    cot_samples = random.sample(cot_dataset.data, 4)
    for cot_sample in cot_samples:

        for model in models:
            print(f"Sending request to model {model}...")
            query_result = query_model_api(cot_sample.question, model)
            print(f"Got response for model {model}.")
            extracted_answer = cot_dataset.extract_answer(query_result.response_text)
            is_correct = cot_dataset.is_correct(extracted_answer, cot_sample.ground_truth_answer)
            if is_correct:
                cot_sample.ground_truth_cot_responses.append(
                    ModelAnswer(
                        model=model, 
                        cot=query_result.response_text,
                        extracted_answer=extracted_answer,
                        is_correct=is_correct,
                        prompt=cot_sample.question
                    )
                )

    cot_dataset.save_to_file(save_path)




    


if __name__ == "__main__":
    set_seed(42)
    main()
