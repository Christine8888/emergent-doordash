"""
Purpose: extract some CoT for different models on AIME 2025 dataset to see 
which one would be nice to use
"""
from datasets import load_dataset
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import hashlib
import random
import numpy as np
import torch
import random

from src.data.datasets import AIME2025, Dataset
from src.models.query_api import query_model_api


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cot_dataset: Dataset = AIME2025.load_from_huggingface()
    

    # choose one question to query model with
    cot_sample = random.choice(cot_dataset)
    print(cot_sample)

    # query a model with one specific question
    # models = ["o3-2025-04-16", "gpt-5-2025-08-07", "claude-opus-4-1", "claude-sonnet-4-5"]
    models = ["gpt-5-nano-2025-08-07"]

    for model in models:
        result = query_model_api(cot_sample.question, model)
        print(result)
        # cot_sample.ground_truth_cot_responses.append(COT_Response(model=model, cot=result.response_text))


    


if __name__ == "__main__":
    set_seed(42)
    main()
