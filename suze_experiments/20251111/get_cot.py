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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SavableBaseModel(BaseModel):
    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, file_path: str) -> "SavableBaseModel":
        with open(file_path) as f:
            return cls.model_validate_json(f.read())

class COT_Response(SavableBaseModel):
    model: str
    cot: str

class Datum(SavableBaseModel):
    id: str
    ground_truth_answer: str
    question: str
    ground_truth_cot_responses: Optional[List[COT_Response]] = None # list of responses from different models


def main():
    cot_dataset: list[Datum] = []
    ids: List[str] = []
    # get question and answer to query a model with
    dataset_name = "opencompass/AIME2025"
    for config in ['AIME2025-I', 'AIME2025-II']:
        dataset = load_dataset(dataset_name, config, split="test")

        for row in dataset:
            question, ground_truth_answer = row['question'], row['answer']

            id = hashlib.md5(question.encode()).hexdigest()[:8]
            assert id not in ids, "ID already exists"
            ids.append(id)

            cot_dataset.append(Datum(id=id, question=question, ground_truth_answer=ground_truth_answer))
            break

    # choose one question to query model with
    import random
    cot_sample = random.choice(cot_dataset)
    print(cot_sample)

    # query a model with one specific question
    


if __name__ == "__main__":
    set_seed(42)
    main()
