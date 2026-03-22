import hashlib
import re
from typing import List

from datasets import load_dataset
from pydantic import BaseModel

class Datum(BaseModel): # only for questions of this dataset!
    id: str
    question: str
    answer: str


class AIME2025(BaseModel):
    data: List[Datum] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Datum:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def load(cls) -> "AIME2025":
        data = []
        for config in ["AIME2025-I", "AIME2025-II"]:
            for example in load_dataset("opencompass/AIME2025", config, split="test"):
                question = example["question"]
                answer = re.search(r"\d+", example["answer"]).group()
                id = hashlib.md5(question.encode()).hexdigest()[:8]
                data.append(Datum(id=id, question=question, answer=answer))
        return cls(data=data)

    def is_correct(self, model_answer: str, datum: Datum) -> bool:
        # the model answer should already be take from the model response
        return model_answer.strip() == datum.answer.strip()