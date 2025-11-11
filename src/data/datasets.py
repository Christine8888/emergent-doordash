import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from datasets import load_dataset
from pydantic import BaseModel, Field

NO_ANSWER_FOUND_ERROR = "<NO_ANSWER_FOUND>"


class SavableBaseModel(BaseModel):
    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, file_path: str) -> "SavableBaseModel":
        with open(file_path) as f:
            return cls.model_validate_json(f.read())


class ModelAnswer(SavableBaseModel):
    model: str
    cot: str
    extracted_answer: str
    is_correct: bool
    prompt: Optional[str] = None

class Datum(SavableBaseModel):
    id: str
    ground_truth_answer: str
    question: str
    ground_truth_cot_responses: List[ModelAnswer] = [] # list of responses from different models
    sampled_answers: List[ModelAnswer] = []



class Dataset(SavableBaseModel, ABC):
    data: List[Datum]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Datum:
        return self.data[index]

    @abstractmethod
    def is_correct(self, model_answer: str, real_answer: str) -> bool: ...

    @classmethod
    @abstractmethod
    def load_from_huggingface(cls) -> "Dataset": ...

    @abstractmethod
    def extract_answer(self, model_output: str) -> str:
        # Extract the answer from the model output
        ...


class AIME2025(Dataset):
    def is_correct(self, model_answer: str, real_answer: str) -> bool:
        return model_answer.strip() == real_answer.strip()

    @classmethod
    def load_from_huggingface(cls) -> "AIME2025":
        dataset_name = "opencompass/AIME2025"

        configs = ["AIME2025-I", "AIME2025-II"]

        ids: List[str] = []
        questions: List[str] = []
        answers: List[str] = []
        for config in configs:
            dataset = load_dataset(dataset_name, config, split="test")

            for example in dataset:
                answer = example["answer"]
                numerical_answer = re.search(r"\d+", answer).group()  # type: ignore
                answers.append(numerical_answer)

                question = example["question"]
                assert question not in questions, "Question already exists"
                questions.append(question)

                id = hashlib.md5(question.encode()).hexdigest()[:8]
                assert id not in ids, "ID already exists"
                ids.append(id)

        return cls(
            data=[Datum(id=id, question=question, ground_truth_answer=answer) for id, question, answer in zip(ids, questions, answers)],
        )

    def extract_answer(self, model_output: str) -> str:
        # Select the last number in the model output
        matches = re.findall(r"\d+", model_output)
        if not matches:
            return NO_ANSWER_FOUND_ERROR
        return matches[-1]
