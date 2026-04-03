from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ResultEntry(BaseModel):
    datum_id: str
    question: str
    ground_truth_answer: str
    prompt: str
    model: str
    response_text: str
    extracted_answer: Optional[str]
    is_correct: Optional[bool]
    input_token_count: int
    output_token_count: int
    cost: float
    is_error: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ResultsLog(BaseModel):
    entries: List[ResultEntry] = []

    def add(self, entry: ResultEntry):
        self.entries.append(entry)

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, file_path: str) -> "ResultsLog":
        with open(file_path) as f:
            return cls.model_validate_json(f.read())

    def summary(self) -> str:
        total = len(self.entries)
        correct = sum(1 for e in self.entries if e.is_correct)
        errors = sum(1 for e in self.entries if e.is_error)
        total_cost = sum(e.cost for e in self.entries)
        return f"{correct}/{total} correct, {errors} errors, ${total_cost:.4f} total cost"
