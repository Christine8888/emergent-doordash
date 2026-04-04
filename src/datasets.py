from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    answer: str
    source: str


class DatasetSpecBase(ABC):
    name: str

    def load_problems(self) -> list[Problem]:
        raise NotImplementedError

    def extract_answer(self, response_text: str) -> str | None:
        raise NotImplementedError

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        raise NotImplementedError


class AIME20252026Spec(DatasetSpecBase):
    name = "aime2025_2026"

    def _dataset_cache_path(self) -> Path:
        return Path("data") / "datasets" / f"{self.name}.jsonl"

    def _load_problems_from_cache(self, path: Path) -> list[Problem]:
        problems: list[Problem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                problems.append(
                    Problem(
                        problem_id=row["problem_id"],
                        question=row["question"],
                        answer=row["answer"],
                        source=row["source"],
                    )
                )
        return problems

    def _save_problems_to_cache(self, path: Path, problems: list[Problem]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for problem in problems:
                f.write(
                    json.dumps(
                        {
                            "problem_id": problem.problem_id,
                            "question": problem.question,
                            "answer": problem.answer,
                            "source": problem.source,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        rows: list[tuple[str, str, str]] = []

        for config_name in ("AIME2025-I", "AIME2025-II"):
            dataset = load_dataset("opencompass/AIME2025", config_name, split="test")
            for example in dataset:
                rows.append(
                    (
                        str(example["question"]),
                        str(example["answer"]),
                        "opencompass/AIME2025",
                    )
                )

        dataset = load_dataset("math-ai/aime26", split="test")
        for example in dataset:
            rows.append(
                (
                    str(example["problem"]),
                    str(example["answer"]),
                    "math-ai/aime26",
                )
            )

        problems: list[Problem] = []
        for i, (question, answer, source) in enumerate(rows, start=1):
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:04d}",
                    question=question,
                    answer=answer,
                    source=source,
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems

    def extract_answer(self, response_text: str) -> str | None:
        match = re.search(r"<answer>(.*?)</answer>", response_text, re.IGNORECASE | re.DOTALL)
        if match is None:
            return None
        return match.group(1).strip()

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        return extracted_answer.strip() == problem.answer.strip()


def get_dataset_spec(benchmark_name: str) -> DatasetSpecBase:
    specs = {
        "aime2025_2026": AIME20252026Spec(),
    }
    return specs[benchmark_name.lower()]
