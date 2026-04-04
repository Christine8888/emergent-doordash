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

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        """Normalize common math formatting wrappers around an answer."""
        value = text.strip()
        while True:
            updated = value

            if len(updated) >= 2 and updated.startswith("$") and updated.endswith("$"):
                updated = updated[1:-1].strip()

            boxed_match = re.fullmatch(r"\\boxed\s*\{(.*)\}", updated, flags=re.DOTALL)
            if boxed_match is not None:
                updated = boxed_match.group(1).strip()

            if updated == value:
                break
            value = updated
        return value

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

        dataset = load_dataset("MathArena/aime_2025", split="train")
        for example in dataset:
            rows.append(
                (
                    str(example["problem"]),
                    str(example["answer"]),
                    "MathArena/aime_2025",
                )
            )

        dataset = load_dataset("MathArena/aime_2026", split="train")
        for example in dataset:
            rows.append(
                (
                    str(example["problem"]),
                    str(example["answer"]),
                    "MathArena/aime_2026",
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
        if match is not None:
            return match.group(1).strip()

        # Fallback for prompts that ask for final answers in \boxed{...}.
        boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", response_text)
        if boxed_matches:
            return boxed_matches[-1].strip()
        return None

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        normalized_extracted = self._normalize_answer_text(extracted_answer)
        normalized_gold = self._normalize_answer_text(problem.answer)
        return normalized_extracted == normalized_gold


def get_dataset_spec(benchmark_name: str) -> DatasetSpecBase:
    specs = {
        "aime2025_2026": AIME20252026Spec(),
    }
    return specs[benchmark_name.lower()]
