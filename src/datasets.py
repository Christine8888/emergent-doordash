from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re

@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    answer: str
    source: str


class HintType(str, Enum):
    masked = "masked"


class DatasetSpecBase(ABC):
    name: str
    PROMPT_VERSIONS: dict[HintType, str] = {}
    PROMPT_BUILDERS: dict[HintType, str] = {}
    POST_PROCESS_VERSIONS: dict[HintType, str] = {}
    POST_PROCESSORS: dict[HintType, str] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls is DatasetSpecBase:
            return

        expected = set(HintType)
        versions_keys = set(cls.PROMPT_VERSIONS.keys())
        builders_keys = set(cls.PROMPT_BUILDERS.keys())
        post_versions_keys = set(cls.POST_PROCESS_VERSIONS.keys())
        post_processors_keys = set(cls.POST_PROCESSORS.keys())

        if versions_keys != expected:
            missing = sorted(h.value for h in (expected - versions_keys))
            extra = sorted(h.value for h in (versions_keys - expected))
            raise TypeError(
                f"{cls.__name__}.PROMPT_VERSIONS must include exactly all hint types. "
                f"missing={missing} extra={extra}"
            )
        if builders_keys != expected:
            missing = sorted(h.value for h in (expected - builders_keys))
            extra = sorted(h.value for h in (builders_keys - expected))
            raise TypeError(
                f"{cls.__name__}.PROMPT_BUILDERS must include exactly all hint types. "
                f"missing={missing} extra={extra}"
            )
        if post_versions_keys != expected:
            missing = sorted(h.value for h in (expected - post_versions_keys))
            extra = sorted(h.value for h in (post_versions_keys - expected))
            raise TypeError(
                f"{cls.__name__}.POST_PROCESS_VERSIONS must include exactly all hint types. "
                f"missing={missing} extra={extra}"
            )
        if post_processors_keys != expected:
            missing = sorted(h.value for h in (expected - post_processors_keys))
            extra = sorted(h.value for h in (post_processors_keys - expected))
            raise TypeError(
                f"{cls.__name__}.POST_PROCESSORS must include exactly all hint types. "
                f"missing={missing} extra={extra}"
            )

        for hint_type, method_name in cls.PROMPT_BUILDERS.items():
            if not isinstance(method_name, str) or not hasattr(cls, method_name):
                raise TypeError(
                    f"{cls.__name__}.PROMPT_BUILDERS[{hint_type.value!r}] "
                    f"must reference an existing method name."
                )
        for hint_type, method_name in cls.POST_PROCESSORS.items():
            if not isinstance(method_name, str) or not hasattr(cls, method_name):
                raise TypeError(
                    f"{cls.__name__}.POST_PROCESSORS[{hint_type.value!r}] "
                    f"must reference an existing method name."
                )

    def load_problems(self) -> list[Problem]:
        raise NotImplementedError

    def supported_hint_types(self) -> list[str]:
        return [hint_type.value for hint_type in HintType]

    def prompt_version(self, hint_type: str) -> str:
        return self.PROMPT_VERSIONS[HintType(hint_type)]

    def build_hint_prompt(self, problem: Problem, hint_type: str) -> str:
        builder_name = self.PROMPT_BUILDERS[HintType(hint_type)]
        builder = getattr(self, builder_name)
        return builder(problem)

    def post_process_version(self, hint_type: str) -> str:
        return self.POST_PROCESS_VERSIONS[HintType(hint_type)]

    def post_process_hint(self, problem: Problem, hint_type: str, model_output: str) -> str:
        processor_name = self.POST_PROCESSORS[HintType(hint_type)]
        processor = getattr(self, processor_name)
        return processor(problem, model_output)

    def extract_answer(self, response_text: str) -> str | None:
        raise NotImplementedError

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        raise NotImplementedError


class AIME20252026Spec(DatasetSpecBase):
    name = "aime2025_2026"
    PROMPT_VERSIONS = {
        HintType.masked: f"{name}_masked_v1",
    }
    PROMPT_BUILDERS = {
        HintType.masked: "_build_masked_prompt",
    }
    POST_PROCESS_VERSIONS = {
        HintType.masked: f"{name}_masked_post_v1",
    }
    POST_PROCESSORS = {
        HintType.masked: "_post_process_masked",
    }

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

    def _build_masked_prompt(self, problem: Problem) -> str:
        # return (
        #     "Write a detailed solution to the following problem.\n"
        #     "The final answer should be placed between <answer></answer> tags\n"
        #     "Do not reveal the final answer before placing it between the tags.\n"
        #     "You can do verification and validation of the answer, but only after <answer></answer> tags so the answer is not revealed before then"
        #     f"Here is the problem: {problem.question}"
        # )
        # return (
        #     "Solve the following problem step by step.\n"
        #     "Rules:\n"
        #     "1. Show all reasoning and intermediate calculations freely.\n"
        #     "2. When you reach the final step, write \"Final answer: <answer>YOUR ANSWER HERE</answer>\""
        #     " — do not state the numerical result anywhere before this line.\n"
        #     "3. Any verification must appear after the </answer> tag.\n\n"
        #     f"Here is the problem: {problem.question}"
        # )
        return (
            "Write a detailed solution to the following problem.\n"
            "Work through all intermediate steps, but stop just before the final computation.\n"
            "Perform the last step (e.g., summing results, final arithmetic) inside "
            "<final_step></final_step> tags — this is the ONLY place the final numerical "
            "result should appear.\n"
            "After the closing </final_step> tag, restate just the bare answer in "
            "<answer></answer> tags.\n"
            # "You may do verification and validation after the closing </answer> tag.\n\n"
            f"Here is the problem: {problem.question}"
        )

    def _post_process_masked(self, problem: Problem, model_output: str) -> str:
        _ = problem
        tag_match = re.search(r"<\s*(final_step|answer)\b", model_output, re.IGNORECASE)
        if tag_match is None:
            return model_output.strip()
        return model_output[: tag_match.start()].rstrip()

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
