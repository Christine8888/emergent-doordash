from __future__ import annotations

from abc import ABC
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    answer: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetSpecBase(ABC):
    name: str

    def load_problems(self) -> list[Problem]:
        raise NotImplementedError

    def extract_answer(self, response_text: str) -> str | None:
        raise NotImplementedError

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        raise NotImplementedError

    def grade_response(self, response_text: str, problem: Problem) -> dict[str, Any]:
        extracted_answer = self.extract_answer(response_text)
        return {
            "is_correct": self.is_correct(extracted_answer, problem),
            "extracted_answer": extracted_answer,
            "metadata": {
                "grader_type": "dataset_extract_and_match",
            },
        }

    def build_prompt(self, problem: Problem) -> str:
        return problem.question

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
                        metadata=dict(row.get("metadata", {})),
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
                            "metadata": problem.metadata,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HLE_JUDGE_MODEL = "o3-mini-2025-01-31"
HLE_JUDGE_REASONING_EFFORT = "low"

HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


_load_project_env()


def _extract_tagged_answer(response_text: str) -> str | None:
    match = _ANSWER_TAG_RE.search(response_text)
    if match is None:
        return None
    return match.group(1).strip() or None


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_free_form_answer(text: str) -> str:
    value = _collapse_whitespace(text)
    value = value.strip()
    while True:
        updated = value
        if len(updated) >= 2 and updated[0] == updated[-1] and updated[0] in {"'", '"'}:
            updated = updated[1:-1].strip()
        updated = updated.rstrip(".")
        if updated == value:
            break
        value = updated
    return value


def _choice_labels(num_choices: int) -> list[str]:
    return list(_CHOICE_LABELS[:num_choices])


def _index_to_letter(index: int) -> str:
    return _CHOICE_LABELS[index]


def _extract_multiple_choice_answer(
    response_text: str,
    *,
    choices: list[str],
) -> str | None:
    tagged = _extract_tagged_answer(response_text)
    candidate_texts: list[str] = []
    if tagged is not None:
        candidate_texts.append(tagged)
    stripped = response_text.strip()
    if stripped:
        candidate_texts.append(stripped)

    labels = _choice_labels(len(choices))
    normalized_choice_to_label = {
        _normalize_free_form_answer(choice).lower(): label
        for label, choice in zip(labels, choices, strict=True)
    }
    for candidate_text in candidate_texts:
        normalized = _normalize_free_form_answer(candidate_text)
        upper = normalized.upper()
        if upper in labels:
            return upper
        digit_match = re.fullmatch(r"[1-9]\d*", normalized)
        if digit_match is not None:
            index = int(normalized) - 1
            if 0 <= index < len(labels):
                return labels[index]

        for pattern in (
            r"(?i)(?:final answer|answer)\s*(?:is|:)?\s*[\(\[]?([A-Z])[\)\]]?",
            r"[\(\[]([A-Z])[\)\]]",
            r"\b([A-Z])\b",
        ):
            matches = re.findall(pattern, candidate_text)
            for match in reversed(matches):
                label = match.upper()
                if label in labels:
                    return label

        mapped_label = normalized_choice_to_label.get(normalized.lower())
        if mapped_label is not None:
            return mapped_label
    return None


def _safe_literal_eval(text: str) -> Any:
    return ast.literal_eval(text)


def _json_safe_value(value: Any) -> Any:
    """Convert dataset values to something stable for local JSONL caches."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return str(value)


def _extract_last_boxed_or_tagged_answer(response_text: str) -> str | None:
    tagged = _extract_tagged_answer(response_text)
    if tagged is not None:
        return tagged

    boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", response_text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    return None


class MultipleChoiceDatasetSpec(DatasetSpecBase):
    def build_prompt(self, problem: Problem) -> str:
        choices = list(problem.metadata.get("choices", []))
        if not choices:
            return problem.question

        lines = [
            "Choose the best answer.",
            "Respond with only the answer letter.",
            "",
            problem.question.strip(),
        ]
        for label, choice in zip(_choice_labels(len(choices)), choices, strict=True):
            lines.append(f"{label}. {choice}")
        lines.append("Answer:")
        return "\n".join(lines)

    def extract_answer(self, response_text: str) -> str | None:
        default_choices = ["", "", "", ""]
        return _extract_multiple_choice_answer(response_text, choices=default_choices)

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        return extracted_answer.upper() == problem.answer.upper()


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

    def build_prompt(self, problem: Problem) -> str:
        return (
            "Solve the following competition math problem.\n"
            "Put your final answer within \\boxed{}.\n\n"
            f"Problem:\n{problem.question.strip()}"
        )

    def extract_answer(self, response_text: str) -> str | None:
        return _extract_last_boxed_or_tagged_answer(response_text)

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        normalized_extracted = self._normalize_answer_text(extracted_answer)
        normalized_gold = self._normalize_answer_text(problem.answer)
        return normalized_extracted == normalized_gold


class HellaSwagSpec(MultipleChoiceDatasetSpec):
    name = "hellaswag"

    @staticmethod
    def _preprocess(text: str) -> str:
        value = text.strip()
        value = value.replace(" [title]", ". ")
        value = re.sub(r"\[.*?\]", "", value)
        return _collapse_whitespace(value)

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        dataset = load_dataset("Rowan/hellaswag", split="validation")
        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            ctx = f'{example["ctx_a"]} {str(example["ctx_b"]).capitalize()}'
            query = self._preprocess(f'{example["activity_label"]}: {ctx}')
            choices = [self._preprocess(str(ending)) for ending in example["endings"]]
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:05d}",
                    question=query,
                    answer=_index_to_letter(int(example["label"])),
                    source="Rowan/hellaswag:validation",
                    metadata={"choices": choices},
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems


class PIQASpec(MultipleChoiceDatasetSpec):
    name = "piqa"

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        dataset = load_dataset("baber/piqa", split="validation")
        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:05d}",
                    question=f'Question: {str(example["goal"]).strip()}',
                    answer=_index_to_letter(int(example["label"])),
                    source="baber/piqa:validation",
                    metadata={"choices": [str(example["sol1"]), str(example["sol2"])]},
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems


class MMLUSpec(MultipleChoiceDatasetSpec):
    name = "mmlu"

    _SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        problems: list[Problem] = []
        for subject in self._SUBJECTS:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            for i, example in enumerate(dataset, start=1):
                answer_value = example["answer"]
                if isinstance(answer_value, str) and answer_value in {"A", "B", "C", "D"}:
                    answer = answer_value
                else:
                    answer = _index_to_letter(int(answer_value))
                problems.append(
                    Problem(
                        problem_id=f"{self.name}_{subject}_{i:04d}",
                        question=str(example["question"]).strip(),
                        answer=answer,
                        source=f"cais/mmlu:{subject}:test",
                        metadata={
                            "choices": [str(choice) for choice in example["choices"]],
                            "subject": subject,
                        },
                    )
                )
        self._save_problems_to_cache(cache_path, problems)
        return problems


class BBHSpec(DatasetSpecBase):
    name = "bbh"

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import get_dataset_config_names, load_dataset

        problems: list[Problem] = []
        for config_name in get_dataset_config_names("lukaemon/bbh"):
            dataset = load_dataset("lukaemon/bbh", config_name, split="test")
            for i, example in enumerate(dataset, start=1):
                problems.append(
                    Problem(
                        problem_id=f"{self.name}_{config_name}_{i:04d}",
                        question=str(example["input"]).strip(),
                        answer=str(example["target"]).strip(),
                        source=f"lukaemon/bbh:{config_name}:test",
                        metadata={"subtask": config_name},
                    )
                )
        self._save_problems_to_cache(cache_path, problems)
        return problems

    def build_prompt(self, problem: Problem) -> str:
        return (
            "Solve the following reasoning problem.\n"
            "Return only the final answer.\n\n"
            f"Question:\n{problem.question.strip()}\n\n"
            "Answer:"
        )

    def extract_answer(self, response_text: str) -> str | None:
        tagged = _extract_tagged_answer(response_text)
        if tagged is not None:
            return tagged
        lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
        if not lines:
            return None
        return lines[-1]

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        return _collapse_whitespace(extracted_answer) == _collapse_whitespace(problem.answer)


class ARCChallengeSpec(MultipleChoiceDatasetSpec):
    name = "arc_challenge"

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            choice_labels = [str(label) for label in example["choices"]["label"]]
            answer = str(example["answerKey"]).strip().upper()
            answer_index = choice_labels.index(answer)
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:05d}",
                    question=f'Question: {str(example["question"]).strip()}',
                    answer=_index_to_letter(answer_index),
                    source="allenai/ai2_arc:ARC-Challenge:test",
                    metadata={"choices": [str(text) for text in example["choices"]["text"]]},
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems


class WinograndeSpec(MultipleChoiceDatasetSpec):
    name = "winogrande"

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            sentence = str(example["sentence"])
            option1 = str(example["option1"])
            option2 = str(example["option2"])
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:05d}",
                    question=sentence.strip(),
                    answer=_index_to_letter(int(example["answer"]) - 1),
                    source="allenai/winogrande:winogrande_xl:validation",
                    metadata={"choices": [option1, option2]},
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems

    def build_prompt(self, problem: Problem) -> str:
        sentence = problem.question.strip()
        choices = list(problem.metadata.get("choices", []))
        if "_" not in sentence or len(choices) != 2:
            return super().build_prompt(problem)
        return (
            "Choose the option that best fills the blank.\n"
            "Respond with only the answer letter.\n\n"
            f"Sentence: {sentence}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            "Answer:"
        )


class MathLevel5Spec(DatasetSpecBase):
    name = "math_level_5"

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        value = text.strip()
        value = value.replace("$", "")
        value = value.replace("\\left", "").replace("\\right", "")
        value = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", value)
        value = _collapse_whitespace(value)
        return value

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        dataset_name_tried: list[str] = []
        dataset = None
        for dataset_name in ("hendrycks/competition_math", "nlile/hendrycks-MATH-benchmark"):
            dataset_name_tried.append(dataset_name)
            try:
                dataset = load_dataset(dataset_name, split="test")
                break
            except Exception:
                continue
        if dataset is None:
            tried = ", ".join(dataset_name_tried)
            raise RuntimeError(f"Failed to load a MATH dataset from any of: {tried}")

        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            if str(example.get("level", "")).strip() != "Level 5":
                continue
            problem_type = str(example.get("type", "")).strip()
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:05d}",
                    question=str(example["problem"]).strip(),
                    answer=str(example["solution"]).strip(),
                    source=f'{dataset_name}:{problem_type or "unknown"}:test',
                    metadata={"type": problem_type},
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems

    def build_prompt(self, problem: Problem) -> str:
        return (
            "Solve the following math problem.\n"
            "Put your final answer within \\boxed{}.\n\n"
            f"Problem:\n{problem.question.strip()}"
        )

    def extract_answer(self, response_text: str) -> str | None:
        extracted = _extract_last_boxed_or_tagged_answer(response_text)
        if extracted is not None:
            return extracted
        lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
        if not lines:
            return None
        return lines[-1]

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        gold = AIME20252026Spec._normalize_answer_text(
            _extract_last_boxed_or_tagged_answer(problem.answer) or problem.answer
        )
        pred = AIME20252026Spec._normalize_answer_text(extracted_answer)
        if pred == gold:
            return True
        return self._normalize_answer_text(pred) == self._normalize_answer_text(gold)


class CRUXEvalSpec(DatasetSpecBase):
    name = "cruxeval"

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            return self._load_problems_from_cache(cache_path)

        from datasets import load_dataset

        dataset = load_dataset("cruxeval-org/cruxeval", split="test")
        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            code = str(example["code"]).strip()
            input_text = str(example["input"]).strip()
            output_text = str(example["output"]).strip()
            question = (
                "You are given a Python function and an input.\n"
                "Predict the exact output of calling the function on that input."
            )
            problems.append(
                Problem(
                    problem_id=f"{self.name}_{i:05d}",
                    question=question,
                    answer=output_text,
                    source="cruxeval-org/cruxeval:test",
                    metadata={
                        "code": code,
                        "input": input_text,
                        "task": "output_prediction",
                    },
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems

    def build_prompt(self, problem: Problem) -> str:
        code = str(problem.metadata.get("code", "")).strip()
        input_text = str(problem.metadata.get("input", "")).strip()
        return (
            "You are given a Python function and an input.\n"
            "Return only the final Python output value.\n\n"
            "Function:\n"
            "```python\n"
            f"{code}\n"
            "```\n\n"
            f"Input:\n{input_text}\n\n"
            "Output:"
        )

    def extract_answer(self, response_text: str) -> str | None:
        tagged = _extract_tagged_answer(response_text)
        if tagged is not None:
            return tagged
        text = response_text.strip()
        text = re.sub(r"^```(?:python)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        return lines[-1]

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        if extracted_answer is None:
            return False
        try:
            return _safe_literal_eval(extracted_answer) == _safe_literal_eval(problem.answer)
        except Exception:
            return _normalize_free_form_answer(extracted_answer) == _normalize_free_form_answer(
                problem.answer
            )


class HLESpec(DatasetSpecBase):
    name = "hle"

    @staticmethod
    def _sequential_problem_id(index: int) -> str:
        return f"hle_{index:05d}"

    def _with_sequential_problem_ids(self, problems: list[Problem]) -> tuple[list[Problem], bool]:
        updated: list[Problem] = []
        changed = False
        for i, problem in enumerate(problems, start=1):
            metadata = dict(problem.metadata)
            original_hle_id = metadata.get("problem_id_hle") or metadata.get("id") or problem.problem_id
            metadata["problem_id_hle"] = str(original_hle_id)
            if "id" in metadata:
                metadata.pop("id")
                changed = True
            new_problem_id = self._sequential_problem_id(i)
            if problem.problem_id != new_problem_id:
                changed = True
            updated.append(
                Problem(
                    problem_id=new_problem_id,
                    question=problem.question,
                    answer=problem.answer,
                    source=problem.source,
                    metadata=metadata,
                )
            )
        return (updated if changed else problems), changed

    @staticmethod
    def _answer_to_text(answer: Any) -> str:
        if isinstance(answer, str):
            return answer
        return json.dumps(_json_safe_value(answer), ensure_ascii=False)

    def load_problems(self) -> list[Problem]:
        cache_path = self._dataset_cache_path()
        if cache_path.exists():
            problems, changed = self._with_sequential_problem_ids(
                self._load_problems_from_cache(cache_path)
            )
            if changed:
                self._save_problems_to_cache(cache_path, problems)
            return problems

        from datasets import load_dataset

        dataset = load_dataset("cais/hle", split="test")
        problems: list[Problem] = []
        for i, example in enumerate(dataset, start=1):
            row = {str(key): _json_safe_value(value) for key, value in dict(example).items()}
            image = str(row.get("image") or "").strip()
            problem_id_hle = str(row.get("id") or "")
            metadata = {
                "problem_id_hle": problem_id_hle,
                "answer_type": row.get("answer_type"),
                "image": image,
                "text_only": image == "",
                "category": row.get("category"),
                "raw_subject": row.get("raw_subject"),
                "raw_example": row,
            }
            problems.append(
                Problem(
                    problem_id=self._sequential_problem_id(i),
                    question=str(row.get("question") or "").strip(),
                    answer=self._answer_to_text(row.get("answer")),
                    source="cais/hle:test",
                    metadata=metadata,
                )
            )
        self._save_problems_to_cache(cache_path, problems)
        return problems

    def build_prompt(self, problem: Problem) -> str:
        return problem.question

    def extract_answer(self, response_text: str) -> str | None:
        tagged = _extract_tagged_answer(response_text)
        if tagged is not None:
            return tagged
        answer_line_matches = re.findall(
            r"(?im)^\s*answer\s*:\s*(.*?)\s*$",
            response_text.strip(),
        )
        if answer_line_matches:
            return answer_line_matches[-1].strip() or None
        lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def _normalize_multiple_choice(text: str | None) -> str | None:
        if text is None:
            return None
        value = _normalize_free_form_answer(text)
        match = re.fullmatch(r"(?i)(?:option|choice|answer)?\s*[\(\[]?([A-Z])[\)\].]?", value)
        if match is not None:
            return match.group(1).upper()
        leading_label_match = re.match(
            r"(?i)^\s*(?:option|choice|answer)?\s*[\(\[]?([A-Z])[\)\]]?\s*[\.:;-]\s+\S",
            value,
        )
        if leading_label_match is not None:
            return leading_label_match.group(1).upper()
        matches = re.findall(
            r"(?i)(?:answer|choice|option)\s*(?:is|:)?\s*[\(\[]?([A-Z])[\)\]]?",
            text,
        )
        if matches:
            return matches[-1].upper()
        return value.upper() if len(value) == 1 and value.isalpha() else None

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        try:
            payload = json.loads(stripped)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _judge_exact_match_response(self, response_text: str, problem: Problem) -> dict[str, Any]:
        from pydantic import BaseModel
        from openai import LengthFinishReasonError, OpenAI, OpenAIError

        class ExtractedAnswer(BaseModel):
            extracted_final_answer: str
            reasoning: str
            correct: Literal["yes", "no"]
            strict: Literal[True] = True

        prompt = HLE_JUDGE_PROMPT.format(
            question=problem.question,
            correct_answer=problem.answer,
            response=response_text,
        )
        try:
            client = OpenAI()
            request: dict[str, Any] = {
                "model": HLE_JUDGE_MODEL,
                "max_completion_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": ExtractedAnswer,
                "reasoning_effort": HLE_JUDGE_REASONING_EFFORT,
            }
            completion = client.beta.chat.completions.parse(**request)
        except (LengthFinishReasonError, OpenAIError) as exc:
            return {
                "is_correct": False,
                "extracted_answer": self.extract_answer(response_text),
                "metadata": {
                    "grader_type": "hle_official_style_llm_judge_error",
                    "judge_model": HLE_JUDGE_MODEL,
                    "judge_reasoning_effort": HLE_JUDGE_REASONING_EFFORT,
                    "answer_type": problem.metadata.get("answer_type"),
                    "judge_error_type": type(exc).__name__,
                    "judge_error": str(exc),
                },
            }
        content = completion.choices[0].message.parsed
        if content is None:
            return {
                "is_correct": False,
                "extracted_answer": self.extract_answer(response_text),
                "metadata": {
                    "grader_type": "hle_official_style_llm_judge_error",
                    "judge_model": HLE_JUDGE_MODEL,
                    "judge_reasoning_effort": HLE_JUDGE_REASONING_EFFORT,
                    "answer_type": problem.metadata.get("answer_type"),
                    "judge_error_type": "NoParsedResponse",
                    "judge_error": "HLE judge returned no parsed response.",
                },
            }
        extracted = content.extracted_final_answer
        is_correct = content.correct == "yes"
        return {
            "is_correct": is_correct,
            "extracted_answer": extracted,
            "metadata": {
                "grader_type": "hle_official_style_llm_judge",
                "judge_model": HLE_JUDGE_MODEL,
                "judge_reasoning_effort": HLE_JUDGE_REASONING_EFFORT,
                "answer_type": problem.metadata.get("answer_type"),
                "reasoning": content.reasoning,
            },
        }

    def grade_response(self, response_text: str, problem: Problem) -> dict[str, Any]:
        answer_type = str(problem.metadata.get("answer_type") or "")
        extracted_answer = self.extract_answer(response_text)
        if answer_type == "multipleChoice":
            normalized_pred = self._normalize_multiple_choice(extracted_answer)
            normalized_gold = self._normalize_multiple_choice(problem.answer)
            return {
                "is_correct": normalized_pred is not None and normalized_pred == normalized_gold,
                "extracted_answer": normalized_pred,
                "metadata": {
                    "grader_type": "hle_multiple_choice_exact_match",
                    "answer_type": answer_type,
                    "raw_extracted_answer": extracted_answer,
                    "gold_answer": normalized_gold,
                },
            }
        if answer_type == "exactMatch":
            return self._judge_exact_match_response(response_text, problem)
        return {
            "is_correct": _normalize_free_form_answer(extracted_answer or "")
            == _normalize_free_form_answer(problem.answer),
            "extracted_answer": extracted_answer,
            "metadata": {
                "grader_type": "hle_unknown_answer_type_fallback",
                "answer_type": answer_type,
            },
        }

    def is_correct(self, extracted_answer: str | None, problem: Problem) -> bool:
        answer_type = str(problem.metadata.get("answer_type") or "")
        if answer_type == "multipleChoice":
            normalized_pred = self._normalize_multiple_choice(extracted_answer)
            normalized_gold = self._normalize_multiple_choice(problem.answer)
            return normalized_pred is not None and normalized_pred == normalized_gold
        if answer_type == "exactMatch":
            # Exact-match HLE rows require the full model response for the official-style judge.
            # Use grade_response() when evaluating model outputs.
            return _normalize_free_form_answer(extracted_answer or "") == _normalize_free_form_answer(
                problem.answer
            )
        return _normalize_free_form_answer(extracted_answer or "") == _normalize_free_form_answer(
            problem.answer
        )


def get_dataset_spec(benchmark_name: str) -> DatasetSpecBase:
    specs = {
        "aime2025_2026": AIME20252026Spec(),
        "hellaswag": HellaSwagSpec(),
        "piqa": PIQASpec(),
        "mmlu": MMLUSpec(),
        "bbh": BBHSpec(),
        "arc_challenge": ARCChallengeSpec(),
        "arc_ai2": ARCChallengeSpec(),
        "winogrande": WinograndeSpec(),
        "math_level_5": MathLevel5Spec(),
        "mathlevel5": MathLevel5Spec(),
        "cruxeval": CRUXEvalSpec(),
        "hle": HLESpec(),
        "cais/hle": HLESpec(),
    }
    return specs[benchmark_name.lower()]
