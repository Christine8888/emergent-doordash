from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict
import re

from src.datasets import Problem
from src.storage import build_hint_generation_path, read_jsonl

class HintType(str, Enum):
    masked = "masked"
    bag_of_hints = "bag_of_hints"
    basic_hint = "basic_hint"


class HintGenerationContext(TypedDict, total=False):
    source_benchmark_name: str
    source_hint_type: str
    source_data_path: str
    source_hint_id: str
    source_rollout_id: int
    source_generator_model: str
    source_model_output: str
    source_answer: str


class HintTypeSpecBase(ABC):
    name: HintType
    prompt_version: str
    post_process_version: str
    grade_model_output: bool

    def __init__(
        self,
        *,
        name: HintType,
        prompt_version: str,
        post_process_version: str,
        grade_model_output: bool,
        uses_context: bool | None = None,
        required_context_keys: tuple[str, ...] = (),
        source_hint_type: HintType | None = None,
    ) -> None:
        self.name = name
        self.prompt_version = prompt_version
        self.post_process_version = post_process_version
        self.grade_model_output = grade_model_output
        self.source_hint_type = source_hint_type
        if uses_context is None:
            uses_context = source_hint_type is not None

        if source_hint_type is not None and not required_context_keys:
            required_context_keys = (
                "source_benchmark_name",
                "source_hint_type",
                "source_data_path",
                "source_hint_id",
                "source_rollout_id",
                "source_generator_model",
                "source_model_output",
                "source_answer",
            )
        self.uses_context = uses_context
        self.required_context_keys = required_context_keys
        self._source_rows_by_problem_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def _source_rows_by_problem(self, benchmark_name: str) -> dict[str, list[dict[str, Any]]]:
        if self.source_hint_type is None:
            raise ValueError(f"Hint type {self.name.value!r} has no source_hint_type configured.")

        cache_key = f"{benchmark_name}::{self.source_hint_type.value}"
        if cache_key in self._source_rows_by_problem_cache:
            return self._source_rows_by_problem_cache[cache_key]

        path = build_hint_generation_path(
            benchmark_name=benchmark_name,
            hint_type=self.source_hint_type.value,
            data_root="data",
        )
        source_path = Path(path)
        if not source_path.exists():
            raise ValueError(
                f"Missing source hints for derived hint type {self.name.value!r}. "
                f"Expected file: {source_path}. Generate {self.source_hint_type.value!r} first."
            )

        rows = read_jsonl(path, model_cls=None)
        by_problem: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            problem_id = str(row["problem_id"])
            by_problem.setdefault(problem_id, []).append(row)

        for problem_rows in by_problem.values():
            problem_rows.sort(
                key=lambda row: (
                    int(row.get("rollout_id", 0)),
                    str(row.get("created_at", "")),
                    str(row.get("hint_id", "")),
                )
            )

        self._source_rows_by_problem_cache[cache_key] = by_problem
        return by_problem

    def build_context(
        self,
        *,
        benchmark_name: str,
        problem: Problem,
        rollout_id: int,
    ) -> HintGenerationContext:
        if self.source_hint_type is None:
            return {}

        by_problem = self._source_rows_by_problem(benchmark_name)
        if problem.problem_id not in by_problem:
            raise ValueError(
                f"Missing source rows for problem_id={problem.problem_id!r} "
                f"in derived hint type {self.name.value!r}. "
                f"Generate {self.source_hint_type.value!r} for this benchmark/problem first."
            )

        source_rows = by_problem[problem.problem_id]
        matching_rows: list[dict[str, Any]] = []
        for row in source_rows:
            row_rollout_id = row.get("rollout_id")
            if isinstance(row_rollout_id, str) and row_rollout_id.isdigit():
                row_rollout_id = int(row_rollout_id)
            if row_rollout_id == rollout_id:
                matching_rows.append(row)

        if not matching_rows:
            available_rollout_ids = sorted(
                {
                    int(row["rollout_id"])
                    for row in source_rows
                    if isinstance(row.get("rollout_id"), int)
                    or (isinstance(row.get("rollout_id"), str) and str(row.get("rollout_id")).isdigit())
                }
            )
            raise ValueError(
                f"Missing source rollout_id={rollout_id} for problem_id={problem.problem_id!r} "
                f"in derived hint type {self.name.value!r}. "
                f"Available source rollout_ids={available_rollout_ids}. "
                f"Generate {self.source_hint_type.value!r} with enough rollouts first."
            )

        source_row = matching_rows[-1]
        source_path = build_hint_generation_path(
            benchmark_name=benchmark_name,
            hint_type=self.source_hint_type.value,
            data_root="data",
        )
        return {
            "source_benchmark_name": benchmark_name,
            "source_hint_type": str(source_row["hint_type"]),
            "source_data_path": str(source_path),
            "source_hint_id": str(source_row["hint_id"]),
            "source_rollout_id": int(source_row["rollout_id"]),
            "source_generator_model": str(source_row["generator_model"]),
            "source_model_output": str(source_row["model_output"]),
            "source_answer": str(source_row.get("answer", problem.answer)),
        }

    def _validate_context(self, *, context: HintGenerationContext, stage: str) -> None:
        if not self.uses_context and len(context) > 0:
            keys = sorted(context.keys())
            raise ValueError(
                f"Hint type {self.name.value!r} does not use context, but got context keys {keys} during {stage}."
            )

        missing = [key for key in self.required_context_keys if key not in context]
        if missing:
            raise ValueError(
                f"Hint type {self.name.value!r} is missing required context keys {missing} during {stage}."
            )

    def build_prompt(
        self,
        *,
        problem: Problem,
        context: HintGenerationContext,
    ) -> str:
        self._validate_context(context=context, stage="build_prompt")
        return self._build_prompt(problem=problem, context=context)

    def post_process(
        self,
        *,
        model_output: str,
        context: HintGenerationContext,
    ) -> str:
        self._validate_context(context=context, stage="post_process")
        return self._post_process(model_output=model_output, context=context)

    @abstractmethod
    def _build_prompt(
        self,
        *,
        problem: Problem,
        context: HintGenerationContext,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def _post_process(
        self,
        *,
        model_output: str,
        context: HintGenerationContext,
    ) -> str:
        raise NotImplementedError

    def context_metadata(self, context: HintGenerationContext) -> dict[str, Any]:
        if self.source_hint_type is None:
            return {}
        metadata: dict[str, Any] = {}
        for key in (
            "source_benchmark_name",
            "source_hint_type",
            "source_data_path",
            "source_hint_id",
            "source_rollout_id",
            "source_generator_model",
        ):
            if key in context:
                metadata[key] = context[key]
        return metadata


class BasicHintTypeSpec(HintTypeSpecBase):
    def __init__(
        self,
        *,
        name: HintType = HintType.basic_hint,
        prompt_version: str = "basic_v1",
        post_process_version: str = "basic_post_v1",
        grade_model_output: bool = True,
        source_hint_type: HintType | None = None,
    ) -> None:
        super().__init__(
            name=name,
            prompt_version=prompt_version,
            post_process_version=post_process_version,
            grade_model_output=grade_model_output,
            source_hint_type=source_hint_type,
        )

    def _build_prompt(
        self,
        *,
        problem: Problem,
        context: HintGenerationContext,
    ) -> str:
        _ = context
        template = (
            "Write a detailed solution to the following problem.\n"
            "Provide the answer in <answer></answer> tags.\n"
            "Here is the problem: {question}"
        )
        return template.format(question=problem.question)

    def _post_process(
        self,
        *,
        model_output: str,
        context: HintGenerationContext,
    ) -> str:
        _ = context
        return model_output


class MaskedHintTypeSpec(HintTypeSpecBase):
    def __init__(self) -> None:
        super().__init__(
            name=HintType.masked,
            prompt_version="masked_v1",
            post_process_version="masked_post_v1",
            grade_model_output=False,
            source_hint_type=HintType.basic_hint,
        )

    def _build_prompt(
        self,
        *,
        problem: Problem,
        context: HintGenerationContext,
    ) -> str:
        template = ( # NOTE: this template is optimized to not reveal the answer at all. It's quite good at it when you ask claude opus
            "You will be given a problem and a reference solution. "
            "Rewrite the reference solution as a detailed explanation. "
            "The final answer {source_answer} must not appear anywhere in your explanation. "
            "You may compute all intermediate values, but do not perform the final step that directly produces {source_answer} — "
            "stop your explanation at the last intermediate result.\n\n"
            "Problem:\n"
            "{question}\n\n"
            "Reference full solution:\n"
            "{source_solution}\n\n"
            "REMINDER: Do not perform the final step. Your explanation must not contain {source_answer} anywhere."
        )
        return template.format(
            question=problem.question,
            source_answer=context["source_answer"],
            source_solution=context["source_model_output"],
        )

    def _post_process(
        self,
        *,
        model_output: str,
        context: HintGenerationContext,
    ) -> str:
        return model_output



class BaggedHintTypeSpec(HintTypeSpecBase):
    def __init__(self) -> None:
        super().__init__(
            name=HintType.bag_of_hints,
            prompt_version="bag_of_hints_v1",
            post_process_version="bag_of_hints_post_v1",
            grade_model_output=False,
            source_hint_type=HintType.basic_hint,
        )

    def _build_prompt(
        self,
        *,
        problem: Problem,
        context: HintGenerationContext,
    ) -> str:
        template = (
            "You will be given a problem and a reference solution. "
            "Create 10 standalone hints that could be given to a student who only has the problem statement. "
            "Each hint should reveal exactly one useful observation or technique — "
            "no hint should state specific intermediate values, name specific answers, or give away the solution structure. "
            "All hints should be at the same level of abstraction: pointing toward an approach or insight, not spelling it out. "
            "Each hint must be distinct — do not repeat the same idea in different words across multiple hints. "
            "The final answer {source_answer} must not appear in any hint.\n\n"
            "Format your response as exactly 10 hints using this structure:\n"
            "<hint id=1>hint text here</hint>\n"
            "<hint id=2>hint text here</hint>\n"
            "...and so on up to id=10.\n"
            "Do not include any text outside of the hint tags.\n\n"
            "Problem:\n"
            "{question}\n\n"
            "Reference full solution:\n"
            "{source_solution}"
        )
        return template.format(
            question=problem.question,
            source_answer=context["source_answer"],
            source_solution=context["source_model_output"],
        )

    def _post_process(
        self,
        *,
        model_output: str,
        context: HintGenerationContext,
    ) -> str:
        _ = context
        return model_output



HINT_TYPE_SPECS: dict[str, HintTypeSpecBase] = {
    HintType.masked.value: MaskedHintTypeSpec(),
    HintType.basic_hint.value: BasicHintTypeSpec(),
    HintType.bag_of_hints.value: BaggedHintTypeSpec(),
}


def get_hint_type_spec(hint_type: str) -> HintTypeSpecBase:
    return HINT_TYPE_SPECS[hint_type]
