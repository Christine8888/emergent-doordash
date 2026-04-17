from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from inspect_evals.constants import DEFAULT_FEWSHOT_SEED
from inspect_evals.math.math import EVAL_VERSION, MATH_DATASET_REVISION, DATASET_PATH, math_solver
from inspect_evals.math.utils import (
    MathLevel,
    MathSubject,
    filter_dataset,
    is_equiv_sympy,
    last_boxed_only_string,
    normalize_final_answer,
    record_to_sample,
    remove_boxed,
    strip_string,
)
from inspect_evals.utils.huggingface import hf_dataset


def _strip_wrapping_math_delimiters(text: str) -> str:
    value = text.strip()
    changed = True
    while changed:
        changed = False
        if value.startswith("$$") and value.endswith("$$") and len(value) >= 4:
            value = value[2:-2].strip()
            changed = True
        elif value.startswith("$") and value.endswith("$") and len(value) >= 2:
            value = value[1:-1].strip()
            changed = True
        elif value.startswith(r"\[") and value.endswith(r"\]") and len(value) >= 4:
            value = value[2:-2].strip()
            changed = True
    return value


def _trim_unbalanced_braces(text: str) -> str:
    value = text.strip()
    while value.startswith("}"):
        value = value[1:].lstrip()
    while value and value.count("{") < value.count("}") and value.endswith("}"):
        value = value[:-1].rstrip()
    return value


def _unwrap_outer_command(text: str, command: str) -> str | None:
    prefix = f"\\{command}" + "{"
    if not text.startswith(prefix) or not text.endswith("}"):
        return None
    depth = 0
    for idx, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if idx == len(text) - 1:
                    return text[len(prefix) : -1]
                return None
    return None


def _sanitize_candidate(candidate: str) -> str:
    value = candidate.strip()
    previous = None
    while value and value != previous:
        previous = value
        value = value.strip()
        value = value.strip("*").strip()
        value = _strip_wrapping_math_delimiters(value)
        value = _trim_unbalanced_braces(value)

        boxed = last_boxed_only_string(value)
        if boxed == value:
            try:
                value = remove_boxed(boxed).strip()
                continue
            except Exception:
                pass

        for command in ("text", "textbf", "mathrm", "operatorname"):
            inner = _unwrap_outer_command(value, command)
            if inner is not None:
                value = inner.strip()
                break

        value = re.sub(r"(?is)^ANSWER\s*:\s*", "", value).strip()
        value = re.sub(r"(?is)^FINAL\s+ANSWER\s*:\s*", "", value).strip()
        value = re.sub(r"(?is)^\\text\{\s*ANSWER:\s*\}\s*", "", value).strip()
        value = re.sub(r"(?is)^\\textbf\{\s*ANSWER:\s*\}\s*", "", value).strip()
        value = _trim_unbalanced_braces(value)
    return value


def _extract_candidates(completion: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(method: str, raw: str | None) -> None:
        if not raw:
            return
        cleaned = _sanitize_candidate(raw)
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append((method, cleaned))

    matches = list(re.finditer(r"(?i)ANSWER\s*:\s*([^\n]+)\s*\Z", completion))
    if matches:
        add("answer_line_strict", matches[-1].group(1))

    relaxed_matches = re.findall(r"(?im)ANSWER\s*:\s*([^\n]+)", completion)
    if relaxed_matches:
        add("answer_line_relaxed", relaxed_matches[-1])

    boxed = last_boxed_only_string(completion)
    if boxed:
        try:
            add("boxed", remove_boxed(boxed))
        except Exception:
            add("boxed", boxed)

    for display in re.findall(r"\$\$(.*?)\$\$", completion, flags=re.S):
        add("display_math", display)

    for display in re.findall(r"\\\[(.*?)\\\]", completion, flags=re.S):
        add("display_brackets", display)

    non_empty_lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if non_empty_lines:
        add("last_line", non_empty_lines[-1])

    return candidates


def _canon_for_numeric_compare(text: str) -> str:
    value = text.strip()
    value = value.replace(r"\$", "")
    value = value.replace(r"\!", "")
    value = value.replace(",", "")
    value = value.replace(" ", "")
    value = value.replace(r"^\circ", "")
    value = value.replace(r"\circ", "")
    return value


def _split_top_level_csv(text: str) -> list[str] | None:
    if r"\cup" in text:
        return None
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts if len(parts) > 1 else None


def _sympy_text_ready(text: str) -> str:
    value = text.strip()
    replacements = {
        r"\pi": "pi",
        r"\sqrt": "sqrt",
        "^": "**",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def _sympy_text_equiv(answer: str, target: str) -> bool:
    try:
        import sympy
    except Exception:
        return False

    try:
        parsed_answer = sympy.sympify(_sympy_text_ready(answer))
        parsed_target = sympy.sympify(_sympy_text_ready(target))
        return bool(sympy.simplify(parsed_answer - parsed_target) == 0)
    except Exception:
        return False


async def _unordered_csv_match(answer: str, target: str) -> bool:
    answer_parts = _split_top_level_csv(answer)
    target_parts = _split_top_level_csv(target)
    if answer_parts is None or target_parts is None or len(answer_parts) != len(target_parts):
        return False
    answer_norm = sorted([await strip_string(part) for part in answer_parts])
    target_norm = sorted([await strip_string(part) for part in target_parts])
    return answer_norm == target_norm


async def _match_answer(answer: str, target: str) -> str | None:
    if answer == target:
        return "raw_exact"

    try:
        stripped_answer = await strip_string(answer)
        stripped_target = await strip_string(target)
        if stripped_answer == stripped_target:
            return "strip_string"
    except Exception:
        pass

    try:
        normalized_answer = await normalize_final_answer(answer)
        normalized_target = await normalize_final_answer(target)
        if normalized_answer == normalized_target:
            return "normalized_exact"

        stripped_normalized_answer = await strip_string(normalized_answer)
        stripped_normalized_target = await strip_string(normalized_target)
        if stripped_normalized_answer == stripped_normalized_target:
            return "normalized_strip_string"

        if await is_equiv_sympy(normalized_answer, normalized_target):
            return "sympy"
    except Exception:
        pass

    if _sympy_text_equiv(answer, target):
        return "sympy_text"

    if _canon_for_numeric_compare(answer) == _canon_for_numeric_compare(target):
        return "numeric_format"

    try:
        if await _unordered_csv_match(answer, target):
            return "unordered_csv"
    except Exception:
        pass

    return None


@scorer(metrics=[accuracy(), stderr()])
def deterministic_math_match() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        candidates = _extract_candidates(state.output.completion)
        for method, candidate in candidates:
            match_method = await _match_answer(candidate, target.text)
            if match_method is not None:
                return Score(
                    value=CORRECT,
                    explanation=state.output.completion,
                    metadata={
                        "extracted_answer": candidate,
                        "extraction_method": method,
                        "match_method": match_method,
                        "candidates": [candidate_text for _, candidate_text in candidates],
                    },
                )

        return Score(
            value=INCORRECT,
            explanation=state.output.completion,
            metadata={
                "extracted_answer": candidates[0][1] if candidates else "",
                "extraction_method": candidates[0][0] if candidates else None,
                "match_method": None,
                "candidates": [candidate_text for _, candidate_text in candidates],
            },
        )

    return score


@task
def math_relaxed(
    levels: list[MathLevel] | MathLevel = [],
    subjects: list[MathSubject] | MathSubject = [],
    fewshot: int = 0,
    fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    shuffle: bool = True,
) -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="test",
        name="default",
        sample_fields=record_to_sample,
        shuffle=shuffle,
        revision=MATH_DATASET_REVISION,
    )
    dataset = filter_dataset(dataset=dataset, levels=levels, subjects=subjects)

    return Task(
        dataset=dataset,
        solver=math_solver(fewshot=fewshot, fewshot_seed=fewshot_seed),
        scorer=[deterministic_math_match()],
        config=GenerateConfig(temperature=0.5),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata() | {"custom_grader": "deterministic_math_match"},
    )
