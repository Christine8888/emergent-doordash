from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
import re


INPUT_PATH = Path(
    "suze_experiments/20260321/consolidated_hinted_results_v2/"
    "aime_solution/Qwen3-14B/hint_fraction_1.0/solution_intext_masked.jsonl"
)
TOY_OUTPUT_DIR = Path("suze_experiments/20260321/regrader_toy_outputs")
OUTPUT_PATH = TOY_OUTPUT_DIR / "qwen3_14b_hint1_solution_intext_masked.new_incorrect_only.json"

OLD_SCORER_NAME = "aime_scorer"
NEW_SCORER_NAME = "aime_scorer_v2"

MAX_EXAMPLES_PER_BUCKET = 5
PROMPT_PREVIEW_CHARS = 600
OUTPUT_PREVIEW_CHARS = 600


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environments.math.utils import grade_math_answer



def extract_answer_fixed(completion: str) -> str:
    r"""
    Improved answer extraction for math-style tasks.

    Differences from extract_answer():
    - Supports markdown-decorated answer lines such as "**ANSWER: $432$**"
    - Supports boxed answers inside answer lines, e.g. "ANSWER: $\boxed{432}$"
    - Applies latex/markdown cleanup consistently to boxed fallbacks
    """
    # Match labeled answer lines, including variants like:
    # - ANSWER: 432
    # - Final answer: 432
    # - Target answer:       60
    # - ANSWER: 60
    pattern = (
        r"(?im)(?:^|\n)\s*(?:[\*\-_`>#]+\s*)?"
        r"(?:target\s+answer|final\s+answer|answer)\s*:[ \t]*([^\n]+)"
    )
    matches = list(re.finditer(pattern, completion, re.MULTILINE))
    if matches:
        raw_answer = matches[-1].group(1).strip()
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer:
            return clean_latex_and_markdown(remove_boxed(boxed_answer))
        cleaned = clean_latex_and_markdown(raw_answer)
        if cleaned:
            return cleaned

    # Fall back to last boxed expression in the completion.
    boxed_answer = last_boxed_only_string(completion)
    if boxed_answer:
        return clean_latex_and_markdown(remove_boxed(boxed_answer))

    # Final fallback: use the last full number in the solution text.
    number = extract_last_full_number(completion)
    if number is not None:
        return number

    return ""


# Both remove_boxed() and last_boxed_only_string() functions borrowed from:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L53C1-L94C18
def remove_boxed(s: str) -> str:
    s = s.strip()
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :].strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1].strip()
    return s


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        tail = string.split("\\boxed ")[-1]
        token = tail.split("$")[0].strip()
        return "\\boxed " + token if token else None
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]



def clean_latex_and_markdown(text: str) -> str:
    """Remove LaTeX delimiters and markdown formatting from text.

    Removes:
    - LaTeX delimiters: $...$, \\(...\\), \\[...\\]
    - Markdown bold: **...**
    - \\boxed{} wrapper
    - Extra whitespace

    Args:
        text: Raw extracted answer text

    Returns:
        Cleaned text with formatting removed
    """
    # Remove markdown bold (**text**)
    text = re.sub(r'\*\*', '', text)

    # Remove LaTeX display delimiters \[ \]
    text = re.sub(r'\\\[|\\\]', '', text)

    # Remove LaTeX inline delimiters \( \)
    text = re.sub(r'\\\(|\\\)', '', text)

    # Remove $ delimiters
    text = re.sub(r'\$', '', text)

    # Remove \boxed{} wrapper (handles nested braces properly)
    text = text.strip()
    if text.startswith('\\boxed{') and text.endswith('}'):
        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    # If this closing brace is at the end, extract contents
                    if i == len(text) - 1:
                        text = text[7:-1]  # Remove \boxed{ and }
                    break

    # Remove common trailing markdown punctuation.
    text = text.rstrip(" .,:;!`*_")

    # Strip leading labels if present in extracted text.
    text = re.sub(r"(?i)^\s*(?:target\s+answer|final\s+answer|answer)\s*:\s*", "", text).strip()

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_last_full_number(text: str) -> str | None:
    # Match full numeric tokens, avoiding partial captures inside words.
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9_])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])",
            text,
        )
    )
    if not matches:
        return None
    value = matches[-1].group(0)
    return value.replace(",", "")


async def grade_rollout_with_aime_scorer_v2(rollout: dict[str, Any]) -> dict[str, Any]:
    output_text = str(rollout.get("output_text") or "")
    target = rollout.get("target")
    target_text = str(target).strip() if target is not None else None

    # extracted_answer = extract_answer(output_text) # this is the original functionality
    extracted_answer = extract_answer_fixed(output_text)
    extraction_status = "ok" if extracted_answer and str(extracted_answer).strip() != "" else "failed"

    if extraction_status == "ok" and target_text is not None and target_text != "":
        is_correct = await grade_math_answer(
            answer=str(extracted_answer),
            target=target_text,
            exact_match=True,
            use_sympy=True,
        )
        score_raw_value = "C" if is_correct else "I"
    else:
        is_correct = False
        score_raw_value = "I"

    return {
        "score_raw_value": score_raw_value,
        "score_normalized": score_raw_value,
        "is_correct": is_correct,
        "extracted_answer": extracted_answer,
        "extraction_status": extraction_status,
    }

def parse_status(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "unknown"




def add_example(bucket: list[dict[str, Any]], rollout: dict[str, Any], sample_id: Any, old_status: str, new_status: str, old_outcome: dict[str, Any], new_outcome: dict[str, Any]) -> None:
    if len(bucket) >= MAX_EXAMPLES_PER_BUCKET:
        return
    bucket.append(
        {
            "sample_id": sample_id,
            "epoch": rollout.get("epoch"),
            "rollout_id": rollout.get("rollout_id"),
            "target": rollout.get("target"),
            "old_status": old_status,
            "new_status": new_status,
            "extracted_answer_old": old_outcome.get("extracted_answer"),
            "extracted_answer_v2": new_outcome.get("extracted_answer"),
            "extraction_status_v2": new_outcome.get("extraction_status"),
            "prompt_preview": rollout.get("prompt_text"),
            "output_preview": rollout.get("output_text"),
        }
    )


def print_examples(title: str, rows: list[dict[str, Any]]) -> None:
    print()
    print(f"=== {title} ({len(rows)}) ===")
    if not rows:
        print("none")
        return
    for i, row in enumerate(rows, start=1):
        print(f"\n[{i}] sample_id={row['sample_id']} epoch={row['epoch']} rollout_id={row['rollout_id']}")
        print(f"target={row['target']} old={row['old_status']} new={row['new_status']}")
        print(f"extracted_v2={row['extracted_answer_v2']} extraction_status_v2={row['extraction_status_v2']}")
        print(f"prompt: {row['prompt_preview']}")
        print(f"output: {row['output_preview']}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    TOY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    totals = {
        "samples": 0,
        "rollouts": 0,
        "old_correct": 0,
        "old_incorrect": 0,
        "old_unknown": 0,
        "new_correct": 0,
        "new_incorrect": 0,
        "new_unknown": 0,
        "flip_to_correct": 0,
        "flip_from_correct": 0,
    }

    changed_examples: list[dict[str, Any]] = []
    new_correct_examples: list[dict[str, Any]] = []
    new_incorrect_examples: list[dict[str, Any]] = []
    new_incorrect_rows_for_json: list[dict[str, Any]] = []

    print(f"Reading: {INPUT_PATH}")
    print(f"Writing new incorrect rollouts to: {OUTPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as in_f:
        for line_number, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue
            sample_obj = json.loads(line)
            totals["samples"] += 1
            sample_id = sample_obj.get("sample_id")
            rollouts = sample_obj.get("rollouts")
            if not isinstance(rollouts, list):
                continue

            for rollout in rollouts:
                if not isinstance(rollout, dict):
                    continue
                totals["rollouts"] += 1

                rollout_copy = dict(rollout)
                score_outcomes = rollout_copy.get("score_outcomes")
                if isinstance(score_outcomes, dict):
                    score_outcomes_copy = dict(score_outcomes)
                else:
                    score_outcomes_copy = {}

                old_outcome = score_outcomes_copy.get(OLD_SCORER_NAME)
                if not isinstance(old_outcome, dict):
                    old_outcome = {}
                old_status = parse_status(old_outcome.get("is_correct"))

                if old_status == "correct":
                    totals["old_correct"] += 1
                elif old_status == "incorrect":
                    totals["old_incorrect"] += 1
                else:
                    totals["old_unknown"] += 1

                new_outcome = asyncio.run(grade_rollout_with_aime_scorer_v2(rollout_copy))
                new_status = parse_status(new_outcome.get("is_correct"))
                score_outcomes_copy[NEW_SCORER_NAME] = new_outcome

                if new_status == "correct":
                    totals["new_correct"] += 1
                elif new_status == "incorrect":
                    totals["new_incorrect"] += 1
                else:
                    totals["new_unknown"] += 1

                if old_status != new_status:
                    add_example(changed_examples, rollout_copy, sample_id, old_status, new_status, old_outcome, new_outcome)
                if old_status != "correct" and new_status == "correct":
                    totals["flip_to_correct"] += 1
                if old_status == "correct" and new_status != "correct":
                    totals["flip_from_correct"] += 1

                if new_status == "correct":
                    add_example(new_correct_examples, rollout_copy, sample_id, old_status, new_status, old_outcome, new_outcome)
                if new_status == "incorrect":
                    add_example(new_incorrect_examples, rollout_copy, sample_id, old_status, new_status, old_outcome, new_outcome)
                    new_incorrect_rows_for_json.append(
                        {
                            "sample_id": sample_id,
                            "epoch": rollout_copy.get("epoch"),
                            "rollout_id": rollout_copy.get("rollout_id"),
                            "target": rollout_copy.get("target"),
                            "old_score_outcome": old_outcome,
                            "new_score_outcome": new_outcome,
                            "prompt_text": rollout_copy.get("prompt_text"),
                            "output_text": rollout_copy.get("output_text"),
                        }
                    )

            if line_number % 200 == 0:
                print(f"processed lines={line_number} samples={totals['samples']} rollouts={totals['rollouts']}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(new_incorrect_rows_for_json, f, ensure_ascii=False, indent=2)

    print()
    print("=== Summary ===")
    print(f"samples={totals['samples']}")
    print(f"rollouts={totals['rollouts']}")
    print(f"old_correct={totals['old_correct']} old_incorrect={totals['old_incorrect']} old_unknown={totals['old_unknown']}")
    print(f"new_correct={totals['new_correct']} new_incorrect={totals['new_incorrect']} new_unknown={totals['new_unknown']}")
    print(f"delta_correct={totals['new_correct'] - totals['old_correct']}")
    print(f"flip_to_correct={totals['flip_to_correct']} flip_from_correct={totals['flip_from_correct']}")
    print(f"new_incorrect_json_file={OUTPUT_PATH}")
    print(f"new_incorrect_json_rows={len(new_incorrect_rows_for_json)}")

    # print_examples("New scorer correct examples", new_correct_examples)
    # print_examples("New scorer incorrect examples", new_incorrect_examples)
    print(f"\n=== Still incorrect with new grader (showing up to 3) ===")
    for i, ex in enumerate(new_incorrect_examples[:1], start=1):
        print(f"\n--- [{i}] sample_id={ex['sample_id']} ---")
        print(f"  Target answer:       \"{ex['target']}\"")
        print(f"  Old extracted answer: \"{ex['extracted_answer_old']}\"")
        print(f"  Old score:           \"{ex['old_status']}\"")
        print(f"  New extracted answer: \"{ex['extracted_answer_v2']}\"")
        print(f"  New score:           \"{ex['new_status']}\"")
        print(f"  Full prompt:\n\"{ex['prompt_preview']}\"")
        print(f"  Full output:\n\"{ex['output_preview']}\"")


if __name__ == "__main__":
    # python suze_experiments/20260321/regrade_qwen3_14b_hint1_masked.py
    main()
