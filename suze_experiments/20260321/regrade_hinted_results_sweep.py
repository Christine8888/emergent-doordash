from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any


# --- Editable constants ---
INPUT_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2")
OUTPUT_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2_regraded")
NEW_SCORER_NAME = "aime_scorer_v2"
OLD_SCORER_NAME = "aime_scorer"
FILE_GLOB = "*.jsonl"
SKIP_EXISTING_OUTPUT_FILES = True
PROGRESS_EVERY_SAMPLES = 200
# --------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environments.math.utils import grade_math_answer


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def suppress_known_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*antlr4\.error\.ErrorListener module is not installed.*",
        category=UserWarning,
    )


def parse_status(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "unknown"


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
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\\\[|\\\]", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"\$", "", text)

    text = text.strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        depth = 0
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    if i == len(text) - 1:
                        text = text[7:-1]
                    break

    text = text.rstrip(" .,:;!`*_")
    text = re.sub(r"(?i)^\s*(?:target\s+answer|final\s+answer|answer)\s*:\s*", "", text).strip()
    return text


def extract_last_full_number(text: str) -> str | None:
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9_])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])",
            text,
        )
    )
    if not matches:
        return None
    return matches[-1].group(0).replace(",", "")


def extract_answer_fixed(completion: str) -> str:
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

    boxed_answer = last_boxed_only_string(completion)
    if boxed_answer:
        return clean_latex_and_markdown(remove_boxed(boxed_answer))

    number = extract_last_full_number(completion)
    if number is not None:
        return number

    return ""


async def grade_rollout_with_aime_scorer_v2(rollout: dict[str, Any]) -> dict[str, Any]:
    output_text = str(rollout.get("output_text") or "")
    target = rollout.get("target")
    target_text = str(target).strip() if target is not None else None

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


async def process_file(input_path: Path, output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    stats = {
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

    with input_path.open("r", encoding="utf-8") as in_f, tmp_path.open("w", encoding="utf-8") as out_f:
        for line_number, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue

            sample_obj = json.loads(line)
            stats["samples"] += 1

            rollouts = sample_obj.get("rollouts")
            if not isinstance(rollouts, list):
                out_f.write(json.dumps(sample_obj, ensure_ascii=False) + "\n")
                continue

            updated_rollouts: list[dict[str, Any]] = []
            for rollout in rollouts:
                if not isinstance(rollout, dict):
                    continue
                stats["rollouts"] += 1

                rollout_copy = dict(rollout)
                score_outcomes = rollout_copy.get("score_outcomes")
                score_outcomes_copy = dict(score_outcomes) if isinstance(score_outcomes, dict) else {}

                old_outcome = score_outcomes_copy.get(OLD_SCORER_NAME)
                if not isinstance(old_outcome, dict):
                    old_outcome = {}
                old_status = parse_status(old_outcome.get("is_correct"))
                if old_status == "correct":
                    stats["old_correct"] += 1
                elif old_status == "incorrect":
                    stats["old_incorrect"] += 1
                else:
                    stats["old_unknown"] += 1

                new_outcome = await grade_rollout_with_aime_scorer_v2(rollout_copy)
                new_status = parse_status(new_outcome.get("is_correct"))
                score_outcomes_copy[NEW_SCORER_NAME] = new_outcome
                rollout_copy["score_outcomes"] = score_outcomes_copy
                updated_rollouts.append(rollout_copy)

                if new_status == "correct":
                    stats["new_correct"] += 1
                elif new_status == "incorrect":
                    stats["new_incorrect"] += 1
                else:
                    stats["new_unknown"] += 1

                if old_status != "correct" and new_status == "correct":
                    stats["flip_to_correct"] += 1
                if old_status == "correct" and new_status != "correct":
                    stats["flip_from_correct"] += 1

            sample_obj["rollouts"] = updated_rollouts
            out_f.write(json.dumps(sample_obj, ensure_ascii=False) + "\n")

            if line_number % PROGRESS_EVERY_SAMPLES == 0:
                print(
                    f"[{ts_now()}] {input_path.name}: processed sample lines={line_number} "
                    f"rollouts={stats['rollouts']}"
                )

    os.replace(tmp_path, output_path)
    return stats


def find_jsonl_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob(FILE_GLOB) if p.is_file()], key=lambda p: str(p))


async def main() -> None:
    suppress_known_warnings()

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    files = find_jsonl_files(INPUT_ROOT)
    if not files:
        raise FileNotFoundError(f"No files matching {FILE_GLOB} under {INPUT_ROOT}")

    print(f"[{ts_now()}] Input root: {INPUT_ROOT}")
    print(f"[{ts_now()}] Output root: {OUTPUT_ROOT}")
    print(f"[{ts_now()}] Files to process: {len(files)}")

    total = {
        "files_processed": 0,
        "files_skipped": 0,
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

    started = time.perf_counter()
    for idx, input_path in enumerate(files, start=1):
        rel = input_path.relative_to(INPUT_ROOT)
        output_path = OUTPUT_ROOT / rel

        if SKIP_EXISTING_OUTPUT_FILES and output_path.exists():
            total["files_skipped"] += 1
            print(f"[{ts_now()}] [{idx}/{len(files)}] Skip existing: {rel}")
            continue

        print(f"[{ts_now()}] [{idx}/{len(files)}] Regrading: {rel}")
        stats = await process_file(input_path, output_path)
        total["files_processed"] += 1
        for k in stats:
            total[k] += stats[k]

        print(
            f"[{ts_now()}] Done {rel} | samples={stats['samples']} rollouts={stats['rollouts']} "
            f"delta_correct={stats['new_correct'] - stats['old_correct']}"
        )

    elapsed = time.perf_counter() - started
    print()
    print("=== Sweep Summary ===")
    print(f"files_processed={total['files_processed']}")
    print(f"files_skipped={total['files_skipped']}")
    print(f"samples={total['samples']}")
    print(f"rollouts={total['rollouts']}")
    print(f"old_correct={total['old_correct']} old_incorrect={total['old_incorrect']} old_unknown={total['old_unknown']}")
    print(f"new_correct={total['new_correct']} new_incorrect={total['new_incorrect']} new_unknown={total['new_unknown']}")
    print(f"delta_correct={total['new_correct'] - total['old_correct']}")
    print(f"flip_to_correct={total['flip_to_correct']} flip_from_correct={total['flip_from_correct']}")
    print(f"elapsed_sec={elapsed:.1f}")
    print(f"output_root={OUTPUT_ROOT}")


if __name__ == "__main__":
    # Run with:
    # conda run -n ed python suze_experiments/20260321/regrade_hinted_results_sweep.py
    asyncio.run(main())
