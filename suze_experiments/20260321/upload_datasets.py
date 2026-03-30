"""Normalize JSONL files (fill missing keys with null) and upload to Hugging Face."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# ==== Edit these constants ====
HF_NAMESPACE = "suzeva"
HF_REPO_TYPE = "dataset"
HF_CLI = "hf"

# Map: local file path -> dataset repo name (e.g. "bbh_hints" or "suzeva/bbh_hints")
FILES_TO_UPLOAD: dict[str, str] = {
    # these are hints for all the questions in a certain dataset. These were generated in suze_experiments/20260209/generate_hint_data.py
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/bbh.jsonl": "bbh_hints_solution",
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/arc_challenge.jsonl": "arc_challenge_hints_solution",
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/math_level_5.jsonl": "math_level_5_hints_solution",
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/mmlu_0_shot.jsonl": "mmlu_0_shot_hints_solution",
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/piqa.jsonl": "piqa_hints_solution",
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/winogrande.jsonl": "winogrande_hints_solution",
    # "/nlp/scr/suzeva/projects/emergent-doordash/suze_experiments/data/solution/hellaswag.jsonl": "hellaswag_hints_solution",

    # now lets upload aime and gpqa
    # "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/data/cot/aime.jsonl": "aime_hints_cot",
    "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/data/solution/aime.jsonl": "aime_hints_solution",
    "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/data/cot/gpqa.jsonl": "gpqa_hints_cot",
    "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/data/solution/gpqa.jsonl": "gpqa_hints_solution",
    


}

NORMALIZED_DIR = Path(__file__).resolve().parent / "normalized_for_upload"
# ==============================

LIST_TOKEN = "[]"


def collect_schema(
    value: Any,
    keys_by_path: dict[tuple[str, ...], list[str]],
    first_non_null_line_by_path: dict[tuple[str, ...], int],
    line_no: int,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, dict):
        keys = keys_by_path.setdefault(path, [])
        for key, child in value.items():
            if key not in keys:
                keys.append(key)
            collect_schema(child, keys_by_path, first_non_null_line_by_path, line_no, path + (key,))
        return

    if isinstance(value, list):
        for child in value:
            collect_schema(child, keys_by_path, first_non_null_line_by_path, line_no, path + (LIST_TOKEN,))
        return

    if value is not None and path not in first_non_null_line_by_path:
        first_non_null_line_by_path[path] = line_no


def normalize_value(
    value: Any,
    keys_by_path: dict[tuple[str, ...], list[str]],
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in keys_by_path.get(path, []):
            child_path = path + (key,)
            if key in value:
                out[key] = normalize_value(value[key], keys_by_path, child_path)
            else:
                out[key] = None
        return out

    if isinstance(value, list):
        return [normalize_value(child, keys_by_path, path + (LIST_TOKEN,)) for child in value]

    return value


def normalize_jsonl(input_path: Path, output_path: Path) -> int:
    keys_by_path: dict[tuple[str, ...], list[str]] = {}
    first_non_null_line_by_path: dict[tuple[str, ...], int] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_no}: {exc}") from exc
            collect_schema(obj, keys_by_path, first_non_null_line_by_path, line_no)

    # Put representative rows first so HF sees non-null examples early without changing values.
    seed_line_numbers = sorted(set(first_non_null_line_by_path.values()))
    seed_line_set = set(seed_line_numbers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    normalized_rows: list[str] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_no}: {exc}") from exc
            normalized_rows.append(json.dumps(normalize_value(obj, keys_by_path), ensure_ascii=False))
            count += 1

    with output_path.open("w", encoding="utf-8") as fout:
        for line_no in seed_line_numbers:
            if 1 <= line_no <= len(normalized_rows):
                fout.write(normalized_rows[line_no - 1] + "\n")

        for i, row in enumerate(normalized_rows, 1):
            if i not in seed_line_set:
                fout.write(row + "\n")

    return count


def upload_file(repo_id: str, local_file: Path, remote_path: str) -> None:
    cmd = [
        HF_CLI,
        "upload",
        repo_id,
        str(local_file),
        remote_path,
        "--repo-type",
        HF_REPO_TYPE,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    for local_path_str, dataset_name in FILES_TO_UPLOAD.items():
        local_path = Path(local_path_str).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Input file not found: {local_path}")
        if not local_path.is_file():
            raise ValueError(f"Input path is not a file: {local_path}")

        repo_id = dataset_name if "/" in dataset_name else f"{HF_NAMESPACE}/{dataset_name}"
        remote_name = local_path.name
        normalized_path = NORMALIZED_DIR / local_path.name
        rows = normalize_jsonl(local_path, normalized_path)
        print(f"[normalize] {local_path} -> {normalized_path} ({rows} rows)")

        print(f"[upload] {normalized_path} -> {repo_id}:{remote_name}")
        upload_file(repo_id, normalized_path, remote_name)

    print("[done] normalization and upload complete")


if __name__ == "__main__":
    # python suze_experiments/20260321/upload_datasets.py
    main()
