from __future__ import annotations

import json
from pathlib import Path
from typing import Any


JSONL_PATH = Path("suze_experiments/20260313/consolidated_jsonl/results__aime.jsonl")
NUM_OBJECTS_TO_PRINT = 2



def main() -> None:
    if not JSONL_PATH.exists():
        raise SystemExit(f"File not found: {JSONL_PATH}")

    printed = 0
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if printed >= NUM_OBJECTS_TO_PRINT:
                break
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            obj["prompt_text"] = obj.get("prompt_text")
            obj["output_text"] = obj.get("output_text")

            printed += 1
            print(f"\n--- object {printed} ---")
            print(json.dumps(obj, indent=2, ensure_ascii=False))

    if printed == 0:
        print("No JSON objects found.")


if __name__ == "__main__":
    # python suze_experiments/20260321/inspect_consolidated_jsonl.py
    main()
