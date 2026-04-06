from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path

from src.hint_types import HintType
from src.storage import build_hint_generation_path

HF_OWNER = "suzeva"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload one hint JSONL to HF dataset repo (dry-run by default).")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--hint-type", required=True, choices=[h.value for h in HintType])
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    load_env()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in environment/.env")

    main_file = build_hint_generation_path(args.benchmark, args.hint_type, data_root="data")
    if not main_file.exists() or main_file.stat().st_size == 0:
        raise SystemExit(f"Missing or empty file: {main_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_name = f"{args.benchmark}_{args.hint_type}_{timestamp}"
    repo_id = f"{HF_OWNER}/{repo_name}"
    path_in_repo = f"{repo_name}.jsonl"
    print("Planned files:")
    print(f"  - {main_file} -> {repo_id}/{path_in_repo}")

    if not args.apply:
        print("\nDry run only. Pass --apply to execute.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=False)
    api.upload_file(
        path_or_fileobj=str(main_file),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Done.")


if __name__ == "__main__":
    # python -m runs.push_hint_file --benchmark aime2025_2026 --hint-type basic_hint 
    # python -m runs.push_hint_file --benchmark aime2025_2026 --hint-type basic_hint --apply

    # python -m runs.push_hint_file --benchmark aime2025_2026 --hint-type answer_not_revealed 
    # python -m runs.push_hint_file --benchmark aime2025_2026 --hint-type answer_not_revealed --apply

    main()
