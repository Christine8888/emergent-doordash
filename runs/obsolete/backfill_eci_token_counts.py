from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from runs.generate_eci import BENCHMARKS
from src.eci_runner import _extract_token_count
from src.model_config import ALL_MODEL_PATHS
from src.storage import build_eci_score_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill ECI input/output token counts from stored Inspect logs."
    )
    parser.add_argument("--benchmark", type=str, choices=["all"] + BENCHMARKS, default="all")
    parser.add_argument("--model", type=str, choices=["all"] + list(ALL_MODEL_PATHS), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_benchmarks(benchmark: str) -> list[str]:
    if benchmark == "all":
        return list(BENCHMARKS)
    return [benchmark]


def _selected_models(model: str) -> list[str]:
    if model == "all":
        return list(ALL_MODEL_PATHS)
    return [model]


def _load_inspect_index(path: Path, cache: dict[Path, dict[tuple[str, int], tuple[int, int]]]) -> dict[tuple[str, int], tuple[int, int]]:
    cached = cache.get(path)
    if cached is not None:
        return cached

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"Inspect log missing top-level samples list: {path}")

    index: dict[tuple[str, int], tuple[int, int]] = {}
    for sample in raw_samples:
        if not isinstance(sample, dict):
            continue
        sample_id = sample.get("id")
        epoch = sample.get("epoch")
        if not isinstance(sample_id, (str, int)) or not isinstance(epoch, int):
            continue
        input_tokens = _extract_token_count(sample, "input_tokens", "prompt_tokens")
        output_tokens = _extract_token_count(sample, "output_tokens", "completion_tokens")
        index[(str(sample_id), epoch)] = (input_tokens, output_tokens)

    cache[path] = index
    return index


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _backfill_file(
    *,
    benchmark_name: str,
    model: str,
    path: Path,
    dry_run: bool,
    inspect_cache: dict[Path, dict[tuple[str, int], tuple[int, int]]],
) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    updated_rows: list[dict[str, Any]] = []
    changed_count = 0
    missing_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                updated_rows.append(row)
                missing_count += 1
                continue

            inspect_log_path_value = metadata.get("inspect_log_path")
            problem_id = row.get("problem_id")
            epoch = metadata.get("epoch_in_run", row.get("rollout_id"))
            if not isinstance(inspect_log_path_value, str) or not isinstance(problem_id, str) or not isinstance(epoch, int):
                updated_rows.append(row)
                missing_count += 1
                continue

            inspect_log_path = Path(inspect_log_path_value)
            if not inspect_log_path.is_absolute():
                inspect_log_path = inspect_log_path
            if not inspect_log_path.exists():
                updated_rows.append(row)
                missing_count += 1
                continue

            sample_index = _load_inspect_index(inspect_log_path, inspect_cache)
            token_counts = sample_index.get((problem_id, epoch))
            if token_counts is None:
                updated_rows.append(row)
                missing_count += 1
                continue

            input_tokens, output_tokens = token_counts
            if row.get("input_token_count") != input_tokens or row.get("output_token_count") != output_tokens:
                row["input_token_count"] = input_tokens
                row["output_token_count"] = output_tokens
                changed_count += 1
            updated_rows.append(row)

    if changed_count > 0 and not dry_run:
        _rewrite_jsonl(path, updated_rows)

    print(
        f"benchmark={benchmark_name} model={model} changed={changed_count} missing={missing_count} path={path}",
        flush=True,
    )
    return changed_count, missing_count


def main() -> None:
    args = _parse_args()
    benchmark_names = _selected_benchmarks(args.benchmark)
    models = _selected_models(args.model)

    inspect_cache: dict[Path, dict[tuple[str, int], tuple[int, int]]] = {}
    total_changed = 0
    total_missing = 0
    files_touched = 0
    for benchmark_name in benchmark_names:
        for model in models:
            path = build_eci_score_path(
                benchmark_name=benchmark_name,
                model=model,
                data_root=args.data_root,
            )
            changed_count, missing_count = _backfill_file(
                benchmark_name=benchmark_name,
                model=model,
                path=path,
                dry_run=args.dry_run,
                inspect_cache=inspect_cache,
            )
            if changed_count > 0:
                files_touched += 1
            total_changed += changed_count
            total_missing += missing_count

    print(
        f"summary changed={total_changed} missing={total_missing} files_touched={files_touched} dry_run={args.dry_run}",
        flush=True,
    )


if __name__ == "__main__":
    # python -m runs.backfill_eci_token_counts
    main()
