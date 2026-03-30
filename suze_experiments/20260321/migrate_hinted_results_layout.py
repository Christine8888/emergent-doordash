from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, TextIO


DEFAULT_INPUT_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2")
def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_part(value: Any) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_")


def discover_legacy_files(input_root: Path) -> list[Path]:
    files: list[Path] = []
    if not input_root.exists():
        return files
    for family_dir in sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if "_" not in family_dir.name:
            continue
        for model_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            for path in sorted(model_dir.glob("hint_fraction_*.jsonl"), key=lambda p: p.name):
                if path.is_file():
                    files.append(path)
    return files


def solver_file_name(rollout: dict[str, Any], *, src_path: Path, sample_id: Any) -> str:
    solver_name = rollout.get("solver_name")
    if solver_name is not None and str(solver_name).strip() != "":
        return f"{safe_part(solver_name)}.jsonl"

    path_hint_level = rollout.get("path_hint_level")
    if path_hint_level is not None and str(path_hint_level).strip() != "":
        first_segment = str(path_hint_level).split("/", 1)[0]
        if first_segment.strip() != "":
            return f"{safe_part(first_segment)}.jsonl"

    raise ValueError(
        f"Unable to derive solver/type file for sample_id={sample_id} in {src_path}. "
        "Missing both solver_name and path_hint_level."
    )


def convert_file(src_path: Path, input_root: Path, out_root: Path) -> tuple[int, int, int]:
    rel = src_path.relative_to(input_root)
    family = rel.parts[0]
    model = rel.parts[1]
    hint_fraction_name = rel.parts[2].removesuffix(".jsonl")

    handles: dict[Path, TextIO] = {}
    tmp_to_final: dict[Path, Path] = {}
    rows_read = 0
    sample_rows_written = 0
    rollouts_seen = 0

    try:
        with src_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows_read += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{src_path}:{line_number} invalid JSON: {exc}") from exc

                sample_id = obj.get("sample_id")
                if sample_id is None or str(sample_id).strip() == "":
                    raise ValueError(f"{src_path}:{line_number} missing sample_id")

                rollouts = obj.get("rollouts")
                if not isinstance(rollouts, list):
                    raise ValueError(f"{src_path}:{line_number} rollouts must be a list")

                grouped: dict[str, list[dict[str, Any]]] = {}
                for rollout in rollouts:
                    if not isinstance(rollout, dict):
                        raise ValueError(f"{src_path}:{line_number} rollout is not a dict")
                    rollouts_seen += 1
                    solver_file = solver_file_name(rollout, src_path=src_path, sample_id=sample_id)
                    grouped.setdefault(solver_file, []).append(rollout)

                for solver_file, solver_rollouts in grouped.items():
                    final_path = out_root / family / model / hint_fraction_name / solver_file
                    tmp_path = final_path.with_name(final_path.name + ".tmp")
                    if tmp_path not in handles:
                        tmp_path.parent.mkdir(parents=True, exist_ok=True)
                        if tmp_path.exists():
                            tmp_path.unlink()
                        handles[tmp_path] = tmp_path.open("w", encoding="utf-8")
                        tmp_to_final[tmp_path] = final_path
                    out_row = {
                        "sample_id": sample_id,
                        "num_rollouts": len(solver_rollouts),
                        "rollouts": solver_rollouts,
                    }
                    handles[tmp_path].write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    sample_rows_written += 1
    finally:
        for handle in handles.values():
            handle.close()
        for tmp_path, final_path in tmp_to_final.items():
            os.replace(tmp_path, final_path)

    return rows_read, sample_rows_written, rollouts_seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert legacy hinted result layout:\n"
            "  <root>/<family>/<model>/hint_fraction_<x>.jsonl\n"
            "into new layout:\n"
            "  <root>/<family>/<model>/hint_fraction_<x>/<solver>.jsonl"
        )
    )
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to --input-root when omitted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root if args.output_root is not None else input_root

    legacy_files = discover_legacy_files(input_root)
    if not legacy_files:
        print(f"[{ts_now()}] No legacy files found under {input_root}")
        return

    print(f"[{ts_now()}] Found {len(legacy_files)} legacy files")
    print(f"[{ts_now()}] Input root: {input_root}")
    print(f"[{ts_now()}] Output root: {output_root}")

    total_rows_read = 0
    total_sample_rows_written = 0
    total_rollouts_seen = 0
    started = time.time()

    for i, src_path in enumerate(legacy_files, start=1):
        print(f"[{ts_now()}] [{i}/{len(legacy_files)}] Converting {src_path}")
        rows_read, sample_rows_written, rollouts_seen = convert_file(
            src_path, input_root, output_root
        )
        total_rows_read += rows_read
        total_sample_rows_written += sample_rows_written
        total_rollouts_seen += rollouts_seen
        print(
            f"[{ts_now()}] [{i}/{len(legacy_files)}] done rows_read={rows_read:,} "
            f"sample_rows_written={sample_rows_written:,} rollouts_seen={rollouts_seen:,}"
        )

    elapsed = time.time() - started
    print(f"[{ts_now()}] Migration complete")
    print(f"  legacy_files={len(legacy_files):,}")
    print(f"  rows_read={total_rows_read:,}")
    print(f"  sample_rows_written={total_sample_rows_written:,}")
    print(f"  rollouts_seen={total_rollouts_seen:,}")
    print(f"  elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
