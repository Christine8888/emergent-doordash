"""Run post-hoc regraders on corpus rollout tables."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus.ids import make_grader_result_id
from corpus.io import (
    BENCHMARKS_DIRNAME,
    BENCHMARK_INDEX_FILENAME,
    REGRADED_DIRNAME,
    ROLLOUTS_FILENAME,
    RUN_SUMMARY_REGRADE_FILENAME,
    iter_jsonl,
    jsonl_line,
)
from corpus.regraders.registry import resolve_regraders


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class RegradeConfig:
    corpus_dir: Path
    enabled_regraders: list[str]
    overwrite_regraded: bool = True
    progress_every_rollouts: int = 1000
    benchmark_allowlist: list[str] | None = None  # slugs


def benchmark_dirs_from_corpus(corpus_dir: Path) -> list[Path]:
    index_path = corpus_dir / BENCHMARK_INDEX_FILENAME
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        dirs: list[Path] = []
        for row in index.get("benchmarks", []):
            directory = row.get("directory")
            if directory:
                dirs.append(Path(directory))
        return sorted(set(dirs))

    benchmarks_dir = corpus_dir / BENCHMARKS_DIRNAME
    if not benchmarks_dir.exists():
        return []
    return sorted(p for p in benchmarks_dir.iterdir() if p.is_dir())


def should_process_benchmark(benchmark_dir: Path, allowlist: list[str] | None) -> bool:
    if not allowlist:
        return True
    return benchmark_dir.name in set(allowlist)


def targets_benchmark(regrader_name: str, benchmark_slug: str) -> bool:
    # Empty target list means "all benchmarks".
    # Otherwise benchmark must be explicitly listed.
    from corpus.regraders.registry import REGRADER_REGISTRY

    spec = REGRADER_REGISTRY[regrader_name]
    return len(spec.benchmark_slugs) == 0 or benchmark_slug in spec.benchmark_slugs


def regrade_corpus(config: RegradeConfig) -> dict[str, Any]:
    if not config.enabled_regraders:
        raise ValueError("enabled_regraders is empty. Provide at least one regrader name.")
    regraders = resolve_regraders(config.enabled_regraders)

    benchmark_dirs = benchmark_dirs_from_corpus(config.corpus_dir)
    benchmark_dirs = [
        d for d in benchmark_dirs if should_process_benchmark(d, config.benchmark_allowlist)
    ]

    counts = {
        "benchmarks_seen": len(benchmark_dirs),
        "benchmarks_processed": 0,
        "rollouts_seen": 0,
        "rollouts_processed": 0,
        "regrader_rows_written": 0,
    }

    per_benchmark: list[dict[str, Any]] = []
    for bench_idx, benchmark_dir in enumerate(benchmark_dirs, start=1):
        benchmark_slug = benchmark_dir.name
        rollouts_path = benchmark_dir / ROLLOUTS_FILENAME
        if not rollouts_path.exists():
            continue

        active_regraders = [
            regrader for regrader in regraders if targets_benchmark(regrader.name, benchmark_slug)
        ]
        if not active_regraders:
            print(
                f"[{ts_now()}] skip benchmark {benchmark_slug}: "
                f"no enabled regrader targets this benchmark",
                flush=True,
            )
            continue

        regraded_dir = benchmark_dir / REGRADED_DIRNAME
        regraded_dir.mkdir(parents=True, exist_ok=True)

        out_paths = {
            regrader.name: regraded_dir / f"{regrader.version}.jsonl"
            for regrader in active_regraders
        }
        if config.overwrite_regraded:
            for path in out_paths.values():
                if path.exists():
                    path.unlink()

        bench_counts = {
            "benchmark_slug": benchmark_slug,
            "active_regraders": [r.name for r in active_regraders],
            "rollouts_seen": 0,
            "rollouts_processed": 0,
            "regrader_rows_written": 0,
            "output_files": {name: str(path) for name, path in out_paths.items()},
        }

        handles = {name: path.open("a", encoding="utf-8") for name, path in out_paths.items()}
        try:
            for rollout_idx, rollout_row in enumerate(iter_jsonl(rollouts_path), start=1):
                counts["rollouts_seen"] += 1
                bench_counts["rollouts_seen"] += 1

                for regrader in active_regraders:
                    if not regrader.applies(rollout_row):
                        continue
                    try:
                        regraded = asyncio.run(regrader.run(rollout_row))
                    except Exception as exc:  # noqa: BLE001
                        regraded = {
                            "score_raw_value": None,
                            "score_normalized": "U",
                            "extracted_answer": None,
                            "extraction_status": "failed",
                            "explanation": f"Regrader error: {type(exc).__name__}: {exc}",
                            "metadata": {"error": True},
                        }

                    grader_origin = "regraded"
                    grader_name = regrader.name
                    grader_version = regrader.version
                    regraded_row = {
                        "grader_result_id": make_grader_result_id(
                            rollout_id=str(rollout_row.get("rollout_id")),
                            grader_origin=grader_origin,
                            grader_name=grader_name,
                            grader_version=grader_version,
                        ),
                        "rollout_id": rollout_row.get("rollout_id"),
                        "eval_id": rollout_row.get("eval_id"),
                        "source_owner": rollout_row.get("source_owner"),
                        "benchmark_name": rollout_row.get("benchmark_name"),
                        "benchmark_slug": rollout_row.get("benchmark_slug"),
                        "task_name": rollout_row.get("task_name"),
                        "model": rollout_row.get("model"),
                        "hint_fraction": rollout_row.get("hint_fraction"),
                        "sample_id": rollout_row.get("sample_id"),
                        "epoch": rollout_row.get("epoch"),
                        "sample_idx": rollout_row.get("sample_idx"),
                        "target": rollout_row.get("target"),
                        "grader_origin": grader_origin,
                        "grader_name": grader_name,
                        "grader_version": grader_version,
                        "score_raw_value": regraded.get("score_raw_value"),
                        "score_normalized": regraded.get("score_normalized"),
                        "extracted_answer": regraded.get("extracted_answer"),
                        "extraction_status": regraded.get("extraction_status"),
                        "explanation": regraded.get("explanation"),
                        "metadata_json": regraded.get("metadata"),
                    }
                    handles[regrader.name].write(jsonl_line(regraded_row))
                    counts["regrader_rows_written"] += 1
                    bench_counts["regrader_rows_written"] += 1

                counts["rollouts_processed"] += 1
                bench_counts["rollouts_processed"] += 1

                if rollout_idx % config.progress_every_rollouts == 0:
                    print(
                        f"[{ts_now()}] [benchmark {bench_idx}/{len(benchmark_dirs)} {benchmark_dir.name}] "
                        f"rollouts_processed={bench_counts['rollouts_processed']} "
                        f"regrader_rows={bench_counts['regrader_rows_written']}",
                        flush=True,
                    )
        finally:
            for f in handles.values():
                f.close()

        per_benchmark.append(bench_counts)
        counts["benchmarks_processed"] += 1
        print(
            f"[{ts_now()}] finished benchmark {benchmark_dir.name}: "
            f"rollouts={bench_counts['rollouts_processed']} "
            f"regrader_rows={bench_counts['regrader_rows_written']}",
            flush=True,
        )

    summary = {
        **counts,
        "enabled_regraders": config.enabled_regraders,
        "regrader_targets": {
            regrader.name: list(regrader.benchmark_slugs) for regrader in regraders
        },
        "corpus_dir": str(config.corpus_dir),
        "per_benchmark": per_benchmark,
    }
    summary_path = config.corpus_dir / RUN_SUMMARY_REGRADE_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
