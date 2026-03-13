"""Ingest Inspect .eval logs into a benchmark-partitioned corpus."""

from __future__ import annotations

import os
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from inspect_ai.log import read_eval_log

from corpus.ids import make_eval_id, make_grader_result_id, make_rollout_id
from corpus.io import (
    BENCHMARKS_DIRNAME,
    BENCHMARK_INDEX_FILENAME,
    EVAL_MANIFEST_FILENAME,
    LOGGED_GRADER_RESULTS_FILENAME,
    ROLLOUTS_FILENAME,
    RUN_SUMMARY_INGEST_FILENAME,
    benchmark_name_from_eval,
    benchmark_slug,
    extract_scorer_names,
    jsonl_line,
    read_eval_start,
    sample_to_record,
)
from corpus.normalize import extraction_status, normalize_score_value, score_extracted_answer


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class IngestConfig:
    project_root: Path
    eval_roots: dict[str, Path]
    output_dir: Path
    overwrite_outputs: bool = True
    max_eval_files: int | None = None
    progress_every_evals: int = 10


@dataclass
class BenchmarkWriter:
    name: str
    slug: str
    directory: Path
    manifest_path: Path
    rollouts_path: Path
    logged_grader_results_path: Path
    manifest_f: TextIO
    rollouts_f: TextIO
    logged_grader_f: TextIO
    aliases: set[str] = field(default_factory=set)
    eval_files_seen: int = 0
    eval_files_failed: int = 0
    rollouts_written: int = 0
    grader_results_logged_written: int = 0

    def close(self) -> None:
        self.manifest_f.close()
        self.rollouts_f.close()
        self.logged_grader_f.close()


@dataclass(frozen=True)
class ProcessedEvalRecord:
    eval_id: str
    mtime_ns: int | None
    file_size: int | None
    parse_status: str | None


def discover_eval_files(config: IngestConfig) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    owner_totals: dict[str, int] = {}
    for owner, root in config.eval_roots.items():
        if not root.exists():
            print(f"[{ts_now()}] skip missing root: owner={owner} root={root}", flush=True)
            owner_totals[owner] = 0
            continue
        print(f"[{ts_now()}] scanning root: owner={owner} root={root}", flush=True)
        owner_count = 0
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".eval"):
                    eval_path = Path(dirpath) / filename
                    found.append((owner, eval_path))
                    owner_count += 1
                    if owner_count % 5000 == 0:
                        print(
                            f"[{ts_now()}] discovered {owner_count} eval files for owner={owner}",
                            flush=True,
                        )
        print(f"[{ts_now()}] finished scan: owner={owner} eval_files={owner_count}", flush=True)
        owner_totals[owner] = owner_count

    print(f"[{ts_now()}] discovery summary:", flush=True)
    for owner in sorted(owner_totals.keys()):
        print(f"  {owner}: {owner_totals[owner]}", flush=True)
    print(f"  total: {sum(owner_totals.values())}", flush=True)

    found.sort(key=lambda x: (x[0], str(x[1])))
    if config.max_eval_files is not None:
        print(
            f"[{ts_now()}] applying max_eval_files={config.max_eval_files} -> "
            f"processing {min(len(found), config.max_eval_files)}",
            flush=True,
        )
        found = found[: config.max_eval_files]
    return found


def reset_outputs(config: IngestConfig) -> tuple[Path, Path, Path]:
    output_dir = config.output_dir
    benchmarks_dir = output_dir / BENCHMARKS_DIRNAME
    benchmark_index_json = output_dir / BENCHMARK_INDEX_FILENAME
    run_summary_json = output_dir / RUN_SUMMARY_INGEST_FILENAME

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.overwrite_outputs:
        if benchmarks_dir.exists():
            shutil.rmtree(benchmarks_dir)
        for path in [benchmark_index_json, run_summary_json]:
            if path.exists():
                path.unlink()
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    return benchmarks_dir, benchmark_index_json, run_summary_json


def load_processed_eval_index(benchmarks_dir: Path) -> dict[str, ProcessedEvalRecord]:
    """Load previously-ingested eval metadata from benchmark manifests.

    Index is keyed by eval_id and stores the last-seen manifest row for that eval.
    """
    processed: dict[str, ProcessedEvalRecord] = {}
    if not benchmarks_dir.exists():
        return processed

    for benchmark_dir in sorted(p for p in benchmarks_dir.iterdir() if p.is_dir()):
        manifest_path = benchmark_dir / EVAL_MANIFEST_FILENAME
        if not manifest_path.exists():
            continue

        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                eval_id = row.get("eval_id")
                if not isinstance(eval_id, str):
                    continue

                mtime_ns = row.get("mtime_ns")
                file_size = row.get("file_size")
                parse_status = row.get("parse_status")
                processed[eval_id] = ProcessedEvalRecord(
                    eval_id=eval_id,
                    mtime_ns=mtime_ns if isinstance(mtime_ns, int) else None,
                    file_size=file_size if isinstance(file_size, int) else None,
                    parse_status=parse_status if isinstance(parse_status, str) else None,
                )

    return processed


def load_existing_benchmark_index(benchmark_index_json: Path) -> dict[str, dict[str, Any]]:
    """Load existing benchmark index keyed by benchmark_slug."""
    if not benchmark_index_json.exists():
        return {}
    try:
        payload = json.loads(benchmark_index_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("benchmarks", []):
        slug = row.get("benchmark_slug")
        if isinstance(slug, str) and slug:
            out[slug] = row
    return out


def get_or_create_writer(
    writers: dict[str, BenchmarkWriter],
    benchmarks_dir: Path,
    benchmark_name: str,
) -> BenchmarkWriter:
    slug = benchmark_slug(benchmark_name)
    writer = writers.get(slug)
    if writer is not None:
        if benchmark_name != writer.name:
            writer.aliases.add(benchmark_name)
        return writer

    bench_dir = benchmarks_dir / slug
    bench_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bench_dir / EVAL_MANIFEST_FILENAME
    rollouts_path = bench_dir / ROLLOUTS_FILENAME
    logged_grader_results_path = bench_dir / LOGGED_GRADER_RESULTS_FILENAME

    writer = BenchmarkWriter(
        name=benchmark_name,
        slug=slug,
        directory=bench_dir,
        manifest_path=manifest_path,
        rollouts_path=rollouts_path,
        logged_grader_results_path=logged_grader_results_path,
        manifest_f=manifest_path.open("a", encoding="utf-8"),
        rollouts_f=rollouts_path.open("a", encoding="utf-8"),
        logged_grader_f=logged_grader_results_path.open("a", encoding="utf-8"),
    )
    writers[slug] = writer
    return writer


def choose_primary_scorer(
    sample_scores: dict[str, dict[str, Any]],
    configured_scorers: list[str],
) -> str | None:
    if not sample_scores:
        return None
    for scorer_name in configured_scorers:
        if scorer_name in sample_scores:
            return scorer_name
    return next(iter(sample_scores.keys()))


def eval_rel_path(
    *,
    eval_path: Path,
    owner: str,
    owner_root: Path,
    project_root: Path,
) -> str:
    """Return a stable path key even when eval files live outside project_root."""
    try:
        return f"{owner}:{eval_path.relative_to(owner_root)}"
    except ValueError:
        pass

    try:
        return str(eval_path.relative_to(project_root))
    except ValueError:
        # External path fallback (absolute path string keeps it stable).
        return str(eval_path)


def ingest_eval_corpus(config: IngestConfig) -> dict[str, Any]:
    benchmarks_dir, benchmark_index_json, run_summary_json = reset_outputs(config)
    processed_eval_index = (
        {} if config.overwrite_outputs else load_processed_eval_index(benchmarks_dir)
    )
    existing_benchmark_index = (
        {} if config.overwrite_outputs else load_existing_benchmark_index(benchmark_index_json)
    )
    if not config.overwrite_outputs:
        print(
            f"[{ts_now()}] resume mode: loaded {len(processed_eval_index)} prior eval records",
            flush=True,
        )

    eval_files = discover_eval_files(config)
    print(f"[{ts_now()}] total eval files discovered: {len(eval_files)}", flush=True)

    started = time.time()
    counts = {
        "eval_files_discovered": len(eval_files),
        "eval_files_processed": 0,
        "eval_files_failed": 0,
        "eval_files_skipped_unchanged": 0,
        "rollouts_written": 0,
        "grader_results_logged_written": 0,
    }

    writers: dict[str, BenchmarkWriter] = {}
    try:
        for idx, (owner, eval_path) in enumerate(eval_files, start=1):
            owner_root = config.eval_roots[owner]
            rel_path = eval_rel_path(
                eval_path=eval_path,
                owner=owner,
                owner_root=owner_root,
                project_root=config.project_root,
            )
            eval_id = make_eval_id(source_owner=owner, eval_rel_path=rel_path)
            mtime_ns = eval_path.stat().st_mtime_ns
            file_size = eval_path.stat().st_size

            if not config.overwrite_outputs:
                prior = processed_eval_index.get(eval_id)
                if (
                    prior is not None
                    and prior.parse_status == "ok"
                    and prior.mtime_ns == mtime_ns
                    and prior.file_size == file_size
                ):
                    counts["eval_files_skipped_unchanged"] += 1
                    if idx % config.progress_every_evals == 0:
                        elapsed = time.time() - started
                        print(
                            f"[{ts_now()}] [{idx}/{len(eval_files)}] skipped unchanged in {elapsed:.1f}s; "
                            f"processed={counts['eval_files_processed']}, "
                            f"skipped={counts['eval_files_skipped_unchanged']}, "
                            f"failed={counts['eval_files_failed']}",
                            flush=True,
                        )
                    continue

            start_payload = read_eval_start(eval_path)
            eval_obj = start_payload.get("eval", {}) if isinstance(start_payload, dict) else {}
            metadata = eval_obj.get("metadata") if isinstance(eval_obj.get("metadata"), dict) else {}
            hint_fraction = metadata.get("hint_fraction")
            configured_scorers = extract_scorer_names(eval_obj)
            benchmark_name = benchmark_name_from_eval(eval_obj)
            writer = get_or_create_writer(writers, benchmarks_dir, benchmark_name)
            writer.eval_files_seen += 1

            manifest_row: dict[str, Any] = {
                "eval_id": eval_id,
                "source_owner": owner,
                "eval_path": str(eval_path),
                "eval_rel_path": rel_path,
                "mtime_ns": mtime_ns,
                "file_size": file_size,
                "benchmark_name": benchmark_name,
                "benchmark_slug": writer.slug,
                "task_name": eval_obj.get("task"),
                "task_id": eval_obj.get("task_id"),
                "task_display_name": eval_obj.get("task_display_name"),
                "model": eval_obj.get("model"),
                "run_id": eval_obj.get("run_id"),
                "created": eval_obj.get("created"),
                "hint_fraction": hint_fraction,
                "solver_name": metadata.get("solver_name"),
                "metadata_json": metadata,
                "dataset_json": eval_obj.get("dataset"),
                "configured_scorers": configured_scorers,
                "parse_status": "ok",
                "num_samples": None,
                "eval_status": None,
                "error": None,
            }

            try:
                log = read_eval_log(str(eval_path))
                records = [sample_to_record(sample) for sample in log.samples]
                manifest_row["num_samples"] = len(records)
                manifest_row["eval_status"] = getattr(log, "status", None)
            except Exception as exc:  # noqa: BLE001
                manifest_row["parse_status"] = "error"
                manifest_row["error"] = f"{type(exc).__name__}: {exc}"
                writer.manifest_f.write(jsonl_line(manifest_row))
                writer.eval_files_failed += 1
                counts["eval_files_failed"] += 1
                print(
                    f"[{ts_now()}] [{idx}/{len(eval_files)}] parse error: {rel_path} -> "
                    f"{manifest_row['error']}",
                    flush=True,
                )
                processed_eval_index[eval_id] = ProcessedEvalRecord(
                    eval_id=eval_id,
                    mtime_ns=mtime_ns,
                    file_size=file_size,
                    parse_status="error",
                )
                continue

            writer.manifest_f.write(jsonl_line(manifest_row))
            counts["eval_files_processed"] += 1
            processed_eval_index[eval_id] = ProcessedEvalRecord(
                eval_id=eval_id,
                mtime_ns=mtime_ns,
                file_size=file_size,
                parse_status="ok",
            )

            for rollout_ordinal, record in enumerate(records):
                sample_id = record.get("id")
                epoch = record.get("epoch")
                sample_idx = record.get("sample_idx")
                rollout_id = make_rollout_id(
                    eval_id=eval_id,
                    rollout_ordinal=rollout_ordinal,
                    sample_id=sample_id,
                    epoch=epoch,
                    sample_idx=sample_idx,
                )

                output_text = record.get("output")
                target = str(record.get("target") or "")
                sample_scores = (
                    record.get("scores") if isinstance(record.get("scores"), dict) else {}
                )
                primary_scorer = choose_primary_scorer(sample_scores, configured_scorers)

                primary_raw_value = None
                primary_normalized = "U"
                primary_extracted = None
                primary_extraction = "failed"
                if primary_scorer and primary_scorer in sample_scores:
                    primary_score = sample_scores[primary_scorer]
                    primary_raw_value = primary_score.get("value")
                    primary_normalized = normalize_score_value(primary_raw_value)
                    primary_extracted = score_extracted_answer(primary_score)
                    primary_extraction = extraction_status(primary_extracted)

                rollout_row = {
                    "rollout_id": rollout_id,
                    "eval_id": eval_id,
                    "source_owner": owner,
                    "eval_path": str(eval_path),
                    "eval_rel_path": rel_path,
                    "benchmark_name": benchmark_name,
                    "benchmark_slug": writer.slug,
                    "task_name": manifest_row["task_name"],
                    "model": manifest_row["model"],
                    "hint_fraction": hint_fraction,
                    "solver_name": manifest_row["solver_name"],
                    "sample_id": sample_id,
                    "epoch": epoch,
                    "sample_idx": sample_idx,
                    "target": target,
                    "prompt_text": record.get("prompt"),
                    "output_text": output_text,
                    "configured_scorers": configured_scorers,
                    "available_scorers": sorted(sample_scores.keys()) if sample_scores else [],
                    "primary_scorer": primary_scorer,
                    "primary_score_raw_value": primary_raw_value,
                    "primary_score_normalized": primary_normalized,
                    "primary_extracted_answer": primary_extracted,
                    "primary_extraction_status": primary_extraction,
                    "scores_json": sample_scores,
                }
                writer.rollouts_f.write(jsonl_line(rollout_row))
                writer.rollouts_written += 1
                counts["rollouts_written"] += 1

                if sample_scores:
                    for scorer_name, scorer_payload in sample_scores.items():
                        raw_value = scorer_payload.get("value")
                        normalized = normalize_score_value(raw_value)
                        extracted_answer = score_extracted_answer(scorer_payload)
                        grader_origin = "logged"
                        grader_name = str(scorer_name)
                        grader_version = "logged_original"
                        grader_row = {
                            "grader_result_id": make_grader_result_id(
                                rollout_id=rollout_id,
                                grader_origin=grader_origin,
                                grader_name=grader_name,
                                grader_version=grader_version,
                            ),
                            "rollout_id": rollout_id,
                            "eval_id": eval_id,
                            "source_owner": owner,
                            "benchmark_name": benchmark_name,
                            "benchmark_slug": writer.slug,
                            "task_name": manifest_row["task_name"],
                            "model": manifest_row["model"],
                            "hint_fraction": hint_fraction,
                            "sample_id": sample_id,
                            "epoch": epoch,
                            "sample_idx": sample_idx,
                            "target": target,
                            "grader_origin": grader_origin,
                            "grader_name": grader_name,
                            "grader_version": grader_version,
                            "score_raw_value": raw_value,
                            "score_normalized": normalized,
                            "extracted_answer": extracted_answer,
                            "extraction_status": extraction_status(extracted_answer),
                            "explanation": scorer_payload.get("explanation"),
                            "metadata_json": scorer_payload.get("metadata"),
                        }
                        writer.logged_grader_f.write(jsonl_line(grader_row))
                        writer.grader_results_logged_written += 1
                        counts["grader_results_logged_written"] += 1
                else:
                    grader_origin = "logged"
                    grader_name = "__none__"
                    grader_version = "logged_original"
                    grader_row = {
                        "grader_result_id": make_grader_result_id(
                            rollout_id=rollout_id,
                            grader_origin=grader_origin,
                            grader_name=grader_name,
                            grader_version=grader_version,
                        ),
                        "rollout_id": rollout_id,
                        "eval_id": eval_id,
                        "source_owner": owner,
                        "benchmark_name": benchmark_name,
                        "benchmark_slug": writer.slug,
                        "task_name": manifest_row["task_name"],
                        "model": manifest_row["model"],
                        "hint_fraction": hint_fraction,
                        "sample_id": sample_id,
                        "epoch": epoch,
                        "sample_idx": sample_idx,
                        "target": target,
                        "grader_origin": grader_origin,
                        "grader_name": grader_name,
                        "grader_version": grader_version,
                        "score_raw_value": None,
                        "score_normalized": "U",
                        "extracted_answer": None,
                        "extraction_status": "failed",
                        "explanation": "No sample-level score object found in eval log.",
                        "metadata_json": None,
                    }
                    writer.logged_grader_f.write(jsonl_line(grader_row))
                    writer.grader_results_logged_written += 1
                    counts["grader_results_logged_written"] += 1

            if idx % config.progress_every_evals == 0:
                elapsed = time.time() - started
                print(
                    f"[{ts_now()}] [{idx}/{len(eval_files)}] processed in {elapsed:.1f}s; "
                    f"rollouts={counts['rollouts_written']}, "
                    f"logged_grader_rows={counts['grader_results_logged_written']}, "
                    f"skipped={counts['eval_files_skipped_unchanged']}, "
                    f"failed={counts['eval_files_failed']}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    elapsed = time.time() - started
    merged_index = dict(existing_benchmark_index)
    for writer in writers.values():
        merged_index[writer.slug] = {
            "benchmark_name": writer.name,
            "benchmark_slug": writer.slug,
            "aliases": sorted(writer.aliases),
            "directory": str(writer.directory),
            "eval_manifest_jsonl": str(writer.manifest_path),
            "rollouts_jsonl": str(writer.rollouts_path),
            "grader_results_logged_jsonl": str(writer.logged_grader_results_path),
            "grader_results_regraded_dir": str(
                writer.directory / "grader_results_regraded"
            ),
            "eval_files_seen": writer.eval_files_seen,
            "eval_files_failed": writer.eval_files_failed,
            "rollouts_written": writer.rollouts_written,
            "grader_results_logged_written": writer.grader_results_logged_written,
        }

    benchmark_index = {
        "benchmarks": [merged_index[slug] for slug in sorted(merged_index.keys())]
    }
    benchmark_index_json.write_text(json.dumps(benchmark_index, indent=2), encoding="utf-8")

    summary = {
        **counts,
        "elapsed_sec": elapsed,
        "output_dir": str(config.output_dir),
        "benchmarks_dir": str(benchmarks_dir),
        "benchmark_index_json": str(benchmark_index_json),
        "benchmarks_count": len(writers),
    }
    run_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
