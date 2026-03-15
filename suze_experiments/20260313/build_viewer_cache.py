from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "consolidated_jsonl"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "_viewer_cache.duckdb"


@dataclass(frozen=True)
class SourceFileStat:
    path: Path
    size_bytes: int
    mtime_ns: int


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_size(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == "TB":
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} TB"


def initialize_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_files (
            source_file TEXT PRIMARY KEY,
            size_bytes BIGINT NOT NULL,
            mtime_ns BIGINT NOT NULL,
            rows_ingested BIGINT NOT NULL,
            scorers_ingested BIGINT NOT NULL,
            last_ingested_ts TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rollouts (
            rollout_id TEXT,
            eval_id TEXT,
            source_file TEXT NOT NULL,
            source_owner TEXT,
            run_type TEXT,
            benchmark TEXT,
            group_key TEXT,
            task_name TEXT,
            model TEXT,
            model_path TEXT,
            solver_name TEXT,
            hint_fraction DOUBLE,
            path_hint_level TEXT,
            sample_id TEXT,
            sample_idx BIGINT,
            epoch BIGINT,
            target TEXT,
            prompt_text TEXT,
            output_text TEXT,
            eval_path TEXT,
            eval_rel_path TEXT,
            created TEXT,
            num_scorers_logged BIGINT,
            questions_scored_for_rollout BIGINT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rollout_scorers (
            source_file TEXT NOT NULL,
            rollout_id TEXT NOT NULL,
            scorer_name TEXT NOT NULL,
            score_normalized TEXT,
            is_correct BOOLEAN,
            extracted_answer TEXT,
            extraction_status TEXT,
            explanation TEXT,
            metadata_json TEXT
        );
        """
    )


def create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rollouts_filters ON rollouts(run_type, benchmark, path_hint_level, model);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rollouts_sample ON rollouts(sample_id, epoch);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rollout_scorers_filter ON rollout_scorers(scorer_name, score_normalized);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rollout_scorers_rollout ON rollout_scorers(rollout_id);")


def list_group_jsonl_files(data_dir: Path, filenames: list[str] | None) -> list[SourceFileStat]:
    all_files: list[SourceFileStat] = []
    for p in sorted(data_dir.glob("*.jsonl")):
        if p.name.startswith("counts_by_"):
            continue
        if p.name.startswith("_"):
            continue
        if "__" not in p.stem:
            continue
        st_obj = p.stat()
        all_files.append(SourceFileStat(path=p, size_bytes=st_obj.st_size, mtime_ns=st_obj.st_mtime_ns))

    if filenames:
        keep = set(filenames)
        all_files = [f for f in all_files if f.path.name in keep]
    return all_files


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_scorer_sidecar_row(row: dict[str, Any]) -> bool:
    # Sidecar rows contain one scorer payload at top-level (no scorer_outcomes map).
    if not isinstance(row, dict):
        return False
    return (
        row.get("rollout_id") is not None
        and row.get("scorer_name") is not None
        and row.get("scorer_outcomes") is None
    )


def _flush_batch(
    conn: duckdb.DuckDBPyConnection,
    rollout_rows: list[dict[str, Any]],
    scorer_rows: list[dict[str, Any]],
) -> tuple[int, int]:
    nr = len(rollout_rows)
    ns = len(scorer_rows)
    if nr > 0:
        df_roll = pd.DataFrame(rollout_rows)
        conn.register("roll_batch", df_roll)
        conn.execute("INSERT INTO rollouts SELECT * FROM roll_batch")
        conn.unregister("roll_batch")
    if ns > 0:
        # Sidecar rescoring files can contain duplicate scorer rows for the same
        # (rollout_id, scorer_name) due to resume/retry workflows. Deduplicate
        # within each batch before MERGE so PK-constrained DBs do not fail.
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in scorer_rows:
            rid = str(row.get("rollout_id") or "")
            scorer = str(row.get("scorer_name") or "")
            if not rid or not scorer:
                continue
            deduped[(rid, scorer)] = row

        df_score = pd.DataFrame(deduped.values())
        if df_score.empty:
            return nr, ns
        conn.register("score_batch", df_score)
        conn.execute(
            """
            MERGE INTO rollout_scorers AS t
            USING score_batch AS s
            ON t.rollout_id = s.rollout_id AND t.scorer_name = s.scorer_name
            WHEN MATCHED THEN UPDATE SET
                source_file = s.source_file,
                score_normalized = s.score_normalized,
                is_correct = s.is_correct,
                extracted_answer = s.extracted_answer,
                extraction_status = s.extraction_status,
                explanation = s.explanation,
                metadata_json = s.metadata_json
            WHEN NOT MATCHED THEN INSERT
                (source_file, rollout_id, scorer_name, score_normalized, is_correct, extracted_answer, extraction_status, explanation, metadata_json)
            VALUES
                (s.source_file, s.rollout_id, s.scorer_name, s.score_normalized, s.is_correct, s.extracted_answer, s.extraction_status, s.explanation, s.metadata_json)
            """
        )
        conn.unregister("score_batch")
    return nr, ns


def ingest_file(
    conn: duckdb.DuckDBPyConnection,
    src: SourceFileStat,
    *,
    include_full_text: bool,
    include_explanations: bool,
    batch_size: int,
    log_every_sec: float,
) -> tuple[int, int]:
    source_file = str(src.path)
    conn.execute("DELETE FROM rollout_scorers WHERE source_file = ?", [source_file])
    conn.execute("DELETE FROM rollouts WHERE source_file = ?", [source_file])
    conn.execute("DELETE FROM source_files WHERE source_file = ?", [source_file])
    conn.commit()

    rollout_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []

    rows_total = 0
    scorers_total = 0
    bad_json_lines = 0
    bytes_seen = 0
    line_count = 0
    started = time.time()
    last_log = started

    with src.path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line_count += 1
            bytes_seen += len(raw_line.encode("utf-8", errors="ignore"))
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json_lines += 1
                if bad_json_lines <= 5 or bad_json_lines % 1000 == 0:
                    preview = line[:120].replace("\n", "\\n")
                    print(
                        f"[{ts_now()}] WARN {src.path.name} bad_json line={line_count:,} "
                        f"count={bad_json_lines:,} preview={preview!r}",
                        flush=True,
                    )
                continue
            if _is_scorer_sidecar_row(row):
                rollout_id = _to_text(row.get("rollout_id"))
                scorer_name = _to_text(row.get("scorer_name"))
                if not rollout_id or not scorer_name:
                    continue
                scorer_rows.append(
                    {
                        "source_file": source_file,
                        "rollout_id": rollout_id,
                        "scorer_name": scorer_name,
                        "score_normalized": _to_text(row.get("score_normalized")),
                        "is_correct": row.get("is_correct") if isinstance(row.get("is_correct"), bool) else None,
                        "extracted_answer": _to_text(row.get("extracted_answer")),
                        "extraction_status": _to_text(row.get("extraction_status")),
                        "explanation": _to_text(row.get("explanation")) if include_explanations else None,
                        "metadata_json": json.dumps(row.get("metadata_json"), ensure_ascii=False, default=str),
                    }
                )
                if len(scorer_rows) >= batch_size:
                    nr, ns = _flush_batch(conn, rollout_rows, scorer_rows)
                    rows_total += nr
                    scorers_total += ns
                    rollout_rows.clear()
                    scorer_rows.clear()
                now = time.time()
                if now - last_log >= log_every_sec:
                    elapsed = now - started
                    pct = 100.0 * bytes_seen / max(1, src.size_bytes)
                    speed = bytes_seen / max(1e-9, elapsed)
                    print(
                        f"[{ts_now()}] {src.path.name} progress={pct:.2f}% "
                        f"rows={rows_total:,} scorers={scorers_total:,} "
                        f"read={human_size(bytes_seen)} speed={human_size(speed)}/s",
                        flush=True,
                    )
                    last_log = now
                continue

            rollout_id = _to_text(row.get("rollout_id"))
            if not rollout_id:
                continue

            rollout_rows.append(
                {
                    "rollout_id": rollout_id,
                    "eval_id": _to_text(row.get("eval_id")),
                    "source_file": source_file,
                    "source_owner": _to_text(row.get("source_owner")),
                    "run_type": _to_text(row.get("run_type")),
                    "benchmark": _to_text(row.get("benchmark")),
                    "group_key": _to_text(row.get("group_key")),
                    "task_name": _to_text(row.get("task_name")),
                    "model": _to_text(row.get("model")),
                    "model_path": _to_text(row.get("model_path")),
                    "solver_name": _to_text(row.get("solver_name")),
                    "hint_fraction": _to_float(row.get("hint_fraction")),
                    "path_hint_level": _to_text(row.get("path_hint_level")),
                    "sample_id": _to_text(row.get("sample_id")),
                    "sample_idx": _to_int(row.get("sample_idx")),
                    "epoch": _to_int(row.get("epoch")),
                    "target": _to_text(row.get("target")),
                    "prompt_text": _to_text(row.get("prompt_text")) if include_full_text else None,
                    "output_text": _to_text(row.get("output_text")) if include_full_text else None,
                    "eval_path": _to_text(row.get("eval_path")),
                    "eval_rel_path": _to_text(row.get("eval_rel_path")),
                    "created": _to_text(row.get("created")),
                    "num_scorers_logged": _to_int(row.get("num_scorers_logged")),
                    "questions_scored_for_rollout": _to_int(row.get("questions_scored_for_rollout")),
                }
            )

            scorer_outcomes = row.get("scorer_outcomes")
            if isinstance(scorer_outcomes, dict):
                for scorer_name, payload in scorer_outcomes.items():
                    p = payload if isinstance(payload, dict) else {}
                    scorer_rows.append(
                        {
                            "source_file": source_file,
                            "rollout_id": rollout_id,
                            "scorer_name": str(scorer_name),
                            "score_normalized": _to_text(p.get("score_normalized")),
                            "is_correct": p.get("is_correct") if isinstance(p.get("is_correct"), bool) else None,
                            "extracted_answer": _to_text(p.get("extracted_answer")),
                            "extraction_status": _to_text(p.get("extraction_status")),
                            "explanation": _to_text(p.get("explanation")) if include_explanations else None,
                            "metadata_json": json.dumps(p.get("metadata_json"), ensure_ascii=False, default=str),
                        }
                    )

            if len(rollout_rows) >= batch_size or len(scorer_rows) >= batch_size:
                nr, ns = _flush_batch(conn, rollout_rows, scorer_rows)
                rows_total += nr
                scorers_total += ns
                rollout_rows.clear()
                scorer_rows.clear()

            now = time.time()
            if now - last_log >= log_every_sec:
                elapsed = now - started
                pct = 100.0 * bytes_seen / max(1, src.size_bytes)
                speed = bytes_seen / max(1e-9, elapsed)
                print(
                    f"[{ts_now()}] {src.path.name} progress={pct:.2f}% "
                    f"rows={rows_total:,} scorers={scorers_total:,} "
                    f"read={human_size(bytes_seen)} speed={human_size(speed)}/s",
                    flush=True,
                )
                last_log = now

    if rollout_rows or scorer_rows:
        nr, ns = _flush_batch(conn, rollout_rows, scorer_rows)
        rows_total += nr
        scorers_total += ns

    conn.execute(
        """
        INSERT INTO source_files (
            source_file, size_bytes, mtime_ns, rows_ingested, scorers_ingested, last_ingested_ts
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [source_file, src.size_bytes, src.mtime_ns, rows_total, scorers_total, ts_now()],
    )
    conn.commit()
    elapsed = time.time() - started
    print(
        f"[{ts_now()}] DONE {src.path.name} rows={rows_total:,} scorers={scorers_total:,} "
        f"bad_json={bad_json_lines:,} elapsed={elapsed/60:.1f}m "
        f"avg_speed={human_size(src.size_bytes/max(1e-9, elapsed))}/s",
        flush=True,
    )
    return rows_total, scorers_total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DuckDB cache for consolidated eval viewer.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with consolidated *.jsonl files.")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="DuckDB output path.")
    p.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional specific filenames to ingest, e.g. results__aime.jsonl",
    )
    p.add_argument("--rebuild", action="store_true", help="Drop/rebuild cache tables before ingest.")
    p.add_argument(
        "--include-full-text",
        action="store_true",
        help="Store prompt/output text in DB (much slower and larger).",
    )
    p.add_argument(
        "--include-explanations",
        action="store_true",
        help="Store scorer explanations in DB (slower/larger).",
    )
    p.add_argument("--batch-size", type=int, default=20000, help="Rows per DB batch flush.")
    p.add_argument("--log-every-sec", type=float, default=5.0, help="Progress log cadence.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    db_path = args.db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    files = list_group_jsonl_files(data_dir, args.files)
    if not files:
        print(f"[{ts_now()}] No matching JSONL files found in {data_dir}", flush=True)
        return

    total_bytes = sum(f.size_bytes for f in files)
    print(
        f"[{ts_now()}] building cache db={db_path} files={len(files)} total_size={human_size(total_bytes)}",
        flush=True,
    )
    for f in files:
        print(f"  - {f.path.name} ({human_size(f.size_bytes)})", flush=True)

    conn = duckdb.connect(str(db_path))
    conn.execute("PRAGMA threads=8;")
    conn.execute("PRAGMA enable_progress_bar=false;")

    initialize_schema(conn)

    if args.rebuild:
        print(f"[{ts_now()}] rebuild requested: clearing cache tables", flush=True)
        conn.execute("DELETE FROM rollout_scorers")
        conn.execute("DELETE FROM rollouts")
        conn.execute("DELETE FROM source_files")
        conn.commit()

    total_rows = 0
    total_scorers = 0
    started = time.time()
    for i, src in enumerate(files, start=1):
        print(
            f"[{ts_now()}] [{i}/{len(files)}] START {src.path.name} size={human_size(src.size_bytes)}",
            flush=True,
        )
        rows, scorers = ingest_file(
            conn,
            src,
            include_full_text=args.include_full_text,
            include_explanations=args.include_explanations,
            batch_size=args.batch_size,
            log_every_sec=args.log_every_sec,
        )
        total_rows += rows
        total_scorers += scorers

    print(f"[{ts_now()}] creating indexes...", flush=True)
    create_indexes(conn)
    conn.commit()

    elapsed = time.time() - started
    print(
        f"[{ts_now()}] COMPLETE files={len(files)} rows={total_rows:,} scorers={total_scorers:,} "
        f"elapsed={elapsed/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
