from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import duckdb
import pandas as pd
import streamlit as st


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "consolidated_jsonl"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "_viewer_cache.duckdb"
BATCH_SIZE = 2000
ROLLOUT_INDEX_SUFFIX = ".rollout_index.sqlite3"


@dataclass(frozen=True)
class SourceFileStat:
    path: Path
    size_bytes: int
    mtime_ns: int


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def rollout_index_path(source_path: Path) -> Path:
    return source_path.with_name(source_path.name + ROLLOUT_INDEX_SUFFIX)


def index_meta(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}
    try:
        with sqlite3.connect(str(index_path)) as con:
            rows = con.execute("SELECT key, value FROM meta").fetchall()
        return {str(k): str(v) for k, v in rows}
    except Exception:
        return {}


def index_is_fresh(source_path: Path, index_path: Path) -> bool:
    if not index_path.exists():
        return False
    try:
        st_obj = source_path.stat()
    except Exception:
        return False
    meta = index_meta(index_path)
    if not meta:
        return False
    return (
        meta.get("size_bytes") == str(int(st_obj.st_size))
        and meta.get("mtime_ns") == str(int(st_obj.st_mtime_ns))
    )


def lookup_rollout_offsets(
    source_path: Path,
    rollout_ids: list[str],
) -> dict[str, int]:
    """Lookup rollout byte offsets from sidecar SQLite index if available/fresh."""
    if not rollout_ids:
        return {}
    index_path = rollout_index_path(source_path)
    if not index_is_fresh(source_path, index_path):
        return {}
    placeholders = ",".join(["?"] * len(rollout_ids))
    query = (
        "SELECT rollout_id, byte_offset FROM offsets "
        f"WHERE rollout_id IN ({placeholders})"
    )
    try:
        with sqlite3.connect(str(index_path)) as con:
            rows = con.execute(query, rollout_ids).fetchall()
        return {str(rid): int(offset) for rid, offset in rows}
    except Exception:
        return {}


def fetch_rows_by_offsets(source_path: Path, offsets: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Fetch JSON rows from source file via byte offsets."""
    if not offsets:
        return {}
    # Read in ascending offset order to keep disk access mostly sequential.
    items = sorted(offsets.items(), key=lambda kv: kv[1])
    out: dict[str, dict[str, Any]] = {}
    with source_path.open("rb") as f:
        for rid, off in items:
            try:
                f.seek(off)
                line = f.readline()
                if not line:
                    continue
                row = json.loads(line.decode("utf-8", errors="replace"))
                out[str(rid)] = row if isinstance(row, dict) else {}
            except Exception:
                continue
    return out


def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    # Use a fresh connection per rerun/session interaction to avoid stale
    # result handles on some duckdb+streamlit combinations.
    conn = duckdb.connect(db_path)
    conn.execute("PRAGMA threads=4;")
    conn.execute("PRAGMA enable_progress_bar=false;")
    initialize_schema(conn)
    return conn


def _query_df(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any] | None = None,
) -> pd.DataFrame:
    cur = conn.cursor()
    return cur.execute(query, params or []).df()


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
            rollout_id TEXT PRIMARY KEY,
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
            metadata_json TEXT,
            PRIMARY KEY (rollout_id, scorer_name)
        );
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rollouts_filters ON rollouts(run_type, benchmark, path_hint_level, model);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rollouts_sample ON rollouts(sample_id, epoch);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rollout_scorers_filter ON rollout_scorers(scorer_name, score_normalized);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rollout_scorers_rollout ON rollout_scorers(rollout_id);")


def list_group_jsonl_files(data_dir: Path) -> list[SourceFileStat]:
    files: list[SourceFileStat] = []
    for p in sorted(data_dir.glob("*.jsonl")):
        if p.name.startswith("counts_by_"):
            continue
        if p.name.startswith("_"):
            continue
        if "__" not in p.stem:
            continue
        st_obj = p.stat()
        files.append(SourceFileStat(path=p, size_bytes=st_obj.st_size, mtime_ns=st_obj.st_mtime_ns))
    return files


def list_group_jsonl_filenames(data_dir: Path) -> list[str]:
    return [f.path.name for f in list_group_jsonl_files(data_dir)]


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
        x = float(value)
        # Treat NaN/inf as missing so downstream SQL filtering doesn't break.
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def extract_problem_text_from_prompt(prompt_text: str) -> str:
    """Best-effort extraction of just the problem statement from a full prompt."""
    text = (prompt_text or "").strip()
    if not text:
        return ""

    # Common format used by math-style prompts in this repo.
    m = re.search(r"(?is)\bPROBLEM:\s*(.*?)\s*\bSOLUTION:\s*$", text)
    if m:
        return m.group(1).strip()

    # Fallback: if "PROBLEM:" exists but "SOLUTION:" is absent/malformed.
    m2 = re.search(r"(?is)\bPROBLEM:\s*(.*)$", text)
    if m2:
        return m2.group(1).strip()

    # Final fallback: return original prompt.
    return text


def _format_hint_level(value: Any) -> str:
    x = _to_float(value)
    if x is None:
        return "NA"
    return f"{x:.2f}"


def _parse_sample_ids(raw: str) -> list[str]:
    """Parse comma/newline separated sample IDs into de-duplicated exact IDs."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _is_scorer_sidecar_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        row.get("rollout_id") is not None
        and row.get("scorer_name") is not None
        and row.get("scorer_outcomes") is None
    )


def _batched_insert(
    conn: duckdb.DuckDBPyConnection,
    rollout_rows: list[tuple[Any, ...]],
    scorer_rows: list[tuple[Any, ...]],
) -> tuple[int, int]:
    if rollout_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO rollouts (
                rollout_id, eval_id, source_file, source_owner, run_type, benchmark, group_key, task_name,
                model, model_path, solver_name, hint_fraction, path_hint_level, sample_id, sample_idx, epoch,
                target, prompt_text, output_text, eval_path, eval_rel_path, created, num_scorers_logged,
                questions_scored_for_rollout
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rollout_rows,
        )
    if scorer_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO rollout_scorers (
                source_file, rollout_id, scorer_name, score_normalized, is_correct, extracted_answer,
                extraction_status, explanation, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            scorer_rows,
        )
    return len(rollout_rows), len(scorer_rows)


def ingest_file(
    conn: duckdb.DuckDBPyConnection,
    src: SourceFileStat,
    *,
    store_full_text: bool,
    progress_callback: Callable[[float, int, int], None] | None = None,
) -> tuple[int, int]:
    source_file = str(src.path)

    conn.execute("DELETE FROM rollout_scorers WHERE source_file = ?", [source_file])
    conn.execute("DELETE FROM rollouts WHERE source_file = ?", [source_file])

    rollout_rows: list[tuple[Any, ...]] = []
    scorer_rows: list[tuple[Any, ...]] = []
    rollout_count = 0
    scorer_count = 0
    line_count = 0
    bytes_seen = 0
    last_report_line_count = 0
    report_every_lines = 20000

    if progress_callback is not None:
        progress_callback(0.0, rollout_count, scorer_count)

    with src.path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line_count += 1
            bytes_seen += len(raw_line.encode("utf-8", errors="ignore"))
            line = raw_line.strip()
            if not line:
                continue

            row = json.loads(line)
            if _is_scorer_sidecar_row(row):
                rollout_id = _to_text(row.get("rollout_id"))
                scorer_name = _to_text(row.get("scorer_name"))
                if not rollout_id or not scorer_name:
                    continue
                scorer_rows.append(
                    (
                        source_file,
                        rollout_id,
                        scorer_name,
                        _to_text(row.get("score_normalized")),
                        row.get("is_correct") if isinstance(row.get("is_correct"), bool) else None,
                        _to_text(row.get("extracted_answer")),
                        _to_text(row.get("extraction_status")),
                        _to_text(row.get("explanation")) if store_full_text else None,
                        json.dumps(row.get("metadata_json"), ensure_ascii=False, default=str),
                    )
                )

                if len(scorer_rows) >= BATCH_SIZE:
                    a, b = _batched_insert(conn, rollout_rows, scorer_rows)
                    rollout_count += a
                    scorer_count += b
                    rollout_rows.clear()
                    scorer_rows.clear()
                    if progress_callback is not None:
                        file_progress = min(1.0, bytes_seen / max(1, src.size_bytes))
                        progress_callback(file_progress, rollout_count, scorer_count)
                continue

            rollout_id = _to_text(row.get("rollout_id"))
            if not rollout_id:
                continue

            rollout_rows.append(
                (
                    rollout_id,
                    _to_text(row.get("eval_id")),
                    source_file,
                    _to_text(row.get("source_owner")),
                    _to_text(row.get("run_type")),
                    _to_text(row.get("benchmark")),
                    _to_text(row.get("group_key")),
                    _to_text(row.get("task_name")),
                    _to_text(row.get("model")),
                    _to_text(row.get("model_path")),
                    _to_text(row.get("solver_name")),
                    _to_float(row.get("hint_fraction")),
                    _to_text(row.get("path_hint_level")),
                    _to_text(row.get("sample_id")),
                    _to_int(row.get("sample_idx")),
                    _to_int(row.get("epoch")),
                    _to_text(row.get("target")),
                    _to_text(row.get("prompt_text")) if store_full_text else None,
                    _to_text(row.get("output_text")) if store_full_text else None,
                    _to_text(row.get("eval_path")),
                    _to_text(row.get("eval_rel_path")),
                    _to_text(row.get("created")),
                    _to_int(row.get("num_scorers_logged")),
                    _to_int(row.get("questions_scored_for_rollout")),
                )
            )

            scorer_outcomes = row.get("scorer_outcomes")
            if isinstance(scorer_outcomes, dict):
                for scorer_name, payload in scorer_outcomes.items():
                    p = payload if isinstance(payload, dict) else {}
                    scorer_rows.append(
                        (
                            source_file,
                            rollout_id,
                            str(scorer_name),
                            _to_text(p.get("score_normalized")),
                            p.get("is_correct") if isinstance(p.get("is_correct"), bool) else None,
                            _to_text(p.get("extracted_answer")),
                            _to_text(p.get("extraction_status")),
                            _to_text(p.get("explanation")) if store_full_text else None,
                            json.dumps(p.get("metadata_json"), ensure_ascii=False, default=str),
                        )
                    )

            if len(rollout_rows) >= BATCH_SIZE or len(scorer_rows) >= BATCH_SIZE:
                a, b = _batched_insert(conn, rollout_rows, scorer_rows)
                rollout_count += a
                scorer_count += b
                rollout_rows.clear()
                scorer_rows.clear()
                if progress_callback is not None:
                    file_progress = min(1.0, bytes_seen / max(1, src.size_bytes))
                    progress_callback(file_progress, rollout_count, scorer_count)

            if progress_callback is not None and (line_count - last_report_line_count) >= report_every_lines:
                last_report_line_count = line_count
                file_progress = min(1.0, bytes_seen / max(1, src.size_bytes))
                progress_callback(file_progress, rollout_count, scorer_count)

    if rollout_rows or scorer_rows:
        a, b = _batched_insert(conn, rollout_rows, scorer_rows)
        rollout_count += a
        scorer_count += b
        if progress_callback is not None:
            file_progress = min(1.0, bytes_seen / max(1, src.size_bytes))
            progress_callback(file_progress, rollout_count, scorer_count)

    conn.execute(
        """
        INSERT OR REPLACE INTO source_files (
            source_file, size_bytes, mtime_ns, rows_ingested, scorers_ingested, last_ingested_ts
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [source_file, src.size_bytes, src.mtime_ns, rollout_count, scorer_count, ts_now()],
    )
    conn.commit()
    if progress_callback is not None:
        progress_callback(1.0, rollout_count, scorer_count)
    return rollout_count, scorer_count


def sync_database(
    conn: duckdb.DuckDBPyConnection,
    data_dir: Path,
    *,
    force_rebuild: bool = False,
    store_full_text: bool = False,
    selected_filenames: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, int]:
    files = list_group_jsonl_files(data_dir)
    if selected_filenames:
        keep = set(selected_filenames)
        files = [f for f in files if f.path.name in keep]

    if force_rebuild:
        conn.execute("DELETE FROM rollout_scorers")
        conn.execute("DELETE FROM rollouts")
        conn.execute("DELETE FROM source_files")
        conn.commit()

    known = {
        row[0]: (int(row[1]), int(row[2]))
        for row in conn.execute("SELECT source_file, size_bytes, mtime_ns FROM source_files").fetchall()
    }
    current_paths = {str(f.path) for f in files}

    for stale_path in set(known.keys()) - current_paths:
        conn.execute("DELETE FROM rollout_scorers WHERE source_file = ?", [stale_path])
        conn.execute("DELETE FROM rollouts WHERE source_file = ?", [stale_path])
        conn.execute("DELETE FROM source_files WHERE source_file = ?", [stale_path])
    conn.commit()

    summary = {
        "files_total": len(files),
        "files_ingested": 0,
        "files_unchanged": 0,
        "rollouts_ingested": 0,
        "scorers_ingested": 0,
    }

    def emit(frac: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(min(max(frac, 0.0), 1.0), message)

    for idx, src in enumerate(files, start=1):
        key = str(src.path)
        prev = known.get(key)
        if prev is not None and prev == (src.size_bytes, src.mtime_ns):
            summary["files_unchanged"] += 1
            emit(
                idx / max(1, len(files)),
                f"[{idx}/{len(files)}] unchanged {src.path.name} "
                f"(ingested={summary['files_ingested']} unchanged={summary['files_unchanged']})",
            )
            continue
        emit(
            (idx - 1) / max(1, len(files)),
            f"[{idx}/{len(files)}] ingesting {src.path.name} (start)",
        )

        def on_file_progress(file_frac: float, local_rows: int, local_scorers: int) -> None:
            overall = ((idx - 1) + file_frac) / max(1, len(files))
            emit(
                overall,
                f"[{idx}/{len(files)}] ingesting {src.path.name} "
                f"file={file_frac * 100:.1f}% local_rows={local_rows} local_scorers={local_scorers}",
            )

        rows, scorers = ingest_file(
            conn,
            src,
            store_full_text=store_full_text,
            progress_callback=on_file_progress,
        )
        summary["files_ingested"] += 1
        summary["rollouts_ingested"] += rows
        summary["scorers_ingested"] += scorers
        emit(
            idx / max(1, len(files)),
            f"[{idx}/{len(files)}] done {src.path.name} "
            f"(ingested={summary['files_ingested']} unchanged={summary['files_unchanged']} "
            f"rollouts={summary['rollouts_ingested']})",
        )

    return summary


def _where_clause(
    run_types: list[str],
    benchmarks: list[str],
    hint_types: list[str],
    hint_levels: list[float],
    models: list[str],
    sample_ids: list[str],
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if run_types:
        clauses.append(f"r.run_type IN ({','.join(['?'] * len(run_types))})")
        params.extend(run_types)
    if benchmarks:
        clauses.append(f"r.benchmark IN ({','.join(['?'] * len(benchmarks))})")
        params.extend(benchmarks)
    if hint_types:
        clauses.append(f"r.path_hint_level IN ({','.join(['?'] * len(hint_types))})")
        params.extend(hint_types)
    if hint_levels:
        clauses.append(f"r.hint_fraction IN ({','.join(['?'] * len(hint_levels))})")
        params.extend(hint_levels)
    if models:
        clauses.append(f"r.model IN ({','.join(['?'] * len(models))})")
        params.extend(models)
    if sample_ids:
        clauses.append(f"r.sample_id IN ({','.join(['?'] * len(sample_ids))})")
        params.extend(sample_ids)

    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


def load_filter_options(conn: duckdb.DuckDBPyConnection) -> dict[str, list[Any]]:
    run_types = [r[0] for r in conn.execute("SELECT DISTINCT run_type FROM rollouts ORDER BY 1").fetchall() if r[0] is not None]
    benchmarks = [r[0] for r in conn.execute("SELECT DISTINCT benchmark FROM rollouts ORDER BY 1").fetchall() if r[0] is not None]
    hint_types = [r[0] for r in conn.execute("SELECT DISTINCT path_hint_level FROM rollouts ORDER BY 1").fetchall() if r[0] is not None]
    hint_levels = [float(r[0]) for r in conn.execute("SELECT DISTINCT hint_fraction FROM rollouts WHERE hint_fraction IS NOT NULL ORDER BY 1").fetchall()]
    models = [r[0] for r in conn.execute("SELECT DISTINCT model FROM rollouts ORDER BY 1").fetchall() if r[0] is not None]
    scorers = [r[0] for r in conn.execute("SELECT DISTINCT scorer_name FROM rollout_scorers ORDER BY 1").fetchall() if r[0] is not None]
    return {
        "run_types": run_types,
        "benchmarks": benchmarks,
        "hint_types": hint_types,
        "hint_levels": hint_levels,
        "models": models,
        "scorers": scorers,
    }


def query_problem_summary(
    conn: duckdb.DuckDBPyConnection,
    run_types: list[str],
    benchmarks: list[str],
    hint_types: list[str],
    hint_levels: list[float],
    models: list[str],
    sample_ids: list[str],
    scorer_name: str,
    score_labels: list[str],
    limit: int,
) -> pd.DataFrame:
    where_sql, params = _where_clause(run_types, benchmarks, hint_types, hint_levels, models, sample_ids)
    if not score_labels:
        score_labels = ["C", "I", "U"]

    query = f"""
    WITH base AS (
        SELECT
            r.run_type,
            r.benchmark,
            r.model,
            r.path_hint_level AS hint_type,
            r.hint_fraction AS hint_level,
            r.sample_id,
            r.epoch,
            COALESCE(rs.score_normalized, 'U') AS score_label
        FROM rollouts r
        LEFT JOIN rollout_scorers rs
            ON rs.rollout_id = r.rollout_id AND rs.scorer_name = ?
        {where_sql}
    ),
    filtered AS (
        SELECT * FROM base
        WHERE score_label IN ({','.join(['?'] * len(score_labels))})
    )
    SELECT
        run_type,
        benchmark,
        model,
        hint_type,
        hint_level,
        sample_id,
        COUNT(*) AS epochs,
        MIN(epoch) AS min_epoch,
        MAX(epoch) AS max_epoch,
        SUM(CASE WHEN score_label = 'C' THEN 1 ELSE 0 END) AS c_count,
        SUM(CASE WHEN score_label = 'I' THEN 1 ELSE 0 END) AS i_count,
        SUM(CASE WHEN score_label = 'U' THEN 1 ELSE 0 END) AS u_count
    FROM filtered
    GROUP BY run_type, benchmark, model, hint_type, hint_level, sample_id
    ORDER BY benchmark, model, hint_type, hint_level, sample_id
    LIMIT ?
    """

    q_params: list[Any] = [scorer_name]
    q_params.extend(params)
    q_params.extend(score_labels)
    q_params.append(limit)

    return _query_df(conn, query, q_params)


def query_problem_epochs(
    conn: duckdb.DuckDBPyConnection,
    scorer_name: str,
    run_type: str,
    benchmark: str,
    model: str,
    path_hint_level: str,
    hint_fraction: float | None,
    sample_id: str,
) -> pd.DataFrame:
    query = """
    SELECT
        r.epoch,
        r.rollout_id,
        r.eval_id,
        COALESCE(rs.score_normalized, 'U') AS score_label,
        rs.is_correct,
        rs.extracted_answer,
        r.target,
        r.prompt_text,
        r.output_text,
        rs.explanation,
        r.eval_path,
        r.source_file
    FROM rollouts r
    LEFT JOIN rollout_scorers rs
        ON rs.rollout_id = r.rollout_id AND rs.scorer_name = ?
    WHERE r.run_type = ?
      AND r.benchmark = ?
      AND r.model = ?
      AND r.path_hint_level = ?
      AND (
        (? IS NULL AND r.hint_fraction IS NULL)
        OR
        (? IS NOT NULL AND r.hint_fraction IS NOT NULL AND ABS(r.hint_fraction - ?) < 1e-12)
      )
      AND r.sample_id = ?
    ORDER BY r.epoch, r.rollout_id
    """
    return _query_df(
        conn,
        query,
        [scorer_name, run_type, benchmark, model, path_hint_level, hint_fraction, hint_fraction, hint_fraction, sample_id],
    )


def hydrate_epoch_text_from_source(
    df_epochs: pd.DataFrame,
    *,
    scorer_name: str,
    run_type: str,
    benchmark: str,
    model: str,
    path_hint_level: str,
    hint_fraction: float | None,
    sample_id: str,
) -> pd.DataFrame:
    if df_epochs.empty:
        return df_epochs

    missing_prompt = df_epochs["prompt_text"].isna().any() or (df_epochs["prompt_text"] == "").any()
    missing_output = df_epochs["output_text"].isna().any() or (df_epochs["output_text"] == "").any()
    missing_expl = df_epochs["explanation"].isna().any() or (df_epochs["explanation"] == "").any()
    if not (missing_prompt or missing_output or missing_expl):
        return df_epochs

    wanted = set(df_epochs["rollout_id"].astype(str).tolist())
    fills: dict[str, dict[str, Any]] = {}
    source_files = sorted(set(df_epochs["source_file"].astype(str).tolist()))

    def row_matches_scope(row: dict[str, Any], rid: str) -> bool:
        if str(row.get("rollout_id") or "") != rid:
            return False
        if str(row.get("run_type")) != run_type:
            return False
        if str(row.get("benchmark")) != benchmark:
            return False
        if str(row.get("model")) != model:
            return False
        if str(row.get("path_hint_level")) != path_hint_level:
            return False
        row_hint_fraction = _to_float(row.get("hint_fraction"))
        if row_hint_fraction is None and hint_fraction is not None:
            return False
        if row_hint_fraction is not None and hint_fraction is None:
            return False
        if row_hint_fraction is not None and hint_fraction is not None:
            if abs(row_hint_fraction - hint_fraction) > 1e-12:
                return False
        if str(row.get("sample_id")) != sample_id:
            return False
        return True

    def row_fill_payload(row: dict[str, Any], scorer_name: str) -> dict[str, Any]:
        explanation = None
        so = row.get("scorer_outcomes")
        if isinstance(so, dict):
            payload = so.get(scorer_name)
            if isinstance(payload, dict):
                explanation = payload.get("explanation")
        return {
            "prompt_text": row.get("prompt_text"),
            "output_text": row.get("output_text"),
            "target": row.get("target"),
            "explanation": explanation,
        }

    for src in source_files:
        p = Path(src)
        if not p.exists():
            continue
        pending_ids = sorted(wanted - set(fills.keys()))
        if not pending_ids:
            break

        # Fast path: seek directly using prebuilt rollout offset index.
        offsets = lookup_rollout_offsets(p, pending_ids)
        if offsets:
            rows_by_id = fetch_rows_by_offsets(p, offsets)
            for rid, row in rows_by_id.items():
                if not isinstance(row, dict):
                    continue
                if not row_matches_scope(row, rid):
                    continue
                fills[rid] = row_fill_payload(row, scorer_name=scorer_name)
            if len(fills) == len(wanted):
                break
            # Fall through to sequential only for unresolved IDs.
            pending_ids = sorted(wanted - set(fills.keys()))
            if not pending_ids:
                break

        # Fallback when index is missing/stale/incomplete.
        pending_set = set(pending_ids)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = str(row.get("rollout_id") or "")
                if rid not in pending_set:
                    continue
                if not row_matches_scope(row, rid):
                    continue
                fills[rid] = row_fill_payload(row, scorer_name=scorer_name)
                if len(fills) == len(wanted):
                    break
        if len(fills) == len(wanted):
            break

    if not fills:
        return df_epochs

    out = df_epochs.copy()
    for i, r in out.iterrows():
        rid = str(r["rollout_id"])
        fill = fills.get(rid)
        if not fill:
            continue
        if pd.isna(r["prompt_text"]) or str(r["prompt_text"]).strip() == "":
            out.at[i, "prompt_text"] = fill.get("prompt_text")
        if pd.isna(r["output_text"]) or str(r["output_text"]).strip() == "":
            out.at[i, "output_text"] = fill.get("output_text")
        if pd.isna(r["target"]) or str(r["target"]).strip() == "":
            out.at[i, "target"] = fill.get("target")
        if pd.isna(r["explanation"]) or str(r["explanation"]).strip() == "":
            out.at[i, "explanation"] = fill.get("explanation")
    return out


def render_epoch_details(df_epochs: pd.DataFrame) -> None:
    if df_epochs.empty:
        st.info("No rollouts found for this problem under the selected filters.")
        return

    score_counts = df_epochs["score_label"].fillna("U").astype(str).value_counts()
    c_count = int(score_counts.get("C", 0))
    i_count = int(score_counts.get("I", 0))
    u_count = int(score_counts.get("U", 0))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rollouts", int(len(df_epochs)))
    m2.metric("Correct (C)", c_count)
    m3.metric("Incorrect (I)", i_count)
    m4.metric("Unknown (U)", u_count)

    prompt_series = df_epochs["prompt_text"].dropna()
    if not prompt_series.empty and str(prompt_series.iloc[0]).strip():
        first_prompt = str(prompt_series.iloc[0])
        problem_text = extract_problem_text_from_prompt(first_prompt)
        st.markdown("### Problem")
        st.code(problem_text, language="text")
        with st.expander("Full Prompt", expanded=False):
            st.code(first_prompt, language="text")
    else:
        st.warning("Problem text was not found for this selection.")

    summary_cols = ["rollout_id", "epoch", "score_label", "is_correct", "extracted_answer", "target"]
    st.markdown("### Rollouts (All Epochs)")
    st.caption("Each row below is one rollout for the selected problem.")
    st.dataframe(df_epochs[summary_cols], width="stretch", hide_index=True)

    st.markdown("### Rollout Details")
    st.caption("Open any rollout row below to view full prompt/output.")
    for _, row in df_epochs.iterrows():
        label = (
            f"rollout_id={row['rollout_id']} | epoch={row['epoch']} | score={row['score_label']}"
        )
        with st.expander(label):
            st.markdown(f"**Target:** `{row['target']}`")
            if pd.notna(row["extracted_answer"]) and str(row["extracted_answer"]).strip():
                st.markdown(f"**Extracted Answer:** `{row['extracted_answer']}`")
            st.markdown("**Prompt**")
            st.code("" if pd.isna(row["prompt_text"]) else str(row["prompt_text"]), language="text")
            st.markdown("**Output**")
            st.code("" if pd.isna(row["output_text"]) else str(row["output_text"]), language="text")
            if pd.notna(row["explanation"]) and str(row["explanation"]).strip():
                st.markdown("**Scorer Explanation**")
                st.code(str(row["explanation"]), language="text")
            st.caption(f"eval_path: {row['eval_path']}")


def main() -> None:
    st.set_page_config(page_title="Consolidated Eval Viewer", layout="wide")
    st.title("Consolidated Eval Viewer")

    with st.sidebar:
        st.header("Data Source")
        data_dir = Path(
            st.text_input("Consolidated JSONL directory", str(DEFAULT_DATA_DIR))
        ).expanduser()
        db_path = Path(
            st.text_input("DuckDB cache path", str(DEFAULT_DB_PATH))
        ).expanduser()

        sync_clicked = st.button("Sync JSONL -> DuckDB", type="primary")
        rebuild_clicked = st.button("Force Rebuild DB")
        store_full_text = st.checkbox(
            "Store full prompt/output in DB (slow, large)",
            value=False,
            help="Off by default: sync is much faster and text is loaded on-demand for selected problems.",
        )

        available_filenames: list[str] = []
        if data_dir.exists():
            available_filenames = list_group_jsonl_filenames(data_dir)

        selected_filenames = st.multiselect(
            "Sync only these JSONL files",
            options=available_filenames,
            default=[],
            help="Leave empty to sync all files. Example: select only results__aime.jsonl.",
        )

    if not data_dir.exists():
        st.error(f"Data directory does not exist: {data_dir}")
        return

    conn = get_connection(str(db_path))

    if sync_clicked or rebuild_clicked:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def _on_progress(frac: float, message: str) -> None:
            progress_bar.progress(min(max(frac, 0.0), 1.0))
            progress_text.text(message)

        with st.spinner("Syncing JSONL into DuckDB..."):
            summary = sync_database(
                conn,
                data_dir,
                force_rebuild=rebuild_clicked,
                store_full_text=store_full_text,
                selected_filenames=selected_filenames,
                progress_callback=_on_progress,
            )
        progress_bar.progress(1.0)
        progress_text.text("Sync complete.")
        st.success(
            "Sync complete: "
            f"files_ingested={summary['files_ingested']}, "
            f"files_unchanged={summary['files_unchanged']}, "
            f"rollouts_ingested={summary['rollouts_ingested']}, "
            f"scorers_ingested={summary['scorers_ingested']}"
        )

    total_rollouts = conn.execute("SELECT COUNT(*) FROM rollouts").fetchone()[0]
    total_scorer_rows = conn.execute("SELECT COUNT(*) FROM rollout_scorers").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
    st.caption(
        f"DB status: files={total_files:,} rollouts={total_rollouts:,} scorer_rows={total_scorer_rows:,}"
    )

    if total_rollouts == 0:
        st.info("No ingested data yet. Click 'Sync JSONL -> DuckDB' in the sidebar.")
        return

    opts = load_filter_options(conn)
    if not opts["scorers"]:
        st.error("No scorer rows found. Cannot filter by correctness without scorer data.")
        return

    st.subheader("Filters")
    c1, c2, c3, c4, c5, c6 = st.columns([1.0, 1.2, 1.3, 1.1, 1.8, 1.0])
    with c1:
        run_types = st.multiselect(
            "Run Type",
            options=opts["run_types"],
            default=opts["run_types"],
        )
    with c2:
        benchmarks = st.multiselect(
            "Benchmark",
            options=opts["benchmarks"],
            default=opts["benchmarks"],
        )
    with c3:
        hint_types = st.multiselect(
            "Hint Type",
            options=opts["hint_types"],
            default=opts["hint_types"],
        )
    with c4:
        hint_levels = st.multiselect(
            "Hint Level",
            options=opts["hint_levels"],
            default=opts["hint_levels"],
            format_func=_format_hint_level,
        )
    with c5:
        models = st.multiselect(
            "Model",
            options=opts["models"],
            default=[],
            help="Leave empty to include all models.",
        )
    with c6:
        scorer_name = st.selectbox("Scorer", options=opts["scorers"], index=0)

    c7, c8 = st.columns([2.5, 1.0])
    with c7:
        score_labels = st.multiselect(
            "Correctness",
            options=["C", "I", "U"],
            default=["C", "I", "U"],
            help="C=correct, I=incorrect, U=not graded/unknown for selected scorer.",
        )
    with c8:
        row_limit = st.number_input("Problem rows limit", min_value=50, max_value=20000, value=1000, step=50)

    sample_id_text = st.text_input(
        "Sample ID filter (exact)",
        value="",
        help="Optional comma/newline-separated exact sample IDs (e.g. 2009-II-3, 1988-5).",
    )
    sample_ids = _parse_sample_ids(sample_id_text)

    try:
        df_summary = query_problem_summary(
            conn=conn,
            run_types=run_types,
            benchmarks=benchmarks,
            hint_types=hint_types,
            hint_levels=hint_levels,
            models=models,
            sample_ids=sample_ids,
            scorer_name=scorer_name,
            score_labels=score_labels,
            limit=int(row_limit),
        )
    except duckdb.InvalidInputException as exc:
        if "result closed" not in str(exc).lower():
            raise
        conn = get_connection(str(db_path))
        df_summary = query_problem_summary(
            conn=conn,
            run_types=run_types,
            benchmarks=benchmarks,
            hint_types=hint_types,
            hint_levels=hint_levels,
            models=models,
            sample_ids=sample_ids,
            scorer_name=scorer_name,
            score_labels=score_labels,
            limit=int(row_limit),
        )

    st.subheader("Problem Summary")
    st.caption(
        f"Showing {len(df_summary):,} grouped problem rows under current filters (limit={int(row_limit):,})."
    )

    if df_summary.empty:
        return

    df_choices = df_summary.copy()
    df_choices["problem_key"] = (
        df_choices["run_type"].astype(str)
        + " | "
        + df_choices["benchmark"].astype(str)
        + " | "
        + df_choices["model"].astype(str)
        + " | "
        + df_choices["hint_type"].astype(str)
        + " | level="
        + df_choices["hint_level"].apply(_format_hint_level)
        + " | sample_id="
        + df_choices["sample_id"].astype(str)
    )
    problem_keys = df_choices["problem_key"].tolist()

    previous_key = st.session_state.get("summary_problem_key")
    default_idx = 0
    if previous_key and previous_key in set(problem_keys):
        default_idx = problem_keys.index(previous_key)

    selected_idx: int | None = None
    interactive_supported = True
    try:
        event = st.dataframe(
            df_summary,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="problem_summary_table",
        )
        selection = getattr(event, "selection", None)
        if selection is None and isinstance(event, dict):
            selection = event.get("selection")
        if isinstance(selection, dict):
            selected_rows = selection.get("rows", [])
        else:
            selected_rows = getattr(selection, "rows", []) if selection is not None else []
        if isinstance(selected_rows, list) and selected_rows:
            selected_idx = int(selected_rows[0])
    except Exception:
        interactive_supported = False
        st.dataframe(df_summary, width="stretch", hide_index=True)

    if selected_idx is None:
        selected_idx = default_idx

    if interactive_supported:
        st.caption("Click a row in Problem Summary or use the selector below.")
    else:
        st.caption("Row click is not available in this Streamlit build; use the selector below.")

    selected_idx = st.selectbox(
        "Problem",
        options=list(range(len(df_choices))),
        index=int(selected_idx),
        format_func=lambda i: str(df_choices.iloc[int(i)]["problem_key"]),
    )

    if selected_idx < 0 or selected_idx >= len(df_choices):
        selected_idx = 0

    selected_row = df_choices.iloc[selected_idx]
    st.session_state["summary_problem_key"] = str(selected_row["problem_key"])

    st.caption(
        f"Selected: run_type={selected_row['run_type']} benchmark={selected_row['benchmark']} "
        f"model={selected_row['model']} hint_type={selected_row['hint_type']} "
        f"hint_level={_format_hint_level(selected_row['hint_level'])} sample_id={selected_row['sample_id']}"
    )
    st.caption("Rollouts for this selected problem are shown below under 'Rollouts (All Epochs)'.")

    selected_hint_level = _to_float(selected_row["hint_level"])
    try:
        df_epochs = query_problem_epochs(
            conn=conn,
            scorer_name=scorer_name,
            run_type=str(selected_row["run_type"]),
            benchmark=str(selected_row["benchmark"]),
            model=str(selected_row["model"]),
            path_hint_level=str(selected_row["hint_type"]),
            hint_fraction=selected_hint_level,
            sample_id=str(selected_row["sample_id"]),
        )
    except duckdb.InvalidInputException as exc:
        if "result closed" not in str(exc).lower():
            raise
        conn = get_connection(str(db_path))
        df_epochs = query_problem_epochs(
            conn=conn,
            scorer_name=scorer_name,
            run_type=str(selected_row["run_type"]),
            benchmark=str(selected_row["benchmark"]),
            model=str(selected_row["model"]),
            path_hint_level=str(selected_row["hint_type"]),
            hint_fraction=selected_hint_level,
            sample_id=str(selected_row["sample_id"]),
        )
    df_epochs = hydrate_epoch_text_from_source(
        df_epochs,
        scorer_name=scorer_name,
        run_type=str(selected_row["run_type"]),
        benchmark=str(selected_row["benchmark"]),
        model=str(selected_row["model"]),
        path_hint_level=str(selected_row["hint_type"]),
        hint_fraction=selected_hint_level,
        sample_id=str(selected_row["sample_id"]),
    )
    render_epoch_details(df_epochs)


if __name__ == "__main__":
    # streamlit run suze_experiments/20260313/view_consolidated_eval.py
    main()
