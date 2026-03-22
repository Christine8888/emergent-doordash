from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


INPUT_JSONL = Path(
    os.environ.get(
        "INPUT_JSONL",
        "suze_experiments/20260313/consolidated_jsonl/results__aime.jsonl",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "OUTPUT_ROOT",
        "suze_experiments/20260321/consolidated_hinted_results",
    )
)
SQLITE_DB_PATH = OUTPUT_ROOT / "_rollout_index.sqlite3"
PROGRESS_EVERY = 100_000


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_part(value: Any) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_")


def required_non_null(value: Any, *, field_name: str, line_number: int) -> Any:
    if value is None:
        raise ValueError(f"line {line_number}: missing required field '{field_name}'")
    return value


def required_text(value: Any, *, field_name: str, line_number: int) -> str:
    required_non_null(value, field_name=field_name, line_number=line_number)
    text = str(value).strip()
    if text == "":
        raise ValueError(f"line {line_number}: empty required field '{field_name}'")
    return text


def hint_fraction_file_name(value: Any, *, line_number: int) -> str:
    token = required_text(value, field_name="hint_fraction", line_number=line_number)
    token = token.replace("/", "_").replace("\\", "_")
    return f"hint_fraction_{token}.jsonl"


def aime_family_bucket(row: dict[str, Any], *, line_number: int) -> str | None:
    run_type = row.get("run_type")
    benchmark = row.get("benchmark")
    if run_type != "results" or benchmark != "aime":
        raise ValueError(f"line {line_number}: expected run_type='results' and benchmark='aime', got {run_type} and {benchmark}")

    segments = row.get("path_hint_segments")
    check_values: list[str] = []
    if isinstance(segments, list) and len(segments) > 0:
        check_values.extend([str(seg) for seg in segments])
    else:
        check_values.append(str(row.get("path_hint_level")))

    for value in check_values:
        if value.startswith("solution"):
            return "aime_solution"
        if value.startswith("cot"):
            return "aime_cot"

    raise ValueError(
        f"line {line_number}: unable to classify AIME row into solution/cot family "
        f"(path_hint_segments={row.get('path_hint_segments')}, "
        f"path_hint_level={row.get('path_hint_level')})"
    )


def compact_score_outcomes(scorer_outcomes: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(scorer_outcomes, dict) or len(scorer_outcomes) == 0:
        return None
    out: dict[str, dict[str, Any]] = {}
    for scorer_name, scorer_payload in scorer_outcomes.items():
        payload = scorer_payload if isinstance(scorer_payload, dict) else {}
        out[str(scorer_name)] = {
            "score_raw_value": payload.get("score_raw_value"),
            "score_normalized": payload.get("score_normalized"),
            "is_correct": payload.get("is_correct"),
            "extracted_answer": payload.get("extracted_answer"),
            "extraction_status": payload.get("extraction_status"),
        }
    return out


def rollout_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": row.get("rollout_id"),
        "eval_id": row.get("eval_id"),
        "task_id": row.get("task_id"),
        "eval_path": row.get("eval_path"),
        "source_owner": row.get("source_owner"),
        "run_type": row.get("run_type"),
        "benchmark": row.get("benchmark"),
        "model": row.get("model"),
        "model_path": row.get("model_path"),
        "created": row.get("created"),
        "run_id": row.get("run_id"),
        "hint_fraction": row.get("hint_fraction"),
        "path_hint_level": row.get("path_hint_level"),
        "path_hint_segments": row.get("path_hint_segments"),
        "solver_name": row.get("solver_name"),
        "sample_file": row.get("sample_file"),
        "rollout_ordinal": row.get("rollout_ordinal"),
        "sample_id": row.get("sample_id"),
        "epoch": row.get("epoch"),
        "target": row.get("target"),
        "prompt_text": row.get("prompt_text"),
        "output_text": row.get("output_text"),
        "score_outcomes": compact_score_outcomes(row.get("scorer_outcomes")),
    }


def validate_required_row_fields(row: dict[str, Any], *, line_number: int) -> None:
    required_non_null(row.get("rollout_id"), field_name="rollout_id", line_number=line_number)
    required_non_null(row.get("eval_id"), field_name="eval_id", line_number=line_number)
    required_non_null(row.get("task_id"), field_name="task_id", line_number=line_number)
    required_text(row.get("eval_path"), field_name="eval_path", line_number=line_number)
    required_text(row.get("source_owner"), field_name="source_owner", line_number=line_number)
    required_text(row.get("run_type"), field_name="run_type", line_number=line_number)
    required_text(row.get("benchmark"), field_name="benchmark", line_number=line_number)
    required_text(row.get("created"), field_name="created", line_number=line_number)
    required_text(row.get("run_id"), field_name="run_id", line_number=line_number)
    required_non_null(row.get("hint_fraction"), field_name="hint_fraction", line_number=line_number)
    required_text(row.get("solver_name"), field_name="solver_name", line_number=line_number)
    required_text(row.get("sample_file"), field_name="sample_file", line_number=line_number)
    required_non_null(row.get("rollout_ordinal"), field_name="rollout_ordinal", line_number=line_number)
    required_non_null(row.get("epoch"), field_name="epoch", line_number=line_number)
    required_text(row.get("sample_id"), field_name="sample_id", line_number=line_number)
    required_text(row.get("target"), field_name="target", line_number=line_number)
    required_text(row.get("prompt_text"), field_name="prompt_text", line_number=line_number)

    path_hint_segments = row.get("path_hint_segments")
    if not isinstance(path_hint_segments, list) or len(path_hint_segments) == 0:
        raise ValueError(
            f"line {line_number}: required field 'path_hint_segments' must be a non-empty list"
        )

def warn_if_missing_output_text(row: dict[str, Any], *, line_number: int) -> None:
    output_text = row.get("output_text")
    if output_text is not None and str(output_text).strip() != "":
        return
    print(
        f"[{ts_now()}] WARNING missing output_text "
        f"line={line_number} "
        f"sample_id={row.get('sample_id')} "
        f"epoch={row.get('epoch')} "
        f"rollout_id={row.get('rollout_id')} "
        f"rollout_ordinal={row.get('rollout_ordinal')}",
        flush=True,
    )


def warn_if_missing_scorer_outcomes(row: dict[str, Any], *, line_number: int) -> None:
    scorer_outcomes = row.get("scorer_outcomes")
    missing = not isinstance(scorer_outcomes, dict) or len(scorer_outcomes) == 0
    if not missing:
        return
    print(
        f"[{ts_now()}] WARNING missing scorer_outcomes "
        f"line={line_number} "
        f"sample_id={row.get('sample_id')} "
        f"epoch={row.get('epoch')} "
        f"rollout_id={row.get('rollout_id')} "
        f"rollout_ordinal={row.get('rollout_ordinal')}",
        flush=True,
    )


def model_dir_for_row(row: dict[str, Any], *, line_number: int) -> str:
    model_path = row.get("model_path")
    if model_path is not None and str(model_path).strip() != "":
        model_dir = safe_part(model_path)
    else:
        model_dir = safe_part(required_text(row.get("model"), field_name="model", line_number=line_number))
    if model_dir == "":
        raise ValueError(f"line {line_number}: model/model_path resolved to empty directory name")
    return model_dir


def create_db(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS rollouts")
    conn.execute(
        """
        CREATE TABLE rollouts (
            aime_family TEXT NOT NULL,
            model_dir TEXT NOT NULL,
            hint_fraction_key TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            created TEXT,
            eval_id TEXT,
            rollout_ordinal INTEGER,
            rollout_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX idx_rollouts_group_sample "
        "ON rollouts (aime_family, model_dir, hint_fraction_key, sample_id)"
    )


def ingest_rows(conn: sqlite3.Connection) -> tuple[int, int]:
    total_rows = 0
    kept_rows = 0
    batch: list[tuple[Any, ...]] = []

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total_rows += 1
            stripped = line.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"[{ts_now()}] WARNING line {line_number}: invalid JSON: {exc}", flush=True)
                continue

            aime_family = aime_family_bucket(row, line_number=line_number)

            validate_required_row_fields(row, line_number=line_number)
            warn_if_missing_output_text(row, line_number=line_number)
            warn_if_missing_scorer_outcomes(row, line_number=line_number)

            kept_rows += 1
            model_dir = model_dir_for_row(row, line_number=line_number)
            hint_fraction_key = hint_fraction_file_name(row.get("hint_fraction"), line_number=line_number)
            sample_id = required_text(row.get("sample_id"), field_name="sample_id", line_number=line_number)
            compact_rollout = rollout_record(row)

            batch.append(
                (
                    aime_family,
                    model_dir,
                    hint_fraction_key,
                    sample_id,
                    row.get("created"),
                    row.get("eval_id"),
                    row.get("rollout_ordinal"),
                    json.dumps(compact_rollout, ensure_ascii=False),
                )
            )

            if len(batch) >= 1000:
                conn.executemany(
                    """
                    INSERT INTO rollouts (
                        aime_family,
                        model_dir,
                        hint_fraction_key,
                        sample_id,
                        created,
                        eval_id,
                        rollout_ordinal,
                        rollout_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

            if total_rows % PROGRESS_EVERY == 0:
                print(f"[{ts_now()}] Scanned {total_rows:,} rows; kept {kept_rows:,}", flush=True)

    if batch:
        conn.executemany(
            """
            INSERT INTO rollouts (
                aime_family,
                model_dir,
                hint_fraction_key,
                sample_id,
                created,
                eval_id,
                rollout_ordinal,
                rollout_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()

    return total_rows, kept_rows


def write_outputs(conn: sqlite3.Connection) -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    groups = conn.execute(
        "SELECT DISTINCT aime_family, model_dir, hint_fraction_key "
        "FROM rollouts "
        "ORDER BY aime_family, model_dir, hint_fraction_key"
    ).fetchall()

    files_written = 0
    for aime_family, model_dir, hint_fraction_key in groups:
        model_out_dir = OUTPUT_ROOT / aime_family / model_dir
        model_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_out_dir / hint_fraction_key
        tmp_out_path = out_path.with_name(out_path.name + ".tmp")

        cur = conn.execute(
            """
            SELECT sample_id, rollout_json
            FROM rollouts
            WHERE aime_family = ? AND model_dir = ? AND hint_fraction_key = ?
            ORDER BY sample_id, created, eval_id, rollout_ordinal
            """,
            (aime_family, model_dir, hint_fraction_key),
        )

        with tmp_out_path.open("w", encoding="utf-8") as out_f:
            current_sample_id: str | None = None
            current_rollouts: list[dict[str, Any]] = []

            for sample_id, rollout_json in cur:
                rollout_obj = json.loads(rollout_json)
                if current_sample_id is None:
                    current_sample_id = sample_id
                if sample_id != current_sample_id:
                    out_f.write(
                        json.dumps(
                            {
                                "sample_id": current_sample_id,
                                "num_rollouts": len(current_rollouts),
                                "rollouts": current_rollouts,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    current_sample_id = sample_id
                    current_rollouts = []
                current_rollouts.append(rollout_obj)

            if current_sample_id is not None:
                out_f.write(
                    json.dumps(
                        {
                            "sample_id": current_sample_id,
                            "num_rollouts": len(current_rollouts),
                            "rollouts": current_rollouts,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        os.replace(tmp_out_path, out_path)
        files_written += 1
        print(f"[{ts_now()}] Wrote {out_path}", flush=True)

    return files_written


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if SQLITE_DB_PATH.exists():
        SQLITE_DB_PATH.unlink()

    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        create_db(conn)
        print(f"[{ts_now()}] Reading {INPUT_JSONL}", flush=True)
        total_rows, kept_rows = ingest_rows(conn)
        print(f"[{ts_now()}] Finished ingest: scanned={total_rows:,}, kept={kept_rows:,}", flush=True)

        files_written = write_outputs(conn)
        print(f"[{ts_now()}] Done. Files written: {files_written}", flush=True)
        print(f"[{ts_now()}] SQLite index: {SQLITE_DB_PATH}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
