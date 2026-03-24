from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import streamlit as st


DEFAULT_DATA_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2")
KNOWN_DATASETS = ["aime", "gpqa"]
KNOWN_FAMILIES = ["solution", "cot"]

SCORER_FILTER_ALL = "(all scorers)"
STATUS_OPTIONS = {
    "Correct": "correct",
    "Incorrect": "incorrect",
    "Unknown": "unknown",
}


def safe_slug(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_")


def hint_sort_key(path: Path) -> tuple[int, float | str]:
    stem = path.stem
    token = stem
    if stem.startswith("hint_fraction_"):
        token = stem[len("hint_fraction_") :]
    try:
        return (0, float(token))
    except ValueError:
        return (1, token)


def sidecar_index_path(data_path: Path) -> Path:
    return data_path.with_name(data_path.name + ".sample_index.sqlite3")


def _parse_status(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "unknown"


def _summary_for_sample_obj(sample_obj: dict[str, Any]) -> tuple[int, str | None, list[str], dict[str, Any]]:
    rollouts = sample_obj.get("rollouts")
    if not isinstance(rollouts, list):
        return 0, None, [], {}

    num_rollouts = len(rollouts)
    target = None
    scorers: set[str] = set()
    status_by_scorer: dict[str, dict[str, bool]] = {}

    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        if target is None and rollout.get("target") is not None:
            target = str(rollout.get("target"))
        score_outcomes = rollout.get("score_outcomes")
        if not isinstance(score_outcomes, dict):
            continue
        for scorer_name, payload in score_outcomes.items():
            scorer = str(scorer_name)
            scorers.add(scorer)
            if scorer not in status_by_scorer:
                status_by_scorer[scorer] = {
                    "correct": False,
                    "incorrect": False,
                    "unknown": False,
                }
            if not isinstance(payload, dict):
                status_by_scorer[scorer]["unknown"] = True
                continue
            status = _parse_status(payload.get("is_correct"))
            status_by_scorer[scorer][status] = True

    return num_rollouts, target, sorted(scorers), status_by_scorer


def rebuild_index(data_path: Path, index_path: Path) -> None:
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with sqlite3.connect(tmp_path) as conn:
        # Prefer conservative settings for network/shared filesystems.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE sample_offsets (
                sample_order INTEGER PRIMARY KEY,
                sample_id TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                num_rollouts INTEGER NOT NULL,
                target TEXT,
                scorers_json TEXT NOT NULL,
                status_by_scorer_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX idx_sample_id ON sample_offsets(sample_id)")

        batch: list[tuple[Any, ...]] = []
        bad_lines = 0
        sample_order = 0
        with data_path.open("rb") as f:
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    sample_obj = json.loads(line.decode("utf-8"))
                except Exception:
                    bad_lines += 1
                    continue

                sample_id = sample_obj.get("sample_id")
                if sample_id is None:
                    bad_lines += 1
                    continue

                num_rollouts, target, scorers, status_by_scorer = _summary_for_sample_obj(sample_obj)
                batch.append(
                    (
                        sample_order,
                        str(sample_id),
                        int(offset),
                        int(num_rollouts),
                        target,
                        json.dumps(scorers, ensure_ascii=False),
                        json.dumps(status_by_scorer, ensure_ascii=False),
                    )
                )
                sample_order += 1

                if len(batch) >= 1000:
                    conn.executemany(
                        """
                        INSERT INTO sample_offsets (
                            sample_order,
                            sample_id,
                            byte_offset,
                            num_rollouts,
                            target,
                            scorers_json,
                            status_by_scorer_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    batch.clear()

        if batch:
            conn.executemany(
                """
                INSERT INTO sample_offsets (
                    sample_order,
                    sample_id,
                    byte_offset,
                    num_rollouts,
                    target,
                    scorers_json,
                    status_by_scorer_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )

        stat = data_path.stat()
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("source_path", str(data_path)),
                ("size_bytes", str(stat.st_size)),
                ("mtime_ns", str(stat.st_mtime_ns)),
                ("bad_lines", str(bad_lines)),
                ("rows_indexed", str(sample_order)),
            ],
        )
        conn.commit()

    os.replace(tmp_path, index_path)


def ensure_index_is_fresh(data_path: Path) -> Path:
    index_path = sidecar_index_path(data_path)
    stat = data_path.stat()
    expected_size = str(stat.st_size)
    expected_mtime = str(stat.st_mtime_ns)

    needs_rebuild = True
    if index_path.exists():
        try:
            with sqlite3.connect(index_path) as conn:
                meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            needs_rebuild = not (
                meta.get("source_path") == str(data_path)
                and meta.get("size_bytes") == expected_size
                and meta.get("mtime_ns") == expected_mtime
            )
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        rebuild_index(data_path, index_path)
    return index_path


def _read_index_rows(index_path: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(index_path) as conn:
        return conn.execute(
            """
            SELECT
                sample_order,
                sample_id,
                byte_offset,
                num_rollouts,
                target,
                scorers_json,
                status_by_scorer_json
            FROM sample_offsets
            ORDER BY sample_order
            """
        ).fetchall()


@st.cache_data(show_spinner=False)
def load_index_rows(data_path_str: str, source_size: int, source_mtime_ns: int) -> tuple[list[dict[str, Any]], str]:
    del source_size, source_mtime_ns
    data_path = Path(data_path_str)
    index_path = ensure_index_is_fresh(data_path)

    rows: list[dict[str, Any]] = []
    try:
        query_rows = _read_index_rows(index_path)
    except sqlite3.DatabaseError:
        # Index may be corrupted or stale on shared storage; rebuild once.
        if index_path.exists():
            index_path.unlink()
        rebuild_index(data_path, index_path)
        query_rows = _read_index_rows(index_path)

    for row in query_rows:
        scorers = json.loads(row[5]) if row[5] else []
        status_by_scorer = json.loads(row[6]) if row[6] else {}
        rows.append(
            {
                "sample_order": row[0],
                "sample_id": row[1],
                "byte_offset": row[2],
                "num_rollouts": row[3],
                "target": row[4],
                "scorers": scorers,
                "status_by_scorer": status_by_scorer,
            }
        )
    return rows, str(index_path)


def load_sample_at_offset(data_path: Path, offset: int) -> dict[str, Any]:
    with data_path.open("rb") as f:
        f.seek(offset)
        raw = f.readline()
    if not raw:
        raise ValueError(f"No row found at offset {offset}")
    return json.loads(raw.decode("utf-8"))


def render_rollout(rollout: dict[str, Any]) -> None:
    meta_fields = [
        "rollout_id",
        "eval_id",
        "task_id",
        "run_id",
        "created",
        "source_owner",
        "run_type",
        "benchmark",
        "model",
        "model_path",
        "hint_fraction",
        "path_hint_level",
        "path_hint_segments",
        "solver_name",
        "sample_file",
        "rollout_ordinal",
        "sample_id",
        "epoch",
        "target",
        "eval_path",
    ]
    st.json({k: rollout.get(k) for k in meta_fields})

    score_outcomes = rollout.get("score_outcomes")
    st.markdown("**Scores (All Scorers)**")
    if isinstance(score_outcomes, dict) and score_outcomes:
        rows: list[dict[str, Any]] = []
        for scorer_name, payload in score_outcomes.items():
            p = payload if isinstance(payload, dict) else {}
            rows.append(
                {
                    "scorer_name": scorer_name,
                    "score_raw_value": p.get("score_raw_value"),
                    "score_normalized": p.get("score_normalized"),
                    "is_correct": p.get("is_correct"),
                    "extracted_answer": p.get("extracted_answer"),
                    "extraction_status": p.get("extraction_status"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No scorer outcomes found for this rollout.")

    st.markdown("**Prompt Text**")
    st.code(rollout.get("prompt_text") or "", language="text")

    st.markdown("**Output Text**")
    st.code(rollout.get("output_text") or "", language="text")


def sample_matches_filters(
    row: dict[str, Any],
    *,
    sample_query: str,
    scorer_filter: str,
    selected_status_values: set[str],
) -> bool:
    if sample_query and sample_query.lower() not in str(row.get("sample_id", "")).lower():
        return False

    if scorer_filter == SCORER_FILTER_ALL:
        return True

    status_by_scorer = row.get("status_by_scorer", {})
    scorer_status = status_by_scorer.get(scorer_filter)
    if not isinstance(scorer_status, dict):
        return False

    for status in selected_status_values:
        if scorer_status.get(status):
            return True
    return False


def main() -> None:
    st.set_page_config(layout="wide", page_title="Hinted Results Viewer")
    st.title("Hinted Results Viewer")

    data_root = Path(
        st.text_input(
            "Data root",
            str(DEFAULT_DATA_ROOT),
            help="Root containing folders like aime_solution/ and aime_cot/",
        )
    )

    col1, col2 = st.columns(2)
    dataset = col1.selectbox("Dataset", KNOWN_DATASETS, index=0)
    family = col2.selectbox("Family", KNOWN_FAMILIES, index=0)

    family_dir = data_root / f"{dataset}_{family}"
    available_roots = sorted([p.name for p in data_root.glob("*_*") if p.is_dir()]) if data_root.exists() else []

    if not family_dir.exists():
        st.warning(f"Results for `{dataset}_{family}` have not been processed yet.")
        if available_roots:
            st.caption("Available processed roots: " + ", ".join(available_roots))
        st.stop()

    model_dirs = sorted([p for p in family_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not model_dirs:
        st.warning(f"No model folders found in {family_dir}")
        st.stop()

    model_name = st.selectbox("Model", [p.name for p in model_dirs], index=0)
    model_dir = family_dir / model_name

    hint_dirs = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("hint_fraction_")],
        key=hint_sort_key,
    )
    if not hint_dirs:
        legacy_hint_files = sorted(
            [p for p in model_dir.glob("hint_fraction_*.jsonl") if p.is_file()],
            key=hint_sort_key,
        )
        if legacy_hint_files:
            st.warning(
                "This model still uses the legacy flat layout. "
                "Run the migration script first."
            )
        else:
            st.warning(f"No hint-fraction folders found in {model_dir}")
        st.stop()

    hint_dir_name = st.selectbox("Hint Fraction", [p.name for p in hint_dirs], index=0)
    hint_dir = model_dir / hint_dir_name

    solver_files = sorted([p for p in hint_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if not solver_files:
        st.warning(f"No solver/type files found in {hint_dir}")
        st.stop()

    default_solver_idx = 0
    solver_file_names = [p.name for p in solver_files]
    if "solution_intext_masked.jsonl" in solver_file_names:
        default_solver_idx = solver_file_names.index("solution_intext_masked.jsonl")
    solver_file_name = st.selectbox(
        "Type / Solver",
        solver_file_names,
        index=default_solver_idx,
    )
    data_path = hint_dir / solver_file_name

    stat = data_path.stat()
    with st.spinner("Loading index..."):
        summary_rows, index_path_str = load_index_rows(str(data_path), stat.st_size, stat.st_mtime_ns)

    st.caption(
        f"File: {data_path} | samples={len(summary_rows):,} | size={stat.st_size:,} bytes | index={index_path_str}"
    )

    all_scorers: set[str] = set()
    for row in summary_rows:
        for scorer_name in row.get("scorers", []):
            all_scorers.add(str(scorer_name))
    scorer_options = [SCORER_FILTER_ALL] + sorted(all_scorers)

    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])
    scorer_filter = filter_col1.selectbox("Scorer Filter", scorer_options, index=0)
    selected_status_labels = filter_col2.multiselect(
        "Status Filter",
        list(STATUS_OPTIONS.keys()),
        default=list(STATUS_OPTIONS.keys()),
    )
    sample_query = filter_col3.text_input("Search sample_id", "")
    selected_status_values = {STATUS_OPTIONS[label] for label in selected_status_labels}

    filtered_rows = [
        row
        for row in summary_rows
        if sample_matches_filters(
            row,
            sample_query=sample_query,
            scorer_filter=scorer_filter,
            selected_status_values=selected_status_values,
        )
    ]

    st.caption(f"Filtered samples: {len(filtered_rows):,}")
    if not filtered_rows:
        st.info("No samples match current filters.")
        st.stop()

    nav_key = safe_slug(f"{dataset}_{family}_{model_name}_{hint_dir_name}_{solver_file_name}")
    state_key = f"selected_sample_{nav_key}"

    sample_ids = [row["sample_id"] for row in filtered_rows]
    if state_key not in st.session_state or st.session_state[state_key] not in sample_ids:
        st.session_state[state_key] = sample_ids[0]
    current_idx = sample_ids.index(st.session_state[state_key])

    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    if nav_col1.button("<", disabled=(current_idx == 0), width="stretch"):
        st.session_state[state_key] = sample_ids[current_idx - 1]
        current_idx = current_idx - 1
    if nav_col3.button(">", disabled=(current_idx == len(sample_ids) - 1), width="stretch"):
        st.session_state[state_key] = sample_ids[current_idx + 1]
        current_idx = current_idx + 1

    selected_sample_id = nav_col2.selectbox(
        "Sample",
        sample_ids,
        index=current_idx,
        key=f"sample_picker_{nav_key}",
    )
    st.session_state[state_key] = selected_sample_id
    current_idx = sample_ids.index(selected_sample_id)

    selected_summary = filtered_rows[current_idx]
    sample_obj = load_sample_at_offset(data_path, selected_summary["byte_offset"])
    rollouts = sample_obj.get("rollouts")
    if not isinstance(rollouts, list):
        st.error("Selected sample row has invalid rollouts format.")
        st.stop()

    st.subheader(f"Sample {selected_sample_id}")
    st.caption(
        f"Sample {current_idx + 1:,} / {len(sample_ids):,} | "
        f"num_rollouts={selected_summary['num_rollouts']} | "
        f"target={selected_summary.get('target')}"
    )

    tab_labels: list[str] = []
    for i, rollout in enumerate(rollouts, start=1):
        if not isinstance(rollout, dict):
            tab_labels.append(f"rollout {i}")
            continue
        ordinal = rollout.get("rollout_ordinal")
        epoch = rollout.get("epoch")
        rollout_id = str(rollout.get("rollout_id") or "")
        tail = rollout_id[-6:] if rollout_id else "none"
        tab_labels.append(f"ord {ordinal} | ep {epoch} | {tail}")

    if not tab_labels:
        st.info("No rollouts found for this sample.")
        st.stop()

    tabs = st.tabs(tab_labels)
    for tab, rollout in zip(tabs, rollouts):
        with tab:
            if isinstance(rollout, dict):
                render_rollout(rollout)
            else:
                st.warning("Rollout payload is not a dict.")


if __name__ == "__main__":
    main()
