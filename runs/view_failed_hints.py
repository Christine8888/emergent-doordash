from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATASETS_ROOT = DATA_ROOT / "datasets"
HINT_GENERATION_ROOT = DATA_ROOT / "hint_generation"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _discover_benchmarks() -> list[str]:
    names = set()
    if DATASETS_ROOT.exists():
        for path in DATASETS_ROOT.glob("*.jsonl"):
            names.add(path.stem)
    if HINT_GENERATION_ROOT.exists():
        for path in HINT_GENERATION_ROOT.glob("*"):
            if path.is_dir():
                names.add(path.name)
    return sorted(names)


def _discover_failed_hint_types(benchmark_name: str) -> list[str]:
    hint_dir = HINT_GENERATION_ROOT / benchmark_name
    if not hint_dir.exists():
        return []

    hint_types: list[str] = []
    for path in hint_dir.glob("*_failed.jsonl"):
        stem = path.stem
        if stem.endswith("_failed"):
            hint_types.append(stem[: -len("_failed")])
    return sorted(set(hint_types))


@st.cache_data(show_spinner=False)
def load_dataset_df(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["problem_id", "question", "answer", "source"])
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame(columns=["problem_id", "question", "answer", "source"])
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_failed_df(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "model" not in df.columns and "generator_model" in df.columns:
        df["model"] = df["generator_model"]

    def _as_str(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str)

    if "thinking" in df.columns:
        df["thinking_chars"] = _as_str(df["thinking"]).str.len()
    else:
        df["thinking_chars"] = 0

    problem_id_s = _as_str(df.get("problem_id", pd.Series(dtype=object)))
    rollout_s = _as_str(df.get("rollout_id", pd.Series(dtype=object)))
    hint_id_s = _as_str(df.get("hint_id", pd.Series(dtype=object)))
    df["task_key"] = problem_id_s + "::" + rollout_s + "::" + hint_id_s
    return df


def _normalize_problem_id_series(df: pd.DataFrame) -> pd.Series:
    if "problem_id" not in df.columns:
        return pd.Series(dtype=str)
    return df["problem_id"].astype(str)


def main() -> None:
    st.set_page_config(page_title="Failed Hint Viewer", layout="wide")
    st.title("Failed Hint Attempts Viewer")
    st.caption("Inspect *_failed.jsonl files from hint generation.")

    benchmark_options = _discover_benchmarks()
    if not benchmark_options:
        st.error("No benchmark data found under data/datasets or data/hint_generation.")
        return

    with st.sidebar:
        st.header("Data Selection")
        benchmark_name = st.selectbox("Benchmark", options=benchmark_options, index=0)
        hint_type_options = _discover_failed_hint_types(benchmark_name)
        if not hint_type_options:
            st.error(f"No *_failed.jsonl files found for benchmark={benchmark_name!r}.")
            return
        hint_type = st.selectbox("Hint Type", options=hint_type_options, index=0)

        dataset_path = DATASETS_ROOT / f"{benchmark_name}.jsonl"
        failed_path = HINT_GENERATION_ROOT / benchmark_name / f"{hint_type}_failed.jsonl"

        st.caption(f"Dataset: {dataset_path}")
        st.caption(f"Failed Hints: {failed_path}")

    dataset_df = load_dataset_df(str(dataset_path))
    failed_df = load_failed_df(str(failed_path))

    if failed_df.empty:
        st.warning(f"Failed hint file is missing or empty: {failed_path}")
        return

    with st.sidebar:
        st.header("Filters")
        if "model" in failed_df.columns:
            model_options = sorted(failed_df["model"].dropna().astype(str).unique().tolist())
            selected_models = st.multiselect("Model", options=model_options, default=model_options)
        else:
            selected_models = []

        failure_type_options = sorted(failed_df.get("failure_type", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
        selected_failure_types = st.multiselect(
            "Failure Type",
            options=failure_type_options,
            default=failure_type_options,
        )

        task_outcome = st.selectbox(
            "Task Outcome",
            options=["all", "failed_only", "eventually_succeeded"],
            index=0,
        )
        problem_id_query = st.text_input("Problem ID contains", value="").strip()
        max_preview_rows = st.slider("Preview rows", min_value=25, max_value=2000, value=200, step=25)

    filtered = failed_df.copy()
    if selected_models and "model" in filtered.columns:
        filtered = filtered[filtered["model"].astype(str).isin(selected_models)]
    if selected_failure_types and "failure_type" in filtered.columns:
        filtered = filtered[filtered["failure_type"].astype(str).isin(selected_failure_types)]
    if task_outcome != "all" and "task_succeeded" in filtered.columns:
        if task_outcome == "failed_only":
            filtered = filtered[filtered["task_succeeded"] == False]  # noqa: E712
        elif task_outcome == "eventually_succeeded":
            filtered = filtered[filtered["task_succeeded"] == True]  # noqa: E712
    if problem_id_query:
        filtered = filtered[
            _normalize_problem_id_series(filtered).str.contains(problem_id_query, case=False, regex=False)
        ]

    total_dataset_problems = int(dataset_df["problem_id"].astype(str).nunique()) if not dataset_df.empty else 0
    unique_problem_count = int(_normalize_problem_id_series(filtered).nunique()) if not filtered.empty else 0
    unique_task_count = int(filtered["task_key"].nunique()) if not filtered.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Failed Attempts (all)", int(len(failed_df)))
    m2.metric("Failed Attempts (filtered)", int(len(filtered)))
    m3.metric("Unique Failed Tasks", unique_task_count)
    if total_dataset_problems > 0:
        m4.metric("Problem Coverage", f"{(unique_problem_count / total_dataset_problems * 100):.1f}%")
    else:
        m4.metric("Unique Problems", unique_problem_count)

    tabs = st.tabs(["Summary", "Breakdown", "Problem Browser"])

    with tabs[0]:
        st.subheader("Filtered Failed Attempts")
        if filtered.empty:
            st.warning("No failed attempts match current filters.")
        else:
            preview_cols = [
                "problem_id",
                "rollout_id",
                "attempt",
                "model",
                "failure_type",
                "task_succeeded",
                "input_token_count",
                "output_token_count",
                "thinking_chars",
                "stop_reason",
            ]
            keep_cols = [c for c in preview_cols if c in filtered.columns]
            sort_cols = [c for c in ["problem_id", "rollout_id", "attempt"] if c in filtered.columns]
            shown = filtered[keep_cols]
            if sort_cols:
                shown = shown.sort_values(sort_cols)
            st.dataframe(shown.head(max_preview_rows), width="stretch", hide_index=True)

    with tabs[1]:
        if filtered.empty:
            st.warning("No failed attempts match current filters.")
        else:
            st.subheader("Failure Type Breakdown")
            if "failure_type" in filtered.columns:
                failure_breakdown = (
                    filtered.groupby("failure_type", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                    .sort_values("count", ascending=False)
                )
                st.dataframe(failure_breakdown, width="stretch", hide_index=True)
            else:
                st.info("No failure_type column present.")

            st.subheader("Model Breakdown")
            if "model" in filtered.columns:
                model_breakdown = (
                    filtered.groupby("model", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                    .sort_values("count", ascending=False)
                )
                st.dataframe(model_breakdown, width="stretch", hide_index=True)
            else:
                st.info("No model column present.")

            st.subheader("Most Affected Problems")
            problem_breakdown = (
                filtered.groupby("problem_id", as_index=False)
                .size()
                .rename(columns={"size": "failed_attempts"})
                .sort_values("failed_attempts", ascending=False)
            )
            st.dataframe(problem_breakdown.head(200), width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("Per-Problem Failed Attempt Browser")
        if filtered.empty:
            st.warning("No failed attempts match current filters.")
            return

        filtered_problem_ids = filtered["problem_id"].astype(str).drop_duplicates().sort_values().tolist()
        if not filtered_problem_ids:
            st.warning("No problems available under current filters.")
            return

        browser_signature = (
            benchmark_name,
            hint_type,
            tuple(selected_models),
            tuple(selected_failure_types),
            task_outcome,
            problem_id_query,
        )
        if st.session_state.get("failed_hint_browser_signature") != browser_signature:
            st.session_state["failed_hint_browser_signature"] = browser_signature
            st.session_state["failed_hint_browser_idx"] = 0

        if "failed_hint_browser_idx" not in st.session_state:
            st.session_state["failed_hint_browser_idx"] = 0

        st.session_state["failed_hint_browser_idx"] = max(
            0,
            min(st.session_state["failed_hint_browser_idx"], len(filtered_problem_ids) - 1),
        )

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("< Previous", key="failed_browse_prev_top", width="stretch"):
                st.session_state["failed_hint_browser_idx"] = (
                    st.session_state["failed_hint_browser_idx"] - 1
                ) % len(filtered_problem_ids)
                st.rerun()
        with nav2:
            st.markdown(f"**Problem {st.session_state['failed_hint_browser_idx'] + 1} / {len(filtered_problem_ids)}**")
        with nav3:
            if st.button("Next >", key="failed_browse_next_top", width="stretch"):
                st.session_state["failed_hint_browser_idx"] = (
                    st.session_state["failed_hint_browser_idx"] + 1
                ) % len(filtered_problem_ids)
                st.rerun()

        selected_problem_id = filtered_problem_ids[st.session_state["failed_hint_browser_idx"]]
        problem_failures = filtered[filtered["problem_id"].astype(str) == selected_problem_id].copy()
        sort_cols = [c for c in ["rollout_id", "attempt"] if c in problem_failures.columns]
        if sort_cols:
            problem_failures = problem_failures.sort_values(sort_cols).reset_index(drop=True)

        if not dataset_df.empty:
            problem_row = dataset_df[dataset_df["problem_id"].astype(str) == selected_problem_id]
        else:
            problem_row = pd.DataFrame()

        st.markdown("**Problem Metadata**")
        problem_json: dict[str, Any] = {
            "problem_id": selected_problem_id,
            "failed_attempts_visible": int(len(problem_failures)),
            "failed_rollouts_visible": int(problem_failures["rollout_id"].nunique()) if "rollout_id" in problem_failures.columns else None,
        }
        if not problem_row.empty:
            row = problem_row.iloc[0]
            problem_json["source"] = row.get("source", None)
            problem_json["answer"] = row.get("answer", None)
            question_text = str(row.get("question", ""))
            answer_text = str(row.get("answer", ""))
        else:
            first_failure = problem_failures.iloc[0]
            question_text = str(first_failure.get("question", ""))
            answer_text = str(first_failure.get("answer", ""))
        st.json(problem_json, expanded=True)

        st.markdown("**Question**")
        st.text(question_text)
        st.markdown("**Answer**")
        st.code(answer_text, language="text")

        st.markdown("**Failed Attempts**")
        for i, row in problem_failures.iterrows():
            title = (
                f"Attempt {i + 1}: rollout={row.get('rollout_id', '')} | "
                f"attempt={row.get('attempt', '')} | "
                f"model={row.get('model', '')} | "
                f"failure={row.get('failure_type', '')}"
            )
            with st.expander(title, expanded=(i == 0)):
                st.markdown("**Attempt Metadata**")
                st.json(
                    {
                        "hint_id": row.get("hint_id"),
                        "task_succeeded": row.get("task_succeeded"),
                        "input_token_count": row.get("input_token_count"),
                        "output_token_count": row.get("output_token_count"),
                        "stop_reason": row.get("stop_reason"),
                        "extracted_answer": row.get("extracted_answer"),
                        "failure_error": row.get("failure_error"),
                        "thinking_mode": row.get("thinking_mode"),
                        "thinking_budget_tokens": row.get("thinking_budget_tokens"),
                        "thinking_chars": row.get("thinking_chars"),
                        "grader_metadata": row.get("grader_metadata"),
                    },
                    expanded=False,
                )

                prompt_text = row.get("prompt")
                if isinstance(prompt_text, str) and prompt_text.strip():
                    with st.expander("Prompt Used", expanded=False):
                        st.code(prompt_text, language="text")

                with st.expander("Raw Model Output", expanded=False):
                    st.code(str(row.get("model_output", "")), language="text")

                thinking_text = row.get("thinking")
                if isinstance(thinking_text, str) and thinking_text.strip():
                    with st.expander("Thinking", expanded=False):
                        st.code(thinking_text, language="text")


if __name__ == "__main__":
    # streamlit run runs/view_failed_hints.py
    main()
