from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
ECI_SCORES_ROOT = DATA_ROOT / "eci_scores"


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
    if not ECI_SCORES_ROOT.exists():
        return []
    return sorted(path.name for path in ECI_SCORES_ROOT.iterdir() if path.is_dir())


def _discover_models(benchmark_name: str) -> list[str]:
    benchmark_dir = ECI_SCORES_ROOT / benchmark_name
    if not benchmark_dir.exists():
        return []
    return sorted(path.stem for path in benchmark_dir.glob("*.jsonl"))


def _normalize_problem_id_series(df: pd.DataFrame) -> pd.Series:
    if "problem_id" not in df.columns:
        return pd.Series(dtype=str)
    return df["problem_id"].astype(str)


def _meta_get(meta: Any, key: str) -> Any:
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def _grader_get(graders: Any, key: str) -> Any:
    if isinstance(graders, list) and graders:
        grader = graders[0]
        if isinstance(grader, dict):
            if key == "extractor_grader_type":
                return grader.get("extractor_grader_type")
            metadata = grader.get("metadata")
            if isinstance(metadata, dict):
                return metadata.get(key)
    return None


@st.cache_data(show_spinner=False)
def load_scores_df(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "metadata" not in df.columns:
        df["metadata"] = [{} for _ in range(len(df))]
    if "graders" not in df.columns:
        df["graders"] = [[] for _ in range(len(df))]

    df["score_metric"] = df["graders"].apply(lambda g: _grader_get(g, "score_metric"))
    df["score"] = df["graders"].apply(lambda g: _grader_get(g, "score"))
    df["is_correct"] = df["graders"].apply(lambda g: _grader_get(g, "is_correct"))
    df["epoch_in_run"] = df["metadata"].apply(lambda m: _meta_get(m, "epoch_in_run"))
    df["inspect_log_path"] = df["metadata"].apply(lambda m: _meta_get(m, "inspect_log_path"))
    df["sample_error"] = df["metadata"].apply(lambda m: _meta_get(m, "sample_error"))
    df["sample_metadata"] = df["metadata"].apply(lambda m: _meta_get(m, "sample_metadata"))
    df["rendered_prompt"] = df["metadata"].apply(lambda m: _meta_get(m, "rendered_prompt"))
    df["prompt_messages"] = df["metadata"].apply(lambda m: _meta_get(m, "prompt_messages"))
    df["run_metadata"] = df["metadata"].apply(lambda m: _meta_get(m, "run_metadata"))
    return df


def main() -> None:
    st.set_page_config(page_title="ECI Score Viewer", layout="wide")
    st.title("ECI Score Viewer")
    st.caption("Viewer for `data/eci_scores` JSONL files generated from Inspect eval runs.")

    benchmark_options = _discover_benchmarks()
    if not benchmark_options:
        st.error("No ECI score data found under data/eci_scores.")
        return

    with st.sidebar:
        st.header("Data Selection")
        benchmark_name = st.selectbox("Benchmark", options=benchmark_options, index=0)
        model_options = _discover_models(benchmark_name)
        if not model_options:
            st.error(f"No model files found for benchmark={benchmark_name!r}.")
            return
        model_name = st.selectbox("Model", options=model_options, index=0)
        scores_path = ECI_SCORES_ROOT / benchmark_name / f"{model_name}.jsonl"
        st.caption(f"Scores: {scores_path}")

    scores_df = load_scores_df(str(scores_path))
    if scores_df.empty:
        st.warning(f"Score file is missing or empty: {scores_path}")
        return

    with st.sidebar:
        st.header("Filters")
        problem_id_query = st.text_input("Problem ID contains", value="").strip()
        correctness_options = ["all", "correct only", "incorrect only", "error only"]
        correctness_filter = st.selectbox("Correctness", options=correctness_options, index=0)
        max_preview_rows = st.slider("Preview rows", min_value=25, max_value=2000, value=200, step=25)

    filtered_df = scores_df.copy()
    if problem_id_query:
        filtered_df = filtered_df[
            _normalize_problem_id_series(filtered_df).str.contains(problem_id_query, case=False, regex=False)
        ]
    if correctness_filter == "correct only":
        filtered_df = filtered_df[filtered_df["is_correct"] == True]  # noqa: E712
    elif correctness_filter == "incorrect only":
        filtered_df = filtered_df[filtered_df["is_correct"] == False]  # noqa: E712
    elif correctness_filter == "error only":
        filtered_df = filtered_df[filtered_df["is_error"] == True]  # noqa: E712

    total_problem_count = int(_normalize_problem_id_series(scores_df).nunique())
    filtered_problem_count = int(_normalize_problem_id_series(filtered_df).nunique())
    numeric_scores = pd.to_numeric(scores_df["score"], errors="coerce")
    avg_score = float(numeric_scores.mean()) if numeric_scores.notna().any() else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Problems", total_problem_count)
    m2.metric("Rows (all)", int(len(scores_df)))
    m3.metric("Rows (filtered)", int(len(filtered_df)))
    m4.metric("Mean Score", f"{avg_score:.3f}")

    tabs = st.tabs(["Summary", "Coverage", "Problem Browser"])

    with tabs[0]:
        st.subheader("Filtered Score Preview")
        preview_cols = [
            "problem_id",
            "rollout_id",
            "score_metric",
            "score",
            "is_correct",
            "is_error",
            "input_token_count",
            "output_token_count",
            "created_at",
        ]
        keep_cols = [col for col in preview_cols if col in filtered_df.columns]
        if filtered_df.empty:
            st.warning("No score rows match current filters.")
        else:
            st.dataframe(
                filtered_df[keep_cols].sort_values(["problem_id", "rollout_id"]).head(max_preview_rows),
                width="stretch",
                hide_index=True,
            )

        st.subheader("Run Metadata")
        run_metadata = next(
            (
                meta
                for meta in scores_df["run_metadata"].tolist()
                if isinstance(meta, dict) and meta
            ),
            {},
        )
        if run_metadata:
            st.json(run_metadata, expanded=False)

    with tabs[1]:
        st.subheader("Rollout Coverage")
        suggested_rollouts = int(scores_df["rollout_id"].max()) if "rollout_id" in scores_df.columns else 1
        expected_rollouts = st.number_input(
            "Expected rollouts per problem",
            min_value=1,
            value=max(1, suggested_rollouts),
            step=1,
        )

        coverage_df = (
            scores_df.groupby("problem_id", as_index=False)
            .agg(
                rollout_count=("rollout_id", "nunique"),
                mean_score=("score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                error_count=("is_error", "sum"),
            )
        )
        coverage_df["rollout_count"] = coverage_df["rollout_count"].fillna(0).astype(int)
        coverage_df["missing_rollouts"] = (int(expected_rollouts) - coverage_df["rollout_count"]).clip(lower=0)
        coverage_df["is_complete"] = coverage_df["rollout_count"] >= int(expected_rollouts)
        coverage_df["mean_score"] = coverage_df["mean_score"].fillna(0.0)

        incomplete_only = st.checkbox("Show only incomplete problems", value=False)
        shown_df = coverage_df[~coverage_df["is_complete"]].copy() if incomplete_only else coverage_df.copy()

        c1, c2, c3 = st.columns(3)
        complete_problems = int(coverage_df["is_complete"].sum())
        c1.metric("Complete Problems", complete_problems)
        c2.metric("Incomplete Problems", int(len(coverage_df) - complete_problems))
        c3.metric(
            "Filtered Problems",
            filtered_problem_count,
        )

        st.dataframe(
            shown_df.sort_values(["is_complete", "missing_rollouts", "problem_id"], ascending=[True, False, True]),
            width="stretch",
            hide_index=True,
        )

    with tabs[2]:
        st.subheader("Per-Problem Browser")
        if filtered_df.empty:
            st.warning("No score rows match current filters.")
            return

        filtered_problem_ids = filtered_df["problem_id"].astype(str).drop_duplicates().sort_values().tolist()
        if not filtered_problem_ids:
            st.warning("No problems available under current filters.")
            return

        browser_signature = (benchmark_name, model_name, problem_id_query, correctness_filter)
        if st.session_state.get("eci_browser_signature") != browser_signature:
            st.session_state["eci_browser_signature"] = browser_signature
            st.session_state["eci_browser_idx"] = 0

        if "eci_browser_idx" not in st.session_state:
            st.session_state["eci_browser_idx"] = 0

        st.session_state["eci_browser_idx"] = max(
            0,
            min(st.session_state["eci_browser_idx"], len(filtered_problem_ids) - 1),
        )

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("< Previous", key="eci_prev_top", width="stretch"):
                st.session_state["eci_browser_idx"] = (st.session_state["eci_browser_idx"] - 1) % len(
                    filtered_problem_ids
                )
                st.rerun()
        with nav2:
            st.markdown(
                f"**Problem {st.session_state['eci_browser_idx'] + 1} / {len(filtered_problem_ids)}**"
            )
        with nav3:
            if st.button("Next >", key="eci_next_top", width="stretch"):
                st.session_state["eci_browser_idx"] = (st.session_state["eci_browser_idx"] + 1) % len(
                    filtered_problem_ids
                )
                st.rerun()

        selected_problem_id = filtered_problem_ids[st.session_state["eci_browser_idx"]]
        problem_rows = filtered_df[filtered_df["problem_id"].astype(str) == selected_problem_id].copy()
        problem_rows = problem_rows.sort_values(["rollout_id", "created_at"]).reset_index(drop=True)
        row0 = problem_rows.iloc[0]

        st.markdown("**Problem Metadata**")
        st.json(
            {
                "problem_id": selected_problem_id,
                "num_rows_visible": int(len(problem_rows)),
                "benchmark_name": row0.get("benchmark_name"),
                "model": row0.get("model"),
                "answer": row0.get("answer"),
            },
            expanded=True,
        )

        st.markdown("**Question**")
        st.text(str(row0.get("question", "")))
        st.markdown("**Answer**")
        st.code(str(row0.get("answer", "")), language="text")

        st.markdown("**Rollouts**")
        for i, score_row in problem_rows.iterrows():
            title = (
                f"Rollout {int(score_row.get('rollout_id', -1))}: "
                f"score={score_row.get('score')} | "
                f"is_correct={score_row.get('is_correct')} | "
                f"is_error={score_row.get('is_error')}"
            )
            with st.expander(title, expanded=(i == 0)):
                st.markdown("**Score Metadata**")
                st.json(
                    {
                        "created_at": score_row.get("created_at"),
                        "score_metric": score_row.get("score_metric"),
                        "score": score_row.get("score"),
                        "is_correct": score_row.get("is_correct"),
                        "is_error": score_row.get("is_error"),
                        "input_token_count": score_row.get("input_token_count"),
                        "output_token_count": score_row.get("output_token_count"),
                        "epoch_in_run": score_row.get("epoch_in_run"),
                        "sample_error": score_row.get("sample_error"),
                        "sample_metadata": score_row.get("sample_metadata"),
                        "inspect_log_path": score_row.get("inspect_log_path"),
                    },
                    expanded=False,
                )

                with st.expander("Rendered Prompt", expanded=False):
                    rendered_prompt = score_row.get("rendered_prompt")
                    if isinstance(rendered_prompt, str) and rendered_prompt.strip():
                        st.code(rendered_prompt, language="text")
                    else:
                        st.caption("No rendered prompt saved for this row.")

                with st.expander("Prompt Messages", expanded=False):
                    prompt_messages = score_row.get("prompt_messages")
                    if isinstance(prompt_messages, list) and prompt_messages:
                        st.json(prompt_messages, expanded=False)
                    else:
                        st.caption("No prompt messages saved for this row.")

                with st.expander("Model Output", expanded=True):
                    st.code(str(score_row.get("model_output", "")), language="text")

                with st.expander("Graders", expanded=False):
                    st.json(score_row.get("graders", []), expanded=False)

                with st.expander("Run Metadata", expanded=False):
                    st.json(score_row.get("run_metadata", {}), expanded=False)


if __name__ == "__main__":
    # streamlit run runs/view_eci_scores.py
    main()
