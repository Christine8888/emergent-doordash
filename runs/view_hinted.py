from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_config import is_model_excluded_for_fractioner
from src.hinted_outputs import (
    combined_model_response_text,
    extract_provider_reasoning,
    response_text_stats,
)

DATA_ROOT = PROJECT_ROOT / "data"
DATASETS_ROOT = DATA_ROOT / "datasets"
HINTED_INFERENCE_ROOT = DATA_ROOT / "hinted_inference"


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
    if not HINTED_INFERENCE_ROOT.exists():
        return []
    return sorted(path.name for path in HINTED_INFERENCE_ROOT.iterdir() if path.is_dir())


def _discover_models(benchmark_name: str) -> list[str]:
    root = HINTED_INFERENCE_ROOT / benchmark_name
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _split_hint_fractioner(text: str) -> tuple[str, str]:
    if "__" not in text:
        return text, "unknown"
    hint_type, fractioner = text.split("__", 1)
    return hint_type, fractioner


def _discover_hint_fractioners(benchmark_name: str, model_name: str) -> list[str]:
    root = HINTED_INFERENCE_ROOT / benchmark_name / model_name
    if not root.exists():
        return []
    hint_fractioners: list[str] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        _hint_type, fractioner = _split_hint_fractioner(path.name)
        if is_model_excluded_for_fractioner(model_name, fractioner):
            continue
        hint_fractioners.append(path.name)
    return sorted(hint_fractioners)


def _discover_fraction_files(benchmark_name: str, model_name: str, hint_fractioner: str) -> list[Path]:
    root = HINTED_INFERENCE_ROOT / benchmark_name / model_name / hint_fractioner
    if not root.exists():
        return []
    files = list(root.glob("fraction_*.jsonl"))

    def _fraction_sort_key(path: Path) -> float:
        m = re.match(r"^fraction_(.+)\.jsonl$", path.name)
        if not m:
            return float("inf")
        try:
            return float(m.group(1))
        except ValueError:
            return float("inf")

    return sorted(files, key=_fraction_sort_key)


def _extract_primary_grader(row: dict[str, Any]) -> tuple[bool | None, str | None, str | None]:
    graders = row.get("graders")
    if not isinstance(graders, list):
        return None, None, None

    extracted_answer: str | None = None
    grader_type: str | None = None
    for grader in graders:
        if not isinstance(grader, dict):
            continue
        if extracted_answer is None:
            val = grader.get("extracted_answer")
            if isinstance(val, str):
                extracted_answer = val
        if grader_type is None:
            val = grader.get("extractor_grader_type")
            if isinstance(val, str):
                grader_type = val
        is_correct = grader.get("is_correct")
        if isinstance(is_correct, bool):
            return is_correct, extracted_answer, grader_type
    return None, extracted_answer, grader_type


def _grader_label(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _normalize_problem_id_series(df: pd.DataFrame) -> pd.Series:
    if "problem_id" not in df.columns:
        return pd.Series(dtype=str)
    return df["problem_id"].astype(str)


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
def load_hinted_df(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_raw_row"] = rows
    if "metadata" not in df.columns:
        df["metadata"] = [{} for _ in range(len(df))]

    grader_triplets = [_extract_primary_grader(row) for row in rows]
    df["grader_is_correct"] = [triplet[0] for triplet in grader_triplets]
    df["grader_extracted_answer"] = [triplet[1] for triplet in grader_triplets]
    df["grader_type"] = [triplet[2] for triplet in grader_triplets]
    df["grader_label"] = df["grader_is_correct"].apply(_grader_label)
    text_stats = [response_text_stats(row) for row in rows]
    df["provider_reasoning"] = [extract_provider_reasoning(row) for row in rows]
    df["provider_reasoning_chars"] = [stats["provider_reasoning_chars"] for stats in text_stats]
    df["visible_output_chars"] = [stats["visible_output_chars"] for stats in text_stats]
    df["combined_output_chars"] = [stats["combined_output_chars"] for stats in text_stats]
    df["model_output_chars"] = df["visible_output_chars"]

    def _meta_get(meta: Any, key: str) -> Any:
        if isinstance(meta, dict):
            return meta.get(key)
        return None

    df["stop_reason"] = df["metadata"].apply(lambda m: _meta_get(m, "stop_reason"))
    df["slurm_job_id"] = df["metadata"].apply(lambda m: _meta_get(m, "slurm_job_id"))
    return df


def main() -> None:
    st.set_page_config(page_title="Hinted Inference Viewer", layout="wide")
    st.title("Hinted Inference Viewer")
    st.caption("View model outputs from hinted inference runs and grader correctness.")

    benchmark_options = _discover_benchmarks()
    if not benchmark_options:
        st.error("No hinted inference data found under data/hinted_inference.")
        return

    with st.sidebar:
        st.header("Data Selection")
        benchmark_name = st.selectbox("Benchmark", options=benchmark_options, index=0)

        model_options = _discover_models(benchmark_name)
        if not model_options:
            st.error(f"No model folders found for benchmark={benchmark_name!r}.")
            return
        model_name = st.selectbox("Model", options=model_options, index=0)

        hint_fractioner_options = _discover_hint_fractioners(benchmark_name, model_name)
        if not hint_fractioner_options:
            st.error(
                f"No hint/fractioner folders found for benchmark={benchmark_name!r}, model={model_name!r}."
            )
            return

        hint_fractioner = st.selectbox(
            "Hint Type + Fractioner",
            options=hint_fractioner_options,
            index=0,
        )

        fraction_files = _discover_fraction_files(benchmark_name, model_name, hint_fractioner)
        if not fraction_files:
            st.error("No fraction_*.jsonl files found for this selection.")
            return

        selected_fraction_file = st.selectbox(
            "Fraction File",
            options=fraction_files,
            index=0,
            format_func=lambda p: p.name,
        )

        dataset_path = DATASETS_ROOT / f"{benchmark_name}.jsonl"
        st.caption(f"Dataset: {dataset_path}")
        st.caption(f"Inference: {selected_fraction_file}")

    hinted_df = load_hinted_df(str(selected_fraction_file))
    if hinted_df.empty:
        st.warning(f"Inference file is missing or empty: {selected_fraction_file}")
        return

    dataset_df = load_dataset_df(str(dataset_path))

    hint_type, fractioner = _split_hint_fractioner(hint_fractioner)

    with st.sidebar:
        st.header("Filters")
        grader_filter = st.selectbox(
            "Grader Correct",
            options=["all", "true", "false", "unknown"],
            index=0,
        )
        error_filter = st.selectbox(
            "Error Status",
            options=["all", "not_error", "error"],
            index=0,
            format_func=lambda x: {
                "all": "All",
                "not_error": "Not Error",
                "error": "Error Only",
            }[x],
        )
        problem_id_query = st.text_input("Problem ID contains", value="").strip()
        output_query = st.text_input("Response contains (all words)", value="").strip()
        max_preview_rows = st.slider("Preview rows", min_value=25, max_value=2000, value=200, step=25)

    filtered = hinted_df.copy()
    if grader_filter != "all":
        filtered = filtered[filtered["grader_label"] == grader_filter]
    if error_filter == "error":
        filtered = filtered[filtered["is_error"] == True]  # noqa: E712
    elif error_filter == "not_error":
        filtered = filtered[filtered["is_error"] == False]  # noqa: E712
    if problem_id_query:
        filtered = filtered[
            _normalize_problem_id_series(filtered).str.contains(problem_id_query, case=False, regex=False)
        ]
    if output_query:
        output_series = filtered["_raw_row"].apply(lambda row: combined_model_response_text(row))
        for term in output_query.split():
            filtered = filtered[output_series.str.contains(term, case=False, regex=False, na=False)]
            output_series = filtered["_raw_row"].apply(lambda row: combined_model_response_text(row))

    known = filtered["grader_is_correct"].dropna()
    known_total = int(len(known))
    known_correct = int((known == True).sum())  # noqa: E712
    accuracy_text = f"{(known_correct / known_total * 100):.1f}%" if known_total > 0 else "N/A"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Inference Rows (all)", int(len(hinted_df)))
    m2.metric("Inference Rows (filtered)", int(len(filtered)))
    m3.metric("Grader Accuracy", accuracy_text)
    m4.metric("Rows With Reasoning", int((filtered["provider_reasoning_chars"] > 0).sum()))
    if not dataset_df.empty:
        dataset_problem_count = int(dataset_df["problem_id"].astype(str).nunique())
        covered = int(_normalize_problem_id_series(filtered).nunique())
        m5.metric(
            "Problem Coverage",
            f"{(covered / dataset_problem_count * 100):.1f}%" if dataset_problem_count > 0 else "0.0%",
        )
    else:
        m5.metric("Problem Coverage", "N/A")

    tabs = st.tabs(["Summary", "Problem Browser"])

    with tabs[0]:
        st.subheader("Filtered Inference Preview")
        if filtered.empty:
            st.warning("No inference rows match current filters.")
        else:
            preview_cols = [
                "problem_id",
                "inference_id",
                "grader_label",
                "grader_extracted_answer",
                "is_error",
                "input_token_count",
                "output_token_count",
                "provider_reasoning_chars",
                "visible_output_chars",
                "combined_output_chars",
                "model_output_chars",
                "created_at",
                "slurm_job_id",
            ]
            keep_cols = [col for col in preview_cols if col in filtered.columns]
            st.dataframe(
                filtered[keep_cols].sort_values(["problem_id", "inference_id"]).head(max_preview_rows),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Grader Breakdown")
            breakdown = (
                filtered.groupby("grader_label", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )
            st.dataframe(breakdown, width="stretch", hide_index=True)

    with tabs[1]:
        st.subheader("Per-Problem Output Browser")
        if filtered.empty:
            st.warning("No inference rows match current filters.")
            return

        filtered_problem_ids = filtered["problem_id"].astype(str).drop_duplicates().sort_values().tolist()
        if not filtered_problem_ids:
            st.warning("No problems available under current filters.")
            return

        browser_signature = (
            benchmark_name,
            model_name,
            hint_fractioner,
            str(selected_fraction_file),
            grader_filter,
            error_filter,
            problem_id_query,
            output_query,
        )
        if st.session_state.get("hinted_browser_signature") != browser_signature:
            st.session_state["hinted_browser_signature"] = browser_signature
            st.session_state["hinted_browser_idx"] = 0

        if "hinted_browser_idx" not in st.session_state:
            st.session_state["hinted_browser_idx"] = 0
        st.session_state["hinted_browser_idx"] = max(
            0,
            min(st.session_state["hinted_browser_idx"], len(filtered_problem_ids) - 1),
        )

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("< Previous", key="hinted_prev_top", width="stretch"):
                st.session_state["hinted_browser_idx"] = (
                    st.session_state["hinted_browser_idx"] - 1
                ) % len(filtered_problem_ids)
                st.rerun()
        with nav2:
            st.markdown(
                f"**Problem {st.session_state['hinted_browser_idx'] + 1} / {len(filtered_problem_ids)}**"
            )
        with nav3:
            if st.button("Next >", key="hinted_next_top", width="stretch"):
                st.session_state["hinted_browser_idx"] = (
                    st.session_state["hinted_browser_idx"] + 1
                ) % len(filtered_problem_ids)
                st.rerun()

        selected_problem_id = filtered_problem_ids[st.session_state["hinted_browser_idx"]]
        problem_rows = filtered[filtered["problem_id"].astype(str) == selected_problem_id].copy()
        problem_rows = problem_rows.sort_values(["created_at", "inference_id"]).reset_index(drop=True)

        if dataset_df.empty:
            st.warning("Dataset file is missing or empty; showing inference rows only.")
            problem_info = None
        else:
            matching_dataset = dataset_df[dataset_df["problem_id"].astype(str) == selected_problem_id]
            problem_info = matching_dataset.iloc[0] if not matching_dataset.empty else None

        st.markdown("**Problem Metadata**")
        st.json(
            {
                "problem_id": selected_problem_id,
                "model": model_name,
                "hint_type": hint_type,
                "fractioner": fractioner,
                "fraction_file": selected_fraction_file.name,
                "rows_visible": int(len(problem_rows)),
                "dataset_source": problem_info.get("source", None) if problem_info is not None else None,
                "dataset_answer": problem_info.get("answer", None) if problem_info is not None else None,
            },
            expanded=True,
        )

        if problem_info is not None:
            st.markdown("**Question**")
            st.text(str(problem_info.get("question", "")))

        st.markdown("**Inference Rows**")
        for i, row in problem_rows.iterrows():
            title = (
                f"Row {i + 1}: grader={row.get('grader_label')} | "
                f"error={row.get('is_error')} | "
                f"inference_id={row.get('inference_id', '')}"
            )
            with st.expander(title, expanded=True):
                is_correct = row.get("grader_is_correct")
                if is_correct is True:
                    st.success("Grader: TRUE (correct)")
                elif is_correct is False:
                    st.error("Grader: FALSE (incorrect)")
                else:
                    st.warning("Grader: UNKNOWN")

                st.json(
                    {
                        "grader_type": row.get("grader_type"),
                        "grader_extracted_answer": row.get("grader_extracted_answer"),
                        "is_error": row.get("is_error"),
                        "hint_fraction": row.get("hint_fraction"),
                        "input_token_count": row.get("input_token_count"),
                        "output_token_count": row.get("output_token_count"),
                        "provider_reasoning_chars": row.get("provider_reasoning_chars"),
                        "visible_output_chars": row.get("visible_output_chars"),
                        "combined_output_chars": row.get("combined_output_chars"),
                        "created_at": row.get("created_at"),
                        "slurm_job_id": row.get("slurm_job_id"),
                    },
                    expanded=False,
                )

                provider_reasoning = str(row.get("provider_reasoning", "") or "").strip()
                if provider_reasoning:
                    st.markdown("**Provider Reasoning**")
                    st.code(provider_reasoning, language="text")

                st.markdown("**Visible Model Output**")
                st.code(str(row.get("model_output", "")), language="text")

                raw = row.get("_raw_row", {})
                metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
                prompt_text = metadata.get("prompt") if isinstance(metadata, dict) else None

                st.markdown("**Input Prompt**")
                if isinstance(prompt_text, str) and prompt_text.strip():
                    st.code(prompt_text, language="text")
                else:
                    st.warning("Input prompt not found in metadata for this row.")

                hint_obj = raw.get("hint", {}) if isinstance(raw, dict) else {}
                if isinstance(hint_obj, dict):
                    with st.expander("Source Hint Metadata", expanded=False):
                        st.json(
                            {
                                "hint_id": hint_obj.get("hint_id"),
                                "rollout_id": hint_obj.get("rollout_id"),
                                "generator_model": hint_obj.get("generator_model"),
                                "hint_type": hint_obj.get("hint_type"),
                                "created_at": hint_obj.get("created_at"),
                            },
                            expanded=False,
                        )
                    with st.expander("Source Full Hint", expanded=False):
                        st.code(str(hint_obj.get("full_hint", "")), language="text")

                with st.expander("Inference Metadata", expanded=False):
                    st.json(metadata if isinstance(metadata, dict) else {}, expanded=False)


if __name__ == "__main__":
    # streamlit run runs/view_hinted.py
    main()
