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

from src.hint_fractioners import fraction_hint
from src.hint_types import get_hint_type_spec
from src.types import HintGenerationRecord

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


def _discover_hint_types(benchmark_name: str) -> list[str]:
    hint_dir = HINT_GENERATION_ROOT / benchmark_name
    if not hint_dir.exists():
        return []
    return sorted(path.stem for path in hint_dir.glob("*.jsonl"))


def _normalize_problem_id_series(df: pd.DataFrame) -> pd.Series:
    if "problem_id" not in df.columns:
        return pd.Series(dtype=str)
    return df["problem_id"].astype(str)


def _target_is_spoiled(text: str, target: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(target) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, text))


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
def load_hints_df(path_str: str) -> pd.DataFrame:
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

    def _meta_get(meta: Any, key: str) -> Any:
        if isinstance(meta, dict):
            return meta.get(key)
        return None

    df["attempts_used"] = df["metadata"].apply(lambda m: _meta_get(m, "total_attempts_used"))
    df["extracted_answer"] = df["metadata"].apply(lambda m: _meta_get(m, "extracted_answer"))
    df["prompt_version"] = df["metadata"].apply(lambda m: _meta_get(m, "prompt_version"))
    return df


def _apply_fractioning(
    hints_df: pd.DataFrame,
    *,
    fractioner: str,
    hint_fraction: float,
) -> pd.DataFrame:
    if hints_df.empty:
        return hints_df.copy()

    out = hints_df.copy()
    fractioned_texts: list[str] = []
    fractioned_meta: list[dict[str, Any]] = []
    fractioned_error: list[str | None] = []
    fractioned_chars: list[int] = []
    spoilage_eligible: list[bool] = []
    is_spoiled: list[bool | None] = []

    for raw in out["_raw_row"].tolist():
        try:
            record = HintGenerationRecord.model_validate(raw)
            text, meta = fraction_hint(
                hint_record=record,
                fractioner_name=fractioner,
                hint_fraction=hint_fraction,
            )
            fractioned_texts.append(text)
            fractioned_meta.append(meta)
            fractioned_error.append(None)
            fractioned_chars.append(len(text))
            target = str(record.answer).strip()
            if target:
                spoilage_eligible.append(True)
                is_spoiled.append(_target_is_spoiled(text, target))
            else:
                spoilage_eligible.append(False)
                is_spoiled.append(None)
        except Exception as exc:
            fractioned_texts.append("")
            fractioned_meta.append({})
            fractioned_error.append(str(exc))
            fractioned_chars.append(0)
            spoilage_eligible.append(False)
            is_spoiled.append(None)

    out["fractioned_hint"] = fractioned_texts
    out["fractioned_metadata"] = fractioned_meta
    out["fractioned_error"] = fractioned_error
    out["fractioned_chars"] = fractioned_chars
    out["spoilage_eligible"] = spoilage_eligible
    out["is_spoiled"] = is_spoiled
    return out


def main() -> None:
    st.set_page_config(page_title="Fractioned Hint Viewer", layout="wide")
    st.title("Fractioned Hint Viewer")
    st.caption("View transformed hints on the fly from hint_generation JSONL files.")

    benchmark_options = _discover_benchmarks()
    if not benchmark_options:
        st.error("No benchmark data found under data/datasets or data/hint_generation.")
        return

    with st.sidebar:
        st.header("Data Selection")
        benchmark_name = st.selectbox("Benchmark", options=benchmark_options, index=0)
        hint_type_options = _discover_hint_types(benchmark_name)
        if not hint_type_options:
            st.error(f"No hint files found for benchmark={benchmark_name!r}.")
            return
        hint_type = st.selectbox("Hint Type", options=hint_type_options, index=0)

        hint_type_spec = get_hint_type_spec(hint_type)
        fractioner = st.selectbox("Fractioner", options=list(hint_type_spec.allowed_fractioners), index=0)
        hint_fraction = st.slider("Hint Fraction", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

        dataset_path = DATASETS_ROOT / f"{benchmark_name}.jsonl"
        hints_path = HINT_GENERATION_ROOT / benchmark_name / f"{hint_type}.jsonl"
        st.caption(f"Dataset: {dataset_path}")
        st.caption(f"Hints: {hints_path}")

    dataset_df = load_dataset_df(str(dataset_path))
    hints_df = load_hints_df(str(hints_path))

    if dataset_df.empty:
        st.error(f"Dataset file is missing or empty: {dataset_path}")
        return
    if hints_df.empty:
        st.warning(f"Hint file is missing or empty: {hints_path}")
        return

    with st.sidebar:
        st.header("Hint Filters")
        model_options = sorted(hints_df["generator_model"].dropna().astype(str).unique().tolist())
        selected_models = st.multiselect("Generator Model", options=model_options, default=model_options)
        spoilage_filter = st.selectbox(
            "Spoilage",
            options=["all", "spoiled", "not_spoiled"],
            index=1,
            format_func=lambda x: {"all": "All", "spoiled": "Spoiled", "not_spoiled": "Not spoiled"}[x],
        )
        hint_text_query = st.text_input("Hint contains (all words)", value="").strip()
        problem_id_query = st.text_input("Problem ID contains", value="").strip()
        max_preview_rows = st.slider("Preview rows", min_value=25, max_value=2000, value=200, step=25)

    filtered_hints = hints_df[hints_df["generator_model"].astype(str).isin(selected_models)].copy()
    if problem_id_query:
        filtered_hints = filtered_hints[
            _normalize_problem_id_series(filtered_hints).str.contains(problem_id_query, case=False, regex=False)
        ]

    fractioned_df = _apply_fractioning(
        filtered_hints,
        fractioner=fractioner,
        hint_fraction=float(hint_fraction),
    )
    if not fractioned_df.empty and spoilage_filter != "all":
        eligible = fractioned_df["spoilage_eligible"].fillna(False)
        spoiled = fractioned_df["is_spoiled"].fillna(False)
        if spoilage_filter == "spoiled":
            fractioned_df = fractioned_df[eligible & spoiled].copy()
        else:
            fractioned_df = fractioned_df[eligible & (~spoiled)].copy()
    if not fractioned_df.empty and hint_text_query:
        hint_series = fractioned_df["fractioned_hint"].astype(str)
        for term in hint_text_query.split():
            hint_series_match = hint_series.str.contains(term, case=False, regex=False, na=False)
            fractioned_df = fractioned_df[hint_series_match].copy()
            hint_series = fractioned_df["fractioned_hint"].astype(str)

    total_dataset_problems = int(dataset_df["problem_id"].astype(str).nunique())
    hinted_problem_count = int(_normalize_problem_id_series(hints_df).nunique())
    filtered_hint_count = int(len(fractioned_df))
    fraction_error_count = int(fractioned_df["fractioned_error"].notna().sum()) if not fractioned_df.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Problems", total_dataset_problems)
    m2.metric("Hint Rows (filtered)", filtered_hint_count)
    m3.metric("Fraction Errors", fraction_error_count)
    m4.metric("Problem Coverage", f"{(hinted_problem_count / total_dataset_problems * 100):.1f}%")

    tabs = st.tabs(["Summary", "Coverage", "Problem Browser"])

    with tabs[0]:
        st.subheader("Fractioned Hint Preview")
        if fractioned_df.empty:
            st.warning("No hint rows match current filters.")
        else:
            preview_cols = [
                "problem_id",
                "rollout_id",
                "hint_id",
                "generator_model",
                "is_spoiled",
                "fractioned_chars",
                "fractioned_error",
                "created_at",
            ]
            keep_cols = [col for col in preview_cols if col in fractioned_df.columns]
            st.dataframe(
                fractioned_df[keep_cols].sort_values(["problem_id", "rollout_id"]).head(max_preview_rows),
                width="stretch",
                hide_index=True,
            )

    with tabs[1]:
        st.subheader("Rollout Coverage vs Dataset")
        suggested_rollouts = int(hints_df["rollout_id"].max()) + 1 if "rollout_id" in hints_df.columns else 1
        expected_rollouts = st.number_input(
            "Expected rollouts per problem",
            min_value=1,
            value=max(1, suggested_rollouts),
            step=1,
        )

        count_df = (
            hints_df.groupby("problem_id", as_index=False)
            .size()
            .rename(columns={"size": "hint_count"})
        )
        coverage_df = dataset_df[["problem_id", "source", "answer"]].merge(count_df, on="problem_id", how="left")
        coverage_df["hint_count"] = coverage_df["hint_count"].fillna(0).astype(int)
        coverage_df["missing_rollouts"] = (int(expected_rollouts) - coverage_df["hint_count"]).clip(lower=0)
        coverage_df["is_complete"] = coverage_df["hint_count"] >= int(expected_rollouts)

        incomplete_only = st.checkbox("Show only incomplete problems", value=False)
        shown_df = coverage_df[~coverage_df["is_complete"]].copy() if incomplete_only else coverage_df.copy()
        st.dataframe(
            shown_df.sort_values(["is_complete", "missing_rollouts", "problem_id"], ascending=[True, False, True]),
            width="stretch",
            hide_index=True,
        )

    with tabs[2]:
        st.subheader("Per-Problem Fractioned Hint Browser")
        if fractioned_df.empty:
            st.warning("No hint rows match current filters.")
            return

        filtered_problem_ids = (
            fractioned_df["problem_id"].astype(str).drop_duplicates().sort_values().tolist()
        )
        if not filtered_problem_ids:
            st.warning("No problems available under current filters.")
            return

        browser_signature = (
            benchmark_name,
            hint_type,
            fractioner,
            float(hint_fraction),
            tuple(selected_models),
            spoilage_filter,
            hint_text_query,
            problem_id_query,
        )
        if st.session_state.get("fractioned_hint_browser_signature") != browser_signature:
            st.session_state["fractioned_hint_browser_signature"] = browser_signature
            st.session_state["fractioned_hint_browser_idx"] = 0

        if "fractioned_hint_browser_idx" not in st.session_state:
            st.session_state["fractioned_hint_browser_idx"] = 0

        st.session_state["fractioned_hint_browser_idx"] = max(
            0,
            min(st.session_state["fractioned_hint_browser_idx"], len(filtered_problem_ids) - 1),
        )

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("< Previous", key="browse_prev_top", width="stretch"):
                st.session_state["fractioned_hint_browser_idx"] = (
                    st.session_state["fractioned_hint_browser_idx"] - 1
                ) % len(filtered_problem_ids)
                st.rerun()
        with nav2:
            st.markdown(
                f"**Problem {st.session_state['fractioned_hint_browser_idx'] + 1} / {len(filtered_problem_ids)}**"
            )
        with nav3:
            if st.button("Next >", key="browse_next_top", width="stretch"):
                st.session_state["fractioned_hint_browser_idx"] = (
                    st.session_state["fractioned_hint_browser_idx"] + 1
                ) % len(filtered_problem_ids)
                st.rerun()

        selected_problem_id = filtered_problem_ids[st.session_state["fractioned_hint_browser_idx"]]
        problem_row = dataset_df[dataset_df["problem_id"].astype(str) == selected_problem_id]
        problem_hints = fractioned_df[fractioned_df["problem_id"].astype(str) == selected_problem_id].copy()
        problem_hints = problem_hints.sort_values(["rollout_id", "hint_id"]).reset_index(drop=True)

        if problem_row.empty:
            st.error(f"Problem {selected_problem_id} exists in hints but is missing in dataset.")
            return

        row = problem_row.iloc[0]
        st.markdown("**Problem Metadata**")
        st.json(
            {
                "problem_id": selected_problem_id,
                "source": row.get("source", None),
                "answer": row.get("answer", None),
                "num_hints_visible": int(len(problem_hints)),
                "fractioner": fractioner,
                "hint_fraction": float(hint_fraction),
            },
            expanded=True,
        )

        st.markdown("**Question**")
        st.text(str(row.get("question", "")))

        st.markdown("**Hints**")
        for i, hint_row in problem_hints.iterrows():
            title = (
                f"Hint {i + 1}: rollout={int(hint_row.get('rollout_id', -1))} | "
                f"model={hint_row.get('generator_model', '')} | "
                f"hint_id={hint_row.get('hint_id', '')}"
            )
            with st.expander(title, expanded=True):
                error = hint_row.get("fractioned_error")
                if isinstance(error, str) and error:
                    st.error(error)
                    continue

                st.markdown("**Fractioned Hint**")
                st.code(str(hint_row.get("fractioned_hint", "")), language="text")

                fractioned_meta = hint_row.get("fractioned_metadata", {})
                if not isinstance(fractioned_meta, dict):
                    fractioned_meta = {}
                st.markdown("**Fractioning Metadata**")
                st.json(fractioned_meta, expanded=False)
                units_total = fractioned_meta.get("units_total")
                units_visible = fractioned_meta.get("units_visible")
                units_masked = fractioned_meta.get("units_masked")
                if isinstance(units_total, int) and isinstance(units_visible, int) and isinstance(units_masked, int):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Units Total", units_total)
                    c2.metric("Units Visible", units_visible)
                    c3.metric("Units Masked", units_masked)

                if fractioner in ("mask_sentence", "mask_word"):
                    mask_token_count = str(hint_row.get("fractioned_hint", "")).count("[MASK]")
                    st.metric("`[MASK]` Tokens", int(mask_token_count))
                    if mask_token_count == 0:
                        st.warning(
                            "No tokens were masked for this row. "
                            "Try a smaller hint fraction or verify this hint has enough units to mask."
                        )

                meta = hint_row.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                compact_meta = {k: v for k, v in meta.items() if k != "prompt"}
                st.markdown("**Hint Metadata**")
                st.json(
                    {
                        "created_at": hint_row.get("created_at"),
                        "input_token_count": hint_row.get("input_token_count"),
                        "output_token_count": hint_row.get("output_token_count"),
                        "attempts_used": hint_row.get("attempts_used"),
                        "extracted_answer": hint_row.get("extracted_answer"),
                        "prompt_version": hint_row.get("prompt_version"),
                        "metadata": compact_meta,
                    },
                    expanded=False,
                )

                with st.expander("Full Hint", expanded=False):
                    st.code(str(hint_row.get("full_hint", "")), language="text")

                prompt_text = meta.get("prompt") if isinstance(meta, dict) else None
                if isinstance(prompt_text, str) and prompt_text.strip():
                    with st.expander("Prompt Used", expanded=False):
                        st.code(prompt_text, language="text")

                with st.expander("Raw Model Output", expanded=False):
                    st.code(str(hint_row.get("model_output", "")), language="text")


if __name__ == "__main__":
    # streamlit run runs/view_fractioned_hints.py
    main()
