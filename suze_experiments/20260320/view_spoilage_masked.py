"""Streamlit viewer for masked spoilage rows (LLM + regex labels).

Run:
    streamlit run suze_experiments/20260320/view_spoilage_masked.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "spoilage_results_llm_regex_masked.jsonl"

DATASET_TO_SOURCE = {
    "aime_solution": PROJECT_ROOT / "christine_experiments/data/solution/aime.jsonl",
    "gpqa_solution": PROJECT_ROOT / "christine_experiments/data/solution/gpqa.jsonl",
    "aime_cot": PROJECT_ROOT / "christine_experiments/data/cot/aime.jsonl",
    "gpqa_cot": PROJECT_ROOT / "christine_experiments/data/cot/gpqa.jsonl",
}

STOP_STRING = "ANSWER:"
MASK_TOKEN = "[MASK]"
DEFAULT_BENCHMARKS = ["aime"]
DEFAULT_HINT_TYPES = ["solution"]
DEFAULT_FRACTIONS = [1]


def _parse_dataset(dataset: str) -> tuple[str, str]:
    if "_" not in dataset:
        return "unknown", "unknown"
    benchmark, hint_type = dataset.split("_", 1)
    return benchmark, hint_type


def truncate_at_stop(text: str, stop_string: str = STOP_STRING) -> str:
    if stop_string not in text:
        return text
    return text[: text.index(stop_string)].strip()


def split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    tokens = re.split(r"(\s+)", text)
    word_indices = [i for i, tok in enumerate(tokens) if tok.strip()]
    return tokens, word_indices


def get_masked_text(text: str, fraction: float, seed: str | None) -> str:
    text = truncate_at_stop(text)
    tokens, word_indices = split_preserving_whitespace(text)
    num_to_mask = int(len(word_indices) * (1 - fraction))
    if not word_indices or num_to_mask <= 0:
        return text

    rng = random.Random(seed) if seed is not None else random
    mask_indices = set(rng.sample(word_indices, num_to_mask))
    return "".join(MASK_TOKEN if i in mask_indices else tok for i, tok in enumerate(tokens)).strip()


@st.cache_data(show_spinner=False)
def load_results(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            benchmark, hint_type = _parse_dataset(str(row["dataset"]))
            row["benchmark"] = benchmark
            row["hint_type"] = hint_type
            rows.append(row)
    df = pd.DataFrame(rows)
    if "source_model" not in df.columns:
        df["source_model"] = None
    if "source_rationalize" not in df.columns:
        df["source_rationalize"] = False
    df["source_model_display"] = df["source_model"].fillna("unknown")
    return df


@st.cache_data(show_spinner=False)
def load_source_entries() -> dict[tuple[str, str, int], dict]:
    by_key: dict[tuple[str, str, int], dict] = {}
    for dataset, path in DATASET_TO_SOURCE.items():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                hint = str(row.get("hint", ""))
                if not hint.strip():
                    continue
                key = (dataset, str(row["id"]), int(row.get("sample_idx", 0)))
                if key in by_key:
                    continue
                by_key[key] = row
    return by_key


def filter_boolean(df: pd.DataFrame, col: str, state: str, allow_missing_label: bool = False) -> pd.DataFrame:
    if state == "Any":
        return df
    if allow_missing_label and state == "No LLM label":
        return df[df[col].isna()]
    wanted = state == "Spoiled"
    return df[df[col] == wanted]


def main() -> None:
    st.set_page_config(page_title="Spoilage Viewer (Masked)", layout="wide")
    st.title("Spoilage Viewer (Masked Hints)")
    st.caption("Viewer includes masked hints and raw hints for inspection.")

    if not DEFAULT_RESULTS_PATH.exists():
        st.error(f"Missing results file: {DEFAULT_RESULTS_PATH}")
        st.code("python suze_experiments/20260320/build_spoilage_results_llm_regex.py")
        return

    df = load_results(str(DEFAULT_RESULTS_PATH))
    if df.empty:
        st.warning("No rows found in merged results file.")
        return

    with st.sidebar:
        st.header("Filters")
        benchmark_options = sorted(df["benchmark"].dropna().unique().tolist())
        hint_type_options = sorted(df["hint_type"].dropna().unique().tolist())
        fraction_options = sorted(df["fraction"].dropna().unique().tolist())
        default_benchmarks = [x for x in DEFAULT_BENCHMARKS if x in benchmark_options] or benchmark_options
        default_hint_types = [x for x in DEFAULT_HINT_TYPES if x in hint_type_options] or hint_type_options
        default_fractions = [x for x in DEFAULT_FRACTIONS if x in fraction_options] or fraction_options

        selected_benchmarks = st.multiselect(
            "Benchmark (AIME/GPQA)",
            options=benchmark_options,
            default=default_benchmarks,
        )
        selected_hint_types = st.multiselect(
            "Hint Type (solution/CoT)",
            options=hint_type_options,
            default=default_hint_types,
        )
        selected_fractions = st.multiselect(
            "Hint Fraction",
            options=fraction_options,
            default=default_fractions,
        )
        model_options = sorted(df["source_model_display"].dropna().unique().tolist())
        selected_models = st.multiselect(
            "Source Model",
            options=model_options,
            default=model_options,
        )
        rationalize_filter = st.selectbox(
            "Rationalize",
            options=["Any", "True", "False"],
            index=0,
        )
        llm_filter = st.selectbox(
            "LLM Judge",
            options=["Any", "Spoiled", "Not spoiled", "No LLM label"],
            index=0,
        )
        regex_filter = st.selectbox("Regex Judge", options=["Any", "Spoiled", "Not spoiled"], index=0)
        require_both_judges = st.checkbox(
            "Only show rows with both LLM and regex labels",
            value=False,
        )
        id_filter = st.text_input("ID filter (contains)", value="").strip()
        sample_idx_options = sorted(int(x) for x in df["sample_idx"].dropna().unique().tolist())
        selected_sample_indices = st.multiselect(
            "Sample Index",
            options=sample_idx_options,
            default=[],
            help="Leave empty to include all sample indices.",
        )
        max_rows = st.slider("Preview rows", min_value=25, max_value=2000, value=300, step=25)

    filtered = df[
        df["benchmark"].isin(selected_benchmarks)
        & df["hint_type"].isin(selected_hint_types)
        & df["fraction"].isin(selected_fractions)
        & df["source_model_display"].isin(selected_models)
    ]
    if rationalize_filter != "Any":
        wanted_rationalize = rationalize_filter == "True"
        filtered = filtered[filtered["source_rationalize"] == wanted_rationalize]
    filtered = filter_boolean(filtered, "llm_spoiled", llm_filter, allow_missing_label=True)
    filtered = filter_boolean(filtered, "regex_spoiled", regex_filter)
    if require_both_judges:
        filtered = filtered[filtered["llm_judged"] == True]
    if id_filter:
        filtered = filtered[filtered["id"].astype(str).str.contains(id_filter, case=False, regex=False)]
    if selected_sample_indices:
        filtered = filtered[filtered["sample_idx"].isin(selected_sample_indices)]
    filtered = filtered.sort_values(
        by=["benchmark", "hint_type", "fraction", "id", "sample_idx", "mask_seed"]
    )

    left, right = st.columns(2)
    with left:
        st.metric("Rows (filtered)", int(len(filtered)))
    with right:
        st.metric("Rows (total)", int(len(df)))

    if filtered.empty:
        st.warning("No rows match current filters.")
        return

    llm_labeled = filtered[filtered["llm_judged"] == True]
    llm_rate = (float(llm_labeled["llm_spoiled"].mean()) * 100) if not llm_labeled.empty else None
    regex_rate = float(filtered["regex_spoiled"].mean()) * 100
    agree_rate = (
        float((llm_labeled["llm_spoiled"] == llm_labeled["regex_spoiled"]).mean()) * 100
        if not llm_labeled.empty
        else None
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LLM Coverage %", f"{(len(llm_labeled) / len(filtered) * 100):.1f}%")
    m2.metric("LLM Spoiled %", "n/a" if llm_rate is None else f"{llm_rate:.1f}%")
    m3.metric("Regex Spoiled %", f"{regex_rate:.1f}%")
    m4.metric("Judge Agreement %", "n/a" if agree_rate is None else f"{agree_rate:.1f}%")

    browse_df = filtered.reset_index(drop=True)
    table_df = browse_df.head(max_rows).copy()
    st.dataframe(
        table_df[
            [
                "dataset",
                "id",
                "sample_idx",
                "fraction",
                "mask_seed",
                "source_model",
                "source_rationalize",
                "llm_spoiled",
                "llm_judged",
                "regex_spoiled",
                "llm_verdict",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    filter_signature = (
        tuple(selected_benchmarks),
        tuple(selected_hint_types),
        tuple(selected_fractions),
        tuple(selected_models),
        rationalize_filter,
        llm_filter,
        regex_filter,
        require_both_judges,
        id_filter,
        tuple(selected_sample_indices),
    )
    id_df = (
        browse_df[["dataset", "id", "benchmark", "hint_type"]]
        .drop_duplicates()
        .sort_values(by=["benchmark", "hint_type", "id"])
        .reset_index(drop=True)
    )
    if st.session_state.get("browse_filter_signature") != filter_signature:
        st.session_state["browse_filter_signature"] = filter_signature
        st.session_state["browse_id_idx"] = 0

    if "browse_id_idx" not in st.session_state:
        st.session_state["browse_id_idx"] = 0
    n_ids = len(id_df)
    if st.session_state["browse_id_idx"] >= n_ids:
        st.session_state["browse_id_idx"] = n_ids - 1
    if st.session_state["browse_id_idx"] < 0:
        st.session_state["browse_id_idx"] = 0

    selected_id_row = id_df.iloc[st.session_state["browse_id_idx"]]
    selected_dataset = str(selected_id_row["dataset"])
    selected_id = str(selected_id_row["id"])
    id_rows = browse_df[
        (browse_df["dataset"] == selected_dataset) & (browse_df["id"] == selected_id)
    ].sort_values(by=["sample_idx", "fraction", "mask_seed"]).reset_index(drop=True)

    source_by_key = load_source_entries()
    source_key = (
        selected_dataset,
        selected_id,
        int(id_rows.iloc[0]["sample_idx"]),
    )
    source = source_by_key.get(source_key)
    if source is None:
        st.error(f"Could not find source row for {source_key}.")
        return

    st.subheader("Example Detail")
    sample_indices = sorted(int(x) for x in id_rows["sample_idx"].unique().tolist())
    metadata = {
        "dataset": selected_dataset,
        "benchmark": str(selected_id_row["benchmark"]).upper(),
        "hint_type": str(selected_id_row["hint_type"]),
        "id": selected_id,
        "n_outputs_for_id": int(len(id_rows)),
        "n_sample_indices": int(id_rows["sample_idx"].nunique()),
        "sample_indices": sample_indices,
        "fractions": sorted(float(x) for x in id_rows["fraction"].unique().tolist()),
        "source_models": sorted(str(x) for x in id_rows["source_model_display"].unique().tolist()),
        "source_rationalize_values": sorted(bool(x) for x in id_rows["source_rationalize"].unique().tolist()),
        "source_file": str(DATASET_TO_SOURCE[selected_dataset]),
    }
    if len(sample_indices) == 1:
        metadata["sample_idx"] = sample_indices[0]

    st.json(
        metadata,
        expanded=True,
    )

    st.markdown("**Question**")
    st.text(str(source.get("question", source.get("prompt", ""))))
    st.markdown("**Target**")
    st.code(str(source.get("target", id_rows.iloc[0].get("target", ""))), language="text")

    st.markdown("**Outputs For This ID**")
    for i, row in id_rows.iterrows():
        sample_idx = int(row["sample_idx"])
        fraction = float(row["fraction"])
        mask_seed = int(row["mask_seed"])
        hint_type = str(row["hint_type"])
        source_model = row["source_model"] if pd.notna(row["source_model"]) else "unknown"
        source_rationalize = bool(row["source_rationalize"])
        llm_judged = bool(row["llm_judged"])
        llm_spoiled_value = None if pd.isna(row["llm_spoiled"]) else bool(row["llm_spoiled"])
        llm_judgment = (
            "No LLM label"
            if not llm_judged
            else ("SPOILED" if llm_spoiled_value else "NOT SPOILED")
        )
        regex_judgment = "SPOILED" if bool(row["regex_spoiled"]) else "NOT SPOILED"

        output_key = (selected_dataset, selected_id, sample_idx)
        output_source = source_by_key.get(output_key)
        if output_source is None:
            continue

        seed = f"{selected_id}_{sample_idx}_{fraction}_{mask_seed}"
        masked_hint = get_masked_text(
            str(output_source.get("hint", "")),
            fraction=fraction,
            seed=seed,
        )

        expander_title = (
            f"Output {i + 1}: sample_idx={sample_idx} | fraction={fraction:.2f} | "
            f"type={hint_type} | mask_seed={mask_seed} | model={source_model} | rationalize={source_rationalize} | "
            f"LLM={llm_judgment} | regex={regex_judgment}"
        )
        with st.expander(expander_title, expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**LLM Judge**")
                st.code(llm_judgment, language="text")
                if llm_judged:
                    st.caption(f"LLM verdict raw: {row.get('llm_verdict', '')}")
            with c2:
                st.markdown("**Regex Judge**")
                st.code(regex_judgment, language="text")
            with c3:
                st.markdown("**Source Metadata**")
                st.code(
                    f"hint_type: {hint_type}\nmodel: {source_model}\nrationalize: {source_rationalize}",
                    language="text",
                )

            st.markdown("**Masked Hint**")
            st.code(masked_hint, language="text")
            st.markdown("**Raw Hint (Unprocessed, includes ANSWER section)**")
            st.code(str(output_source.get("hint", "")), language="text")

    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("< Previous", key="bottom_prev", use_container_width=True):
            st.session_state["browse_id_idx"] = (st.session_state["browse_id_idx"] - 1) % n_ids
            st.rerun()
    with nav_col2:
        st.markdown(
            f"**ID {st.session_state['browse_id_idx'] + 1} / {n_ids}**",
        )
    with nav_col3:
        if st.button("Next >", key="bottom_next", use_container_width=True):
            st.session_state["browse_id_idx"] = (st.session_state["browse_id_idx"] + 1) % n_ids
            st.rerun()


if __name__ == "__main__":
    main()

# streamlit run suze_experiments/20260320/view_spoilage_masked.py
