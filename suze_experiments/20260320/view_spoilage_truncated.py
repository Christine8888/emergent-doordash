"""Streamlit viewer for truncated spoilage rows (LLM + regex labels).

Run:
    streamlit run suze_experiments/20260320/view_spoilage_truncated.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_RESULTS_PATH = PROJECT_ROOT / "christine_experiments/20260304/spoilage_results.jsonl"

DATASET_TO_SOURCE = {
    "aime_solution": PROJECT_ROOT / "christine_experiments/data/solution/aime.jsonl",
    "gpqa_solution": PROJECT_ROOT / "christine_experiments/data/solution/gpqa.jsonl",
    "aime_cot": PROJECT_ROOT / "christine_experiments/data/cot/aime.jsonl",
    "gpqa_cot": PROJECT_ROOT / "christine_experiments/data/cot/gpqa.jsonl",
}

STOP_STRING = "ANSWER:"
DEFAULT_BENCHMARKS = ["aime"]
DEFAULT_HINT_TYPES = ["solution"]
FRACTION_STEP = 0.01
FRACTION_OPTIONS = [round(i * FRACTION_STEP, 2) for i in range(int(1.0 / FRACTION_STEP) + 1)]
DEFAULT_FRACTIONS = [1.0]
FRACTION_ROUND_DECIMALS = 6


def canonical_fraction(value: float) -> float:
    return round(float(value), FRACTION_ROUND_DECIMALS)


def _parse_dataset(dataset: str) -> tuple[str, str]:
    if "_" not in dataset:
        return "unknown", "unknown"
    benchmark, hint_type = dataset.split("_", 1)
    return benchmark, hint_type


def parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
    return None


def infer_rationalize_from_prompt(prompt: str) -> bool:
    return "HINT: The answer is" in prompt


def extract_source_metadata(row: dict) -> tuple[str | None, bool]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    model = row.get("model") or metadata.get("model")
    raw_rationalize = row.get("rationalize", metadata.get("rationalize"))
    rationalize = parse_bool(raw_rationalize)
    if rationalize is None:
        rationalize = infer_rationalize_from_prompt(str(row.get("prompt", "")))
    return (None if model is None else str(model), bool(rationalize))


def truncate_at_stop(text: str, stop_string: str = STOP_STRING) -> str:
    if stop_string not in text:
        return text
    return text[: text.index(stop_string)].strip()


def split_preserving_whitespace(text: str) -> tuple[list[str], list[int]]:
    tokens = re.split(r"(\s+)", text)
    word_indices = [i for i, tok in enumerate(tokens) if tok.strip()]
    return tokens, word_indices


def get_truncated_split(text: str, fraction: float) -> tuple[str, str, str]:
    """Return (kept_prefix, removed_suffix, full_pre_answer_text)."""
    full_text = truncate_at_stop(text)
    tokens, word_indices = split_preserving_whitespace(full_text)
    if not word_indices:
        return full_text, "", full_text

    if fraction >= 1.0:
        return full_text, "", full_text

    num_words = max(1, int(len(word_indices) * fraction))
    if num_words >= len(word_indices):
        return full_text, "", full_text

    last_idx = word_indices[num_words - 1]
    kept = "".join(tokens[: last_idx + 1]).strip()
    removed = "".join(tokens[last_idx + 1 :])
    return kept, removed, full_text


def target_is_spoiled(prefix_text: str, target: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(target) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, prefix_text))


@st.cache_data(show_spinner=False)
def load_source_rows() -> list[dict]:
    rows: list[dict] = []
    for dataset, path in DATASET_TO_SOURCE.items():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                hint_raw = str(row.get("hint", ""))
                if not hint_raw.strip():
                    continue
                benchmark, hint_type = _parse_dataset(dataset)
                source_model, source_rationalize = extract_source_metadata(row)
                rows.append(
                    {
                        "dataset": dataset,
                        "benchmark": benchmark,
                        "hint_type": hint_type,
                        "id": str(row["id"]),
                        "sample_idx": int(row.get("sample_idx", 0)),
                        "target": str(row.get("target", "")),
                        "question": str(row.get("question", row.get("prompt", ""))),
                        "hint_raw": hint_raw,
                        "source_model": source_model,
                        "source_rationalize": source_rationalize,
                        "source_model_display": source_model if source_model is not None else "unknown",
                    }
                )
    return rows


@st.cache_data(show_spinner=False)
def load_llm_truncated_rows(path_str: str) -> dict[tuple[str, str, int, float], dict]:
    path = Path(path_str)
    by_key: dict[tuple[str, str, int, float], dict] = {}
    if not path.exists():
        return by_key
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (
                str(row["dataset"]),
                str(row["id"]),
                int(row.get("sample_idx", 0)),
                canonical_fraction(float(row["fraction"])),
            )
            by_key[key] = row
    return by_key


def filter_boolean(
    df: pd.DataFrame,
    col: str,
    state: str,
    allow_missing_label: bool = False,
) -> pd.DataFrame:
    if state == "Any":
        return df
    if allow_missing_label and state == "No LLM label":
        return df[df[col].isna()]
    wanted = state == "Spoiled"
    return df[df[col] == wanted]


def build_fraction_rows(source_rows: list[dict], fractions: list[float], llm_map: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for source in source_rows:
        dataset = source["dataset"]
        row_id = source["id"]
        sample_idx = int(source["sample_idx"])
        target = str(source["target"])
        hint_raw = str(source["hint_raw"])

        for fraction in fractions:
            kept, removed, pre_answer = get_truncated_split(hint_raw, fraction)
            llm_row = llm_map.get((dataset, row_id, sample_idx, canonical_fraction(fraction)))
            rows.append(
                {
                    "dataset": dataset,
                    "benchmark": source["benchmark"],
                    "hint_type": source["hint_type"],
                    "id": row_id,
                    "sample_idx": sample_idx,
                    "fraction": float(fraction),
                    "target": target,
                    "question": source["question"],
                    "source_model": source["source_model"],
                    "source_rationalize": bool(source["source_rationalize"]),
                    "source_model_display": source["source_model_display"],
                    "truncated_hint": kept,
                    "removed_hint": removed,
                    "pre_answer_hint": pre_answer,
                    "hint_raw": hint_raw,
                    "llm_spoiled": None if llm_row is None else bool(llm_row["spoiled"]),
                    "llm_verdict": None if llm_row is None else str(llm_row.get("verdict", "")),
                    "llm_judged": llm_row is not None,
                    "regex_spoiled": bool(target_is_spoiled(kept, target)),
                }
            )
    return pd.DataFrame(rows)


def render_truncation_html(kept: str, removed: str) -> None:
    kept_escaped = html.escape(kept).replace("\n", "<br>")
    removed_escaped = html.escape(removed).replace("\n", "<br>")
    line_count = max(1, (kept + removed).count("\n") + 1)
    height_px = min(900, max(120, 24 + 22 * line_count))
    html_block = (
        "<div style='border:1px solid #ddd;border-radius:8px;padding:0.6rem;background:#fff;"
        "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;"
        "line-height:1.45;word-break:break-word;'>"
        f"<span style='color:#111111;'>{kept_escaped}</span>"
        f"<span style='color:#c62828;'>{removed_escaped}</span>"
        "</div>"
    )
    components.html(html_block, height=height_px, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="Spoilage Viewer (Truncated)", layout="wide")
    st.title("Spoilage Viewer (Truncated Hints)")
    st.caption(
        "Truncated hint preview: kept prefix is black, removed suffix is red. "
        "Fractions are configured by FRACTION_OPTIONS in this file."
    )

    source_rows = load_source_rows()
    if not source_rows:
        st.error("No source rows loaded.")
        return
    source_df = pd.DataFrame(source_rows)
    source_by_key = {
        (str(r["dataset"]), str(r["id"]), int(r["sample_idx"])): r for r in source_rows
    }
    llm_map = load_llm_truncated_rows(str(DEFAULT_LLM_RESULTS_PATH))

    with st.sidebar:
        st.header("Filters")
        benchmark_options = sorted(source_df["benchmark"].dropna().unique().tolist())
        hint_type_options = sorted(source_df["hint_type"].dropna().unique().tolist())
        model_options = sorted(source_df["source_model_display"].dropna().unique().tolist())
        sample_idx_options = sorted(int(x) for x in source_df["sample_idx"].dropna().unique().tolist())

        default_benchmarks = [x for x in DEFAULT_BENCHMARKS if x in benchmark_options] or benchmark_options
        default_hint_types = [x for x in DEFAULT_HINT_TYPES if x in hint_type_options] or hint_type_options
        default_fractions = [x for x in DEFAULT_FRACTIONS if x in FRACTION_OPTIONS] or FRACTION_OPTIONS

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
            options=FRACTION_OPTIONS,
            default=default_fractions,
        )
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
        selected_sample_indices = st.multiselect(
            "Sample Index",
            options=sample_idx_options,
            default=[],
            help="Leave empty to include all sample indices.",
        )
        max_rows = st.slider("Preview rows", min_value=25, max_value=2000, value=300, step=25)

    if not selected_fractions:
        st.warning("Select at least one hint fraction.")
        return

    filtered_source = source_df[
        source_df["benchmark"].isin(selected_benchmarks)
        & source_df["hint_type"].isin(selected_hint_types)
        & source_df["source_model_display"].isin(selected_models)
    ]
    if rationalize_filter != "Any":
        wanted_rationalize = rationalize_filter == "True"
        filtered_source = filtered_source[filtered_source["source_rationalize"] == wanted_rationalize]
    if id_filter:
        filtered_source = filtered_source[
            filtered_source["id"].astype(str).str.contains(id_filter, case=False, regex=False)
        ]
    if selected_sample_indices:
        filtered_source = filtered_source[filtered_source["sample_idx"].isin(selected_sample_indices)]

    if filtered_source.empty:
        st.warning("No source rows match current non-fraction filters.")
        return

    selected_fractions_sorted = sorted(float(x) for x in selected_fractions)
    with st.spinner("Building truncated rows for selected filters/fractions..."):
        browse_df = build_fraction_rows(filtered_source.to_dict("records"), selected_fractions_sorted, llm_map)

    browse_df = filter_boolean(browse_df, "llm_spoiled", llm_filter, allow_missing_label=True)
    browse_df = filter_boolean(browse_df, "regex_spoiled", regex_filter)
    if require_both_judges:
        browse_df = browse_df[browse_df["llm_judged"] == True]
    browse_df = browse_df.sort_values(by=["benchmark", "hint_type", "fraction", "id", "sample_idx"])

    left, right = st.columns(2)
    with left:
        st.metric("Rows (filtered)", int(len(browse_df)))
    with right:
        st.metric("Rows (source)", int(len(source_df)))

    if browse_df.empty:
        st.warning("No rows match current filters.")
        return

    llm_labeled = browse_df[browse_df["llm_judged"] == True]
    llm_rate = (float(llm_labeled["llm_spoiled"].mean()) * 100) if not llm_labeled.empty else None
    regex_rate = float(browse_df["regex_spoiled"].mean()) * 100
    agree_rate = (
        float((llm_labeled["llm_spoiled"] == llm_labeled["regex_spoiled"]).mean()) * 100
        if not llm_labeled.empty
        else None
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LLM Coverage %", f"{(len(llm_labeled) / len(browse_df) * 100):.1f}%")
    m2.metric("LLM Spoiled %", "n/a" if llm_rate is None else f"{llm_rate:.1f}%")
    m3.metric("Regex Spoiled %", f"{regex_rate:.1f}%")
    m4.metric("Judge Agreement %", "n/a" if agree_rate is None else f"{agree_rate:.1f}%")

    table_df = browse_df.head(max_rows).copy()
    st.dataframe(
        table_df[
            [
                "dataset",
                "id",
                "sample_idx",
                "fraction",
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
        tuple(round(float(x), 2) for x in selected_fractions_sorted),
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
    if st.session_state.get("browse_filter_signature_truncated") != filter_signature:
        st.session_state["browse_filter_signature_truncated"] = filter_signature
        st.session_state["browse_id_idx_truncated"] = 0

    if "browse_id_idx_truncated" not in st.session_state:
        st.session_state["browse_id_idx_truncated"] = 0
    n_ids = len(id_df)
    if st.session_state["browse_id_idx_truncated"] >= n_ids:
        st.session_state["browse_id_idx_truncated"] = n_ids - 1
    if st.session_state["browse_id_idx_truncated"] < 0:
        st.session_state["browse_id_idx_truncated"] = 0

    selected_id_row = id_df.iloc[st.session_state["browse_id_idx_truncated"]]
    selected_dataset = str(selected_id_row["dataset"])
    selected_id = str(selected_id_row["id"])
    id_rows = browse_df[
        (browse_df["dataset"] == selected_dataset) & (browse_df["id"] == selected_id)
    ].sort_values(by=["sample_idx", "fraction"]).reset_index(drop=True)

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

    st.json(metadata, expanded=True)

    st.markdown("**Question**")
    st.text(str(source.get("question", "")))
    st.markdown("**Target**")
    st.code(str(source.get("target", id_rows.iloc[0].get("target", ""))), language="text")

    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("< Previous", key="top_prev_truncated", use_container_width=True):
            st.session_state["browse_id_idx_truncated"] = (st.session_state["browse_id_idx_truncated"] - 1) % n_ids
            st.rerun()
    with nav_col2:
        st.markdown(
            f"**ID {st.session_state['browse_id_idx_truncated'] + 1} / {n_ids}**",
        )
    with nav_col3:
        if st.button("Next >", key="top_next_truncated", use_container_width=True):
            st.session_state["browse_id_idx_truncated"] = (st.session_state["browse_id_idx_truncated"] + 1) % n_ids
            st.rerun()

    st.markdown("**Outputs For This ID**")
    for i, row in id_rows.iterrows():
        sample_idx = int(row["sample_idx"])
        fraction = float(row["fraction"])
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

        expander_title = (
            f"Output {i + 1}: sample_idx={sample_idx} | fraction={fraction:.2f} | "
            f"type={hint_type} | model={source_model} | rationalize={source_rationalize} | "
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

            st.markdown("**Truncated Hint (black kept, red removed)**")
            render_truncation_html(str(row["truncated_hint"]), str(row["removed_hint"]))
            st.caption("Red text is removed from the pre-ANSWER hint body at this fraction.")

            st.markdown("**Raw Hint (Unprocessed, includes ANSWER section)**")
            st.code(str(output_source.get("hint_raw", "")), language="text")

    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("< Previous", key="bottom_prev_truncated", use_container_width=True):
            st.session_state["browse_id_idx_truncated"] = (st.session_state["browse_id_idx_truncated"] - 1) % n_ids
            st.rerun()
    with nav_col2:
        st.markdown(
            f"**ID {st.session_state['browse_id_idx_truncated'] + 1} / {n_ids}**",
        )
    with nav_col3:
        if st.button("Next >", key="bottom_next_truncated", use_container_width=True):
            st.session_state["browse_id_idx_truncated"] = (st.session_state["browse_id_idx_truncated"] + 1) % n_ids
            st.rerun()


if __name__ == "__main__":
    main()

# streamlit run suze_experiments/20260320/view_spoilage_truncated.py
