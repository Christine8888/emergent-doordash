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
    return pd.DataFrame(rows)


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
    st.caption("Raw hints are never shown in this viewer. Display uses masked text only.")

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

        selected_benchmarks = st.multiselect(
            "Benchmark (AIME/GPQA)",
            options=benchmark_options,
            default=benchmark_options,
        )
        selected_hint_types = st.multiselect(
            "Hint Type (solution/CoT)",
            options=hint_type_options,
            default=hint_type_options,
        )
        selected_fractions = st.multiselect(
            "Hint Fraction",
            options=fraction_options,
            default=fraction_options,
        )
        llm_filter = st.selectbox(
            "LLM Judge",
            options=["Any", "Spoiled", "Not spoiled", "No LLM label"],
            index=0,
        )
        regex_filter = st.selectbox("Regex Judge", options=["Any", "Spoiled", "Not spoiled"], index=0)
        max_rows = st.slider("Rows to show", min_value=25, max_value=2000, value=300, step=25)

    filtered = df[
        df["benchmark"].isin(selected_benchmarks)
        & df["hint_type"].isin(selected_hint_types)
        & df["fraction"].isin(selected_fractions)
    ]
    filtered = filter_boolean(filtered, "llm_spoiled", llm_filter, allow_missing_label=True)
    filtered = filter_boolean(filtered, "regex_spoiled", regex_filter)
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

    table_df = filtered.head(max_rows).copy().reset_index(drop=True)
    st.dataframe(
        table_df[
            [
                "dataset",
                "id",
                "sample_idx",
                "fraction",
                "mask_seed",
                "llm_spoiled",
                "llm_judged",
                "regex_spoiled",
                "llm_verdict",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    def format_row(idx: int) -> str:
        row = table_df.iloc[idx]
        return (
            f"{row['dataset']} | id={row['id']} | sample={row['sample_idx']} | "
            f"f={row['fraction']} | LLM={row['llm_spoiled']} | regex={row['regex_spoiled']}"
        )

    selected_idx = st.selectbox(
        "Inspect row",
        options=list(range(len(table_df))),
        format_func=format_row,
        index=0,
    )
    selected = table_df.iloc[selected_idx]
    llm_spoiled_value = None if pd.isna(selected["llm_spoiled"]) else bool(selected["llm_spoiled"])

    source_by_key = load_source_entries()
    key = (str(selected["dataset"]), str(selected["id"]), int(selected["sample_idx"]))
    source = source_by_key.get(key)
    if source is None:
        st.error(f"Could not find source row for {key}.")
        return

    st.subheader("Example Detail")
    st.write(
        {
            "dataset": selected["dataset"],
            "id": selected["id"],
            "sample_idx": int(selected["sample_idx"]),
            "fraction": float(selected["fraction"]),
            "mask_seed": int(selected["mask_seed"]),
            "llm_spoiled": llm_spoiled_value,
            "llm_judged": bool(selected["llm_judged"]),
            "regex_spoiled": bool(selected["regex_spoiled"]),
        }
    )

    seed = (
        f"{selected['id']}_{int(selected['sample_idx'])}_"
        f"{float(selected['fraction'])}_{int(selected['mask_seed'])}"
    )
    masked_hint = get_masked_text(
        str(source.get("hint", "")),
        fraction=float(selected["fraction"]),
        seed=seed,
    )

    st.markdown("**Question**")
    st.text(str(source.get("question", source.get("prompt", ""))))
    st.markdown("**Masked Hint**")
    st.code(masked_hint, language="text")

    show_target = st.checkbox("Show target", value=False)
    if show_target:
        st.code(str(source.get("target", selected.get("target", ""))), language="text")


if __name__ == "__main__":
    main()
