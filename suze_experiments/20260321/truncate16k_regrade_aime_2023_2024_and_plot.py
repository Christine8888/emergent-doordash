from __future__ import annotations

import asyncio
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from transformers import AutoTokenizer


# --- Editable constants ---
INPUT_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2_regraded")
INPUT_DATASET_FAMILY = "aime_solution"
SOLVER_FILE = "solution_intext_masked.jsonl"

# Consolidated-hinted-results-style output (primary result artifact).
OUTPUT_RESULTS_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2_trunc16k_aime_2023_2024")
OUTPUT_DATASET_FAMILY = "aime_solution"

OLD_SCORER_NAME = "aime_scorer_v2"
NEW_SCORER_NAME = "aime_scorer_v2_trunc16k"
MAX_TOKENS = 16000

MODEL_LIST = [
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    # "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Qwen3-4B",
    "Qwen3-8B",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gemma-3-4b-it",
]
HINT_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
YEAR_PREFIXES = ("2023-", "2024-")

OUTPUT_DIR = Path("suze_experiments/20260321/plots")
ROLLOUT_CSV = OUTPUT_DIR / "aime_2023_2024_solution_intext_masked_trunc16k_rollouts.csv"
OWNER_CSV = OUTPUT_DIR / "aime_2023_2024_solution_intext_masked_trunc16k_owner_by_model_hint.csv"
AGG_CSV = OUTPUT_DIR / "aime_2023_2024_solution_intext_masked_trunc16k_accuracy_by_model_hint.csv"
PLOT_ACCURACY = OUTPUT_DIR / "aime_2023_2024_solution_intext_masked_trunc16k_accuracy_vs_hint_by_model.png"
PLOT_HIST_OLD = OUTPUT_DIR / "aime_2023_2024_solution_intext_masked_old_token_length_hist_by_model.png"

REUSE_EXISTING_RESULTS_FILES = True
REUSE_EXISTING_ROLLOUT_CSV = True
REUSE_EXISTING_OWNER_CSV = True

ECI_FILE = Path("christine_experiments/20260129_fitting/eci_model_capabilities.csv")
# --------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from environments.math.utils import grade_math_answer


OWNER_COLORS = {
    "mine": "#1f77b4",
    "christine": "#ff7f0e",
    "mixed": "#2ca02c",
    "unknown": "#7f7f7f",
}


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def suppress_known_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*antlr4\.error\.ErrorListener module is not installed.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TRANSFORMERS_CACHE.*deprecated.*",
        category=FutureWarning,
    )


def model_to_repo_id(model_name: str) -> str:
    if model_name.startswith("Qwen2.5-") or model_name.startswith("Qwen3-"):
        return f"Qwen/{model_name}"
    if model_name.startswith("Llama-3.1-"):
        return f"meta-llama/{model_name}"
    if model_name.startswith("gemma-3-"):
        return f"google/{model_name}"
    raise ValueError(f"Unknown model naming pattern for tokenizer mapping: {model_name}")


def load_tokenizer_for_model(model_name: str) -> AutoTokenizer:
    repo_id = model_to_repo_id(model_name)
    print(f"[{ts_now()}] Loading tokenizer for {model_name} from Hugging Face repo {repo_id}")
    return AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=True,
    )


def should_keep_sample(sample_id: str) -> bool:
    return sample_id.startswith(YEAR_PREFIXES)


def remove_boxed(s: str) -> str:
    s = s.strip()
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :].strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1].strip()
    return s


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        tail = string.split("\\boxed ")[-1]
        token = tail.split("$")[0].strip()
        return "\\boxed " + token if token else None
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def clean_latex_and_markdown(text: str) -> str:
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\\\\\[|\\\\\]", "", text)
    text = re.sub(r"\\\\\(|\\\\\)", "", text)
    text = re.sub(r"\$", "", text)
    text = text.strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        depth = 0
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and i == len(text) - 1:
                    text = text[7:-1]
                    break
    text = text.rstrip(" .,:;!`*_")
    text = re.sub(r"(?i)^\s*(?:target\s+answer|final\s+answer|answer)\s*:\s*", "", text).strip()
    return text


def extract_last_full_number(text: str) -> str | None:
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9_])-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])",
            text,
        )
    )
    if not matches:
        return None
    return matches[-1].group(0).replace(",", "")


def extract_answer_fixed(completion: str) -> str:
    pattern = (
        r"(?im)(?:^|\n)\s*(?:[\*\-_`>#]+\s*)?"
        r"(?:target\s+answer|final\s+answer|answer)\s*:[ \t]*([^\n]+)"
    )
    matches = list(re.finditer(pattern, completion, re.MULTILINE))
    if matches:
        raw_answer = matches[-1].group(1).strip()
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer:
            return clean_latex_and_markdown(remove_boxed(boxed_answer))
        cleaned = clean_latex_and_markdown(raw_answer)
        if cleaned:
            return cleaned

    boxed_answer = last_boxed_only_string(completion)
    if boxed_answer:
        return clean_latex_and_markdown(remove_boxed(boxed_answer))

    number = extract_last_full_number(completion)
    if number is not None:
        return number

    return ""


def normalize_owner(owner: Any) -> str:
    token = str(owner or "").strip().lower()
    if token in {"suzeva", "suze", "mine"}:
        return "mine"
    if token in {"christine"}:
        return "christine"
    return "unknown"


def owner_label_from_counts(counts: dict[str, int]) -> str:
    nonzero = [k for k, v in counts.items() if v > 0]
    if not nonzero:
        return "unknown"
    if len(nonzero) == 1:
        return nonzero[0]
    return "mixed"


def parse_bool_to_int(value: Any) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    return None


async def grade_with_aime_scorer_v2(output_text: str, target: str | None) -> dict[str, Any]:
    extracted_answer = extract_answer_fixed(output_text)
    extraction_status = "ok" if extracted_answer and extracted_answer.strip() else "failed"

    if extraction_status == "ok" and target is not None and target.strip():
        is_correct = await grade_math_answer(
            answer=str(extracted_answer),
            target=target.strip(),
            exact_match=True,
            use_sympy=True,
        )
        score_raw_value = "C" if is_correct else "I"
    else:
        is_correct = False
        score_raw_value = "I"

    return {
        "score_raw_value": score_raw_value,
        "score_normalized": score_raw_value,
        "is_correct": bool(is_correct),
        "extracted_answer": extracted_answer,
        "extraction_status": extraction_status,
    }


async def process_one_file(input_path: Path, output_path: Path, tokenizer: AutoTokenizer) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    kept_samples = 0
    kept_rollouts = 0

    with input_path.open("r", encoding="utf-8") as in_f, tmp_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            sample_obj = json.loads(line)
            sample_id = str(sample_obj.get("sample_id") or "")
            if not should_keep_sample(sample_id):
                continue

            rollouts = sample_obj.get("rollouts")
            if not isinstance(rollouts, list):
                continue

            updated_rollouts: list[dict[str, Any]] = []
            for rollout in rollouts:
                if not isinstance(rollout, dict):
                    continue

                output_text = str(rollout.get("output_text") or "")
                target = rollout.get("target")
                target_str = str(target).strip() if target is not None else None

                old_token_ids = tokenizer.encode(output_text, add_special_tokens=False)
                old_token_len = len(old_token_ids)
                new_token_ids = old_token_ids[:MAX_TOKENS]
                new_token_len = len(new_token_ids)
                truncated_text = tokenizer.decode(
                    new_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                score_outcomes = rollout.get("score_outcomes")
                score_outcomes_copy = dict(score_outcomes) if isinstance(score_outcomes, dict) else {}
                new_outcome = await grade_with_aime_scorer_v2(truncated_text, target_str)
                score_outcomes_copy[NEW_SCORER_NAME] = new_outcome

                rollout_out = dict(rollout)
                rollout_out["output_text"] = truncated_text
                rollout_out["old_token_len"] = old_token_len
                rollout_out["new_token_len"] = new_token_len
                rollout_out["was_truncated"] = bool(old_token_len > MAX_TOKENS)
                rollout_out["score_outcomes"] = score_outcomes_copy
                updated_rollouts.append(rollout_out)
                kept_rollouts += 1

            if updated_rollouts:
                sample_out = dict(sample_obj)
                sample_out["rollouts"] = updated_rollouts
                out_f.write(json.dumps(sample_out, ensure_ascii=False) + "\n")
                kept_samples += 1

    os.replace(tmp_path, output_path)
    return kept_samples, kept_rollouts


async def build_results_tree() -> None:
    input_family = INPUT_ROOT / INPUT_DATASET_FAMILY
    output_family = OUTPUT_RESULTS_ROOT / OUTPUT_DATASET_FAMILY
    if not input_family.exists():
        raise FileNotFoundError(f"Missing input family dir: {input_family}")

    total_samples = 0
    total_rollouts = 0

    for model in MODEL_LIST:
        try:
            tokenizer = load_tokenizer_for_model(model)
        except Exception as exc:
            print(
                f"[{ts_now()}] WARNING failed to load tokenizer for model={model}; skipping model. "
                f"error={exc}",
                flush=True,
            )
            continue
        for hint in HINT_LEVELS:
            input_path = input_family / model / f"hint_fraction_{hint:.1f}" / SOLVER_FILE
            output_path = output_family / model / f"hint_fraction_{hint:.1f}" / SOLVER_FILE
            if not input_path.exists():
                print(f"[{ts_now()}] WARNING missing input file: {input_path}")
                continue

            if REUSE_EXISTING_RESULTS_FILES and output_path.exists():
                print(f"[{ts_now()}] Skip existing output: {output_path}")
                continue

            print(f"[{ts_now()}] Processing model={model} hint={hint:.1f}")
            kept_samples, kept_rollouts = await process_one_file(input_path, output_path, tokenizer)
            total_samples += kept_samples
            total_rollouts += kept_rollouts
            print(
                f"[{ts_now()}] Wrote {output_path} kept_samples={kept_samples} kept_rollouts={kept_rollouts}",
                flush=True,
            )

    print(
        f"[{ts_now()}] Results-tree build done new_samples={total_samples} new_rollouts={total_rollouts} "
        f"root={output_family}",
        flush=True,
    )


def build_rollout_df_from_results_tree() -> pd.DataFrame:
    output_family = OUTPUT_RESULTS_ROOT / OUTPUT_DATASET_FAMILY
    rows: list[dict[str, Any]] = []

    for model in MODEL_LIST:
        for hint in HINT_LEVELS:
            path = output_family / model / f"hint_fraction_{hint:.1f}" / SOLVER_FILE
            if not path.exists():
                continue

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample_obj = json.loads(line)
                    sample_id = str(sample_obj.get("sample_id") or "")
                    if not should_keep_sample(sample_id):
                        continue

                    rollouts = sample_obj.get("rollouts")
                    if not isinstance(rollouts, list):
                        continue

                    for rollout in rollouts:
                        if not isinstance(rollout, dict):
                            continue
                        score_outcomes = rollout.get("score_outcomes")
                        old_outcome = score_outcomes.get(OLD_SCORER_NAME) if isinstance(score_outcomes, dict) else None
                        new_outcome = score_outcomes.get(NEW_SCORER_NAME) if isinstance(score_outcomes, dict) else None

                        rows.append(
                            {
                                "model": model,
                                "hint": float(hint),
                                "sample_id": sample_id,
                                "rollout_id": rollout.get("rollout_id"),
                                "epoch": rollout.get("epoch"),
                                "source_owner": rollout.get("source_owner"),
                                "target": rollout.get("target"),
                                "old_token_len": rollout.get("old_token_len"),
                                "new_token_len": rollout.get("new_token_len"),
                                "was_truncated": int(bool(rollout.get("was_truncated"))),
                                "old_is_correct": (
                                    parse_bool_to_int(old_outcome.get("is_correct"))
                                    if isinstance(old_outcome, dict)
                                    else None
                                ),
                                "new_is_correct": (
                                    parse_bool_to_int(new_outcome.get("is_correct"))
                                    if isinstance(new_outcome, dict)
                                    else None
                                ),
                            }
                        )

    if not rows:
        raise ValueError(f"No rollout rows found under output tree: {output_family}")

    df = pd.DataFrame(rows)
    df["hint"] = pd.to_numeric(df["hint"], errors="coerce")
    df["old_token_len"] = pd.to_numeric(df["old_token_len"], errors="coerce")
    df["new_token_len"] = pd.to_numeric(df["new_token_len"], errors="coerce")
    df["was_truncated"] = pd.to_numeric(df["was_truncated"], errors="coerce").fillna(0).astype(int)
    df["old_is_correct"] = pd.to_numeric(df["old_is_correct"], errors="coerce")
    df["new_is_correct"] = pd.to_numeric(df["new_is_correct"], errors="coerce")
    df = df.dropna(subset=["hint", "old_token_len", "new_token_len"]).copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROLLOUT_CSV, index=False)
    print(f"[{ts_now()}] Wrote rollout CSV from results tree: {ROLLOUT_CSV} rows={len(df)}")
    return df


def build_owner_cache_csv(rollout_df: pd.DataFrame) -> pd.DataFrame:
    df = rollout_df.copy()
    df["owner_norm"] = df["source_owner"].map(normalize_owner)

    grouped = df.groupby(["model", "hint", "owner_norm"], as_index=False).size()
    pivot = grouped.pivot_table(
        index=["model", "hint"],
        columns="owner_norm",
        values="size",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    for c in ["mine", "christine", "unknown"]:
        if c not in pivot.columns:
            pivot[c] = 0

    out_rows: list[dict[str, Any]] = []
    for _, r in pivot.iterrows():
        counts = {
            "mine": int(r["mine"]),
            "christine": int(r["christine"]),
            "unknown": int(r["unknown"]),
        }
        out_rows.append(
            {
                "model": r["model"],
                "hint": float(r["hint"]),
                "owner_label": owner_label_from_counts(counts),
                "owner_counts_json": json.dumps(counts, sort_keys=True),
                "owner_rollouts_total": int(sum(counts.values())),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["model", "hint"]).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OWNER_CSV, index=False)
    print(f"[{ts_now()}] Wrote owner cache CSV: {OWNER_CSV} rows={len(out)}")
    return out


def load_eci_map(path: Path) -> dict[str, float]:
    eci_df = pd.read_csv(path)
    if "model" not in eci_df.columns or "eci_fitted" not in eci_df.columns:
        raise ValueError(f"ECI file missing required columns: {path}")
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def aggregate_accuracy(rollout_df: pd.DataFrame, owner_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rollout_df.groupby(["model", "hint"], as_index=False)
        .agg(
            n_rollouts=("new_is_correct", "size"),
            old_accuracy=("old_is_correct", "mean"),
            new_accuracy=("new_is_correct", "mean"),
            pct_truncated=("was_truncated", "mean"),
            old_token_len_mean=("old_token_len", "mean"),
            old_token_len_max=("old_token_len", "max"),
            new_token_len_mean=("new_token_len", "mean"),
            new_token_len_max=("new_token_len", "max"),
        )
    )
    agg = agg.merge(owner_df[["model", "hint", "owner_label"]], on=["model", "hint"], how="left")
    agg["owner_label"] = agg["owner_label"].fillna("unknown")
    agg = agg.sort_values(["model", "hint"]).reset_index(drop=True)
    agg.to_csv(AGG_CSV, index=False)
    print(f"[{ts_now()}] Wrote aggregate CSV: {AGG_CSV} rows={len(agg)}")
    return agg


def plot_accuracy_vs_hint(agg_df: pd.DataFrame) -> Path:
    eci_map = load_eci_map(ECI_FILE)
    df = agg_df.copy()
    df["eci"] = df["model"].map(eci_map)
    df = df.dropna(subset=["eci"])
    model_eci = df[["model", "eci"]].drop_duplicates().sort_values("eci")
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        mdf = df[df["model"] == model].sort_values("hint")
        eci = float(mdf["eci"].iloc[0])
        for owner_label, owner_color in OWNER_COLORS.items():
            odf = mdf[mdf["owner_label"] == owner_label]
            if odf.empty:
                continue
            ax.scatter(
                odf["hint"],
                odf["new_accuracy"],
                color=owner_color,
                s=38,
                alpha=0.9,
            )
        ax.set_title(f"{model}\\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    present_owners = [k for k in OWNER_COLORS if (df["owner_label"] == k).any()]
    handles = [Line2D([], [], marker="o", linestyle="", color=OWNER_COLORS[k], label=k) for k in present_owners]
    if handles:
        fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(
        f"AIME 2023+2024 / {SOLVER_FILE.removesuffix('.jsonl')}: "
        f"accuracy vs hint by model (truncated to {MAX_TOKENS} tokens, {NEW_SCORER_NAME})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(PLOT_ACCURACY, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts_now()}] Wrote plot: {PLOT_ACCURACY}")
    return PLOT_ACCURACY


def plot_old_length_hist(rollout_df: pd.DataFrame) -> Path:
    eci_map = load_eci_map(ECI_FILE)
    df = rollout_df.copy()
    df["eci"] = df["model"].map(eci_map)
    df["owner_norm"] = df["source_owner"].map(normalize_owner)
    df = df.dropna(subset=["eci"])
    model_eci = df[["model", "eci"]].drop_duplicates().sort_values("eci")
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        mdf = df[df["model"] == model]
        eci = float(mdf["eci"].iloc[0])
        for owner_label in ["mine", "christine", "unknown"]:
            odf = mdf[mdf["owner_norm"] == owner_label]
            if odf.empty:
                continue
            ax.hist(
                odf["old_token_len"],
                bins=40,
                color=OWNER_COLORS[owner_label],
                alpha=0.35,
                edgecolor="none",
            )
        ax.axvline(MAX_TOKENS, color="#888888", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_title(f"{model}\\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("old token length")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"AIME 2023+2024 / {SOLVER_FILE.removesuffix('.jsonl')}: "
        f"old output token length histogram by model (owner-colored overlays)",
        fontsize=12,
    )
    legend_handles = [
        Line2D([], [], marker="s", linestyle="", color=OWNER_COLORS["mine"], label="mine"),
        Line2D([], [], marker="s", linestyle="", color=OWNER_COLORS["christine"], label="christine"),
        Line2D([], [], marker="s", linestyle="", color=OWNER_COLORS["unknown"], label="unknown"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOT_HIST_OLD, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts_now()}] Wrote plot: {PLOT_HIST_OLD}")
    return PLOT_HIST_OLD


async def main() -> None:
    suppress_known_warnings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    await build_results_tree()

    if REUSE_EXISTING_ROLLOUT_CSV and ROLLOUT_CSV.exists():
        print(f"[{ts_now()}] Using existing rollout CSV: {ROLLOUT_CSV}")
        rollout_df = pd.read_csv(ROLLOUT_CSV)
    else:
        rollout_df = build_rollout_df_from_results_tree()

    rollout_df["hint"] = pd.to_numeric(rollout_df["hint"], errors="coerce")
    rollout_df = rollout_df.dropna(subset=["hint"]).copy()
    rollout_df = rollout_df[
        rollout_df["model"].isin(set(MODEL_LIST))
        & rollout_df["hint"].isin(set(float(h) for h in HINT_LEVELS))
    ].copy()

    if REUSE_EXISTING_OWNER_CSV and OWNER_CSV.exists():
        print(f"[{ts_now()}] Using existing owner cache: {OWNER_CSV}")
        owner_df = pd.read_csv(OWNER_CSV)
    else:
        owner_df = build_owner_cache_csv(rollout_df)

    owner_df["hint"] = pd.to_numeric(owner_df["hint"], errors="coerce")
    owner_df = owner_df.dropna(subset=["hint"]).copy()
    owner_df = owner_df[
        owner_df["model"].isin(set(MODEL_LIST))
        & owner_df["hint"].isin(set(float(h) for h in HINT_LEVELS))
    ].copy()

    agg_df = aggregate_accuracy(rollout_df, owner_df)
    plot_accuracy_vs_hint(agg_df)
    plot_old_length_hist(rollout_df)

    print()
    print("=== Done ===")
    print(f"results_root={OUTPUT_RESULTS_ROOT / OUTPUT_DATASET_FAMILY}")
    print(f"rollout_rows={len(rollout_df)}")
    print(f"models={rollout_df['model'].nunique()}")
    print(f"hints={rollout_df['hint'].nunique()}")
    print(
        f"old_token_len_max={int(pd.to_numeric(rollout_df['old_token_len'], errors='coerce').max())} "
        f"new_token_len_max={int(pd.to_numeric(rollout_df['new_token_len'], errors='coerce').max())}"
    )


if __name__ == "__main__":
    # python suze_experiments/20260321/truncate16k_regrade_aime_2023_2024_and_plot.py
    asyncio.run(main())
