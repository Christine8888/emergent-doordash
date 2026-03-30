from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from transformers import AutoTokenizer


# --- Editable constants ---
DATA_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2_regraded")
DATASET_FAMILY = "aime_solution"
SOLVER_FILE = "solution_intext_masked.jsonl"

FAMILY_TOKENIZER_IDS: dict[str, str] = {
    "Llama-3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen3": "Qwen/Qwen3-0.6B",
    "gemma-3": "google/gemma-3-4b-it",
}

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
TOP_K_LONGEST_BY_CHARS = 10

ECI_FILE = Path("christine_experiments/20260129_fitting/eci_model_capabilities.csv")
OUTPUT_DIR = Path("suze_experiments/20260321/plots")
OUT_CSV = "aime_solution_intext_masked_max_tokens_vs_hint_all_models.csv"
OUT_OWNER_CSV = "aime_solution_intext_masked_owner_by_model_hint.csv"
OUT_PNG_MAX_TOKENS = "aime_solution_intext_masked_max_tokens_vs_hint_all_models.png"
OUT_PNG_AVG_CHARS = "aime_solution_intext_masked_avg_chars_vs_hint_all_models.png"
OUT_PNG_MAX_CHARS = "aime_solution_intext_masked_max_chars_vs_hint_all_models.png"
REUSE_EXISTING_CSV = True
# --------------------------

OWNER_COLORS = {
    "mine": "#1f77b4",
    "christine": "#ff7f0e",
    "mixed": "#2ca02c",
    "unknown": "#7f7f7f",
}


_tokenizer_cache: dict[str, AutoTokenizer] = {}


def _tokenizer_hf_id(model_name: str) -> str:
    for prefix, hf_id in FAMILY_TOKENIZER_IDS.items():
        if model_name.startswith(prefix):
            return hf_id
    raise ValueError(
        f"No tokenizer mapping for model '{model_name}'. "
        f"Add its family prefix to FAMILY_TOKENIZER_IDS."
    )


def get_tokenizer(model_name: str) -> AutoTokenizer:
    hf_id = _tokenizer_hf_id(model_name)
    if hf_id not in _tokenizer_cache:
        print(f"Loading tokenizer for {model_name} → {hf_id}")
        _tokenizer_cache[hf_id] = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True,
        )
    return _tokenizer_cache[hf_id]


def load_eci_map(path: Path) -> dict[str, float]:
    eci_df = pd.read_csv(path)
    if "model" not in eci_df.columns or "eci_fitted" not in eci_df.columns:
        raise ValueError(f"ECI file missing required columns: {path}")
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def normalize_owner(owner: Any) -> str:
    token = str(owner or "").strip().lower()
    if token in {"suzeva", "suze", "mine"}:
        return "mine"
    if token in {"christine"}:
        return "christine"
    if token:
        return token
    return "unknown"


def owner_label_from_counts(counts: dict[str, int]) -> str:
    nonzero = [k for k, v in counts.items() if v > 0]
    if not nonzero:
        return "unknown"
    if len(nonzero) == 1:
        return nonzero[0]
    return "mixed"


def inspect_owner_for_file(path: Path) -> dict[str, Any]:
    owner_counts: dict[str, int] = {}
    rollouts = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample_obj = json.loads(line)
            sample_rollouts = sample_obj.get("rollouts")
            if not isinstance(sample_rollouts, list):
                continue
            for rollout in sample_rollouts:
                if not isinstance(rollout, dict):
                    continue
                rollouts += 1
                owner = normalize_owner(rollout.get("source_owner"))
                owner_counts[owner] = owner_counts.get(owner, 0) + 1

    return {
        "owner_label": owner_label_from_counts(owner_counts),
        "owner_counts_json": json.dumps(owner_counts, sort_keys=True),
        "owner_rollouts_total": int(rollouts),
    }


def collect_owner_df() -> pd.DataFrame:
    family_dir = DATA_ROOT / DATASET_FAMILY
    if not family_dir.exists():
        raise FileNotFoundError(f"Missing family directory: {family_dir}")

    rows: list[dict[str, Any]] = []
    for model in MODEL_LIST:
        model_dir = family_dir / model
        if not model_dir.exists():
            print(f"WARNING missing model dir for owner scan: {model_dir}")
            continue
        for hint in HINT_LEVELS:
            path = model_dir / f"hint_fraction_{hint:.1f}" / SOLVER_FILE
            if not path.exists():
                continue
            owner_info = inspect_owner_for_file(path)
            rows.append(
                {
                    "model": model,
                    "hint": float(hint),
                    "owner_label": owner_info["owner_label"],
                    "owner_counts_json": owner_info["owner_counts_json"],
                    "owner_rollouts_total": owner_info["owner_rollouts_total"],
                }
            )
            print(
                f"owner-scan model={model} hint={hint:.1f} owner={owner_info['owner_label']} "
                f"rollouts={owner_info['owner_rollouts_total']}",
                flush=True,
            )

    if not rows:
        raise ValueError("No owner rows collected.")
    return pd.DataFrame(rows)


def inspect_hint_file(path: Path, tokenizer: AutoTokenizer) -> dict[str, Any]:
    rollouts = 0
    missing_output_text = 0
    total_chars = 0
    counter = 0
    top_k: list[tuple[int, int, str, Any, Any, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample_obj = json.loads(line)
            sample_id = sample_obj.get("sample_id")
            sample_rollouts = sample_obj.get("rollouts")
            if not isinstance(sample_rollouts, list):
                continue

            for rollout in sample_rollouts:
                if not isinstance(rollout, dict):
                    continue
                text_obj = rollout.get("output_text")
                if text_obj is None:
                    missing_output_text += 1
                    text = ""
                else:
                    text = str(text_obj)

                char_count = len(text)
                total_chars += char_count
                rollouts += 1

                candidate = (
                    char_count,
                    counter,
                    str(sample_id),
                    rollout.get("epoch"),
                    rollout.get("rollout_id"),
                    text,
                )
                counter += 1

                if len(top_k) < TOP_K_LONGEST_BY_CHARS:
                    heapq.heappush(top_k, candidate)
                elif candidate[0] > top_k[0][0]:
                    heapq.heapreplace(top_k, candidate)

    top_k_sorted = sorted(top_k, key=lambda x: x[0], reverse=True)
    max_tokens = -1
    max_meta: dict[str, Any] = {}
    for char_count, _, sample_id, epoch, rollout_id, text in top_k_sorted:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            max_tokens = token_count
            max_meta = {
                "sample_id": sample_id,
                "epoch": epoch,
                "rollout_id": rollout_id,
                "chars": char_count,
            }

    return {
        "rollouts": rollouts,
        "missing_output_text": missing_output_text,
        "avg_chars": (total_chars / rollouts) if rollouts else 0.0,
        "max_chars": top_k_sorted[0][0] if top_k_sorted else 0,
        "max_tokens": max_tokens,
        "max_meta": max_meta,
    }


def collect_df() -> pd.DataFrame:
    family_dir = DATA_ROOT / DATASET_FAMILY
    if not family_dir.exists():
        raise FileNotFoundError(f"Missing family directory: {family_dir}")

    rows: list[dict[str, Any]] = []
    for model in MODEL_LIST:
        model_dir = family_dir / model
        if not model_dir.exists():
            print(f"WARNING missing model dir: {model_dir}")
            continue

        tokenizer = get_tokenizer(model)

        for hint in HINT_LEVELS:
            path = model_dir / f"hint_fraction_{hint:.1f}" / SOLVER_FILE
            if not path.exists():
                print(f"WARNING missing file: {path}")
                continue

            stats = inspect_hint_file(path, tokenizer)
            max_meta = stats["max_meta"]
            row = {
                "model": model,
                "hint": hint,
                "rollouts": stats["rollouts"],
                "missing_output_text": stats["missing_output_text"],
                "avg_chars": stats["avg_chars"],
                "max_chars": stats["max_chars"],
                "max_tokens": stats["max_tokens"],
                "max_sample_id": max_meta.get("sample_id"),
                "max_epoch": max_meta.get("epoch"),
                "max_rollout_id": max_meta.get("rollout_id"),
            }
            rows.append(row)

            print(
                f"model={model} hint={hint:.1f} rollouts={row['rollouts']} "
                f"max_tokens={row['max_tokens']} max_chars={row['max_chars']} "
                f"max_sample_id={row['max_sample_id']} max_epoch={row['max_epoch']}",
                flush=True,
            )

    if not rows:
        raise ValueError("No rows collected.")
    return pd.DataFrame(rows)


def plot_metric_df(
    df: pd.DataFrame,
    eci_map: dict[str, float],
    *,
    y_col: str,
    y_label: str,
    title_metric: str,
    out_png_name: str,
) -> Path:
    df = df.copy()
    df["hint"] = pd.to_numeric(df["hint"], errors="coerce")
    df["owner_label"] = df.get("owner_label", "unknown").fillna("unknown")
    df["eci"] = df["model"].map(eci_map)
    missing_eci = sorted(set(df.loc[df["eci"].isna(), "model"].tolist()))
    if missing_eci:
        print("WARNING missing ECI for models (dropping):", ", ".join(missing_eci))
        df = df.dropna(subset=["eci"])
    if df.empty:
        raise ValueError("All rows dropped after ECI join.")

    model_eci = df[["model", "eci"]].drop_duplicates().sort_values("eci")
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    color_map = plt.cm.coolwarm
    colors = {m: color_map(i / max(n_models - 1, 1)) for i, m in enumerate(models_sorted)}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        mdf = df[df["model"] == model].sort_values("hint")
        eci = float(mdf["eci"].iloc[0])
        model_y_max = float(mdf[y_col].max()) if not mdf.empty else 1.0
        if model_y_max <= 0:
            model_y_max = 1.0
        for owner_label, owner_color in OWNER_COLORS.items():
            odf = mdf[mdf["owner_label"] == owner_label]
            if odf.empty:
                continue
            ax.scatter(
                odf["hint"],
                odf[y_col],
                color=owner_color,
                s=35,
                alpha=0.9,
            )
        ax.set_title(f"{model}\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if y_col == "max_tokens":
            ax.axhline(16000, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
            if model_y_max * 1.05 >= 32000:
                ax.axhline(32000, color="#cc4444", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, model_y_max * 1.05)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"AIME solution / {SOLVER_FILE.removesuffix('.jsonl')}: "
        f"{title_metric} vs hint by model "
        f"(max over top {TOP_K_LONGEST_BY_CHARS} by chars, per-model tokenizer)",
        fontsize=12,
    )
    present_owners = [k for k in OWNER_COLORS if (df["owner_label"] == k).any()]
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=OWNER_COLORS[k], label=k) for k in present_owners
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / out_png_name
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / OUT_CSV
    out_owner_csv = OUTPUT_DIR / OUT_OWNER_CSV
    if REUSE_EXISTING_CSV and out_csv.exists():
        print(f"Using existing CSV (skipping recompute): {out_csv}")
        df = pd.read_csv(out_csv)
    else:
        df = collect_df()
        df.to_csv(out_csv, index=False)

    if REUSE_EXISTING_CSV and out_owner_csv.exists():
        print(f"Using existing owner CSV (skipping owner scan): {out_owner_csv}")
        owner_df = pd.read_csv(out_owner_csv)
    else:
        owner_df = collect_owner_df()
        owner_df.to_csv(out_owner_csv, index=False)

    df["hint"] = pd.to_numeric(df["hint"], errors="coerce")
    owner_df["hint"] = pd.to_numeric(owner_df["hint"], errors="coerce")
    # Keep cached CSVs intact; enforce MODEL_LIST/HINT_LEVELS only for plotting.
    model_allow = set(MODEL_LIST)
    hint_allow = {float(h) for h in HINT_LEVELS}
    df_plot = df[df["model"].isin(model_allow) & df["hint"].isin(hint_allow)].copy()
    owner_plot = owner_df[
        owner_df["model"].isin(model_allow) & owner_df["hint"].isin(hint_allow)
    ].copy()
    df_plot = df_plot.merge(owner_plot, on=["model", "hint"], how="left")
    df_plot["owner_label"] = df_plot["owner_label"].fillna("unknown")

    eci_map = load_eci_map(ECI_FILE)
    out_png_max_tokens = plot_metric_df(
        df_plot,
        eci_map,
        y_col="max_tokens",
        y_label="max tokens",
        title_metric="max token length",
        out_png_name=OUT_PNG_MAX_TOKENS,
    )
    out_png_avg_chars = plot_metric_df(
        df_plot,
        eci_map,
        y_col="avg_chars",
        y_label="avg chars",
        title_metric="avg char length",
        out_png_name=OUT_PNG_AVG_CHARS,
    )
    out_png_max_chars = plot_metric_df(
        df_plot,
        eci_map,
        y_col="max_chars",
        y_label="max chars",
        title_metric="max char length",
        out_png_name=OUT_PNG_MAX_CHARS,
    )

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote owner CSV: {out_owner_csv}")
    print(f"Wrote plot: {out_png_max_tokens}")
    print(f"Wrote plot: {out_png_avg_chars}")
    print(f"Wrote plot: {out_png_max_chars}")
    print(f"Rows in cached metrics CSV: {len(df)}")
    print(f"Rows plotted after MODEL_LIST/HINT_LEVELS filter: {len(df_plot)}")
    print(f"Models plotted: {df_plot['model'].nunique()}")


if __name__ == "__main__":
    # python suze_experiments/20260321/plot_max_tokens_vs_hint_all_models.py
    main()
