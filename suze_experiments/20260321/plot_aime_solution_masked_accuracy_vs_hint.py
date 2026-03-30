from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Editable constants ---
DATA_ROOT = Path("suze_experiments/20260321/consolidated_hinted_results_v2_regraded")
DATASET_FAMILY = "aime_solution"
SOLVER_FILE = "solution_intext_masked.jsonl"
SCORER_NAME = "aime_scorer_v2"
MODEL_ALLOWLIST = [
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

ECI_FILE = Path("christine_experiments/20260129_fitting/eci_model_capabilities.csv")
OUTPUT_DIR = Path("suze_experiments/20260321/plots")

CONDENSED_CSV = OUTPUT_DIR / "aime_solution_masked_aime_scorer_v2_rollout_scores.csv"
OUT_BOOTSTRAP_CSV = "aime_solution_masked_accuracy_vs_hint_by_model_bootstrap.csv"
OUT_BOOTSTRAP_PNG = "aime_solution_masked_accuracy_vs_hint_by_model_bootstrap.png"

ONLY_HINTS_0_TO_1_BY_01 = True
PROGRESS_EVERY_SAMPLE_LINES = 200

MIN_ROLLOUTS_PER_SAMPLE = 10
N_BOOTSTRAP = 1000
RANDOM_SEED = 0
MORE_THAN_ROLLOUT_THRESHOLD = 10
# --------------------------


CSV_COLUMNS = [
    "model",
    "hint_fraction",
    "sample_id",
    "rollout_id",
    "epoch",
    "score_raw_value",
    "is_correct",
    "extracted_answer",
    "extraction_status",
]


def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_eci_map(eci_file: Path) -> dict[str, float]:
    eci_df = pd.read_csv(eci_file)
    if "model" not in eci_df.columns or "eci_fitted" not in eci_df.columns:
        raise ValueError(f"ECI file missing required columns: {eci_file}")
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def parse_hint_from_dirname(dirname: str) -> float:
    prefix = "hint_fraction_"
    if not dirname.startswith(prefix):
        raise ValueError(f"Unexpected hint dir name: {dirname}")
    return float(dirname[len(prefix) :])


def should_keep_hint(hint_value: float) -> bool:
    if not ONLY_HINTS_0_TO_1_BY_01:
        return True
    return hint_value in {round(i * 0.1, 1) for i in range(11)}


def normalize_is_correct(value: Any) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return ""


def parse_is_correct_column(series: pd.Series) -> pd.Series:
    def parse_one(v: object) -> float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            iv = int(v)
            if iv == 1:
                return 1.0
            if iv == 0:
                return 0.0
        s = str(v).strip().lower()
        if s in {"1", "true"}:
            return 1.0
        if s in {"0", "false"}:
            return 0.0
        return np.nan

    return series.map(parse_one)


def build_condensed_scores_csv(
    *,
    data_root: Path,
    dataset_family: str,
    solver_file: str,
    scorer_name: str,
    output_csv: Path,
    model_allowlist: list[str],
) -> None:
    family_dir = data_root / dataset_family
    if not family_dir.exists():
        raise FileNotFoundError(f"Missing family directory: {family_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted([p for p in family_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not model_dirs:
        raise ValueError(f"No model directories under: {family_dir}")
    if not model_allowlist:
        raise ValueError("MODEL_ALLOWLIST must be a non-empty list.")
    allow = set(model_allowlist)
    model_dirs = [p for p in model_dirs if p.name in allow]
    if not model_dirs:
        raise ValueError("MODEL_ALLOWLIST provided but no matching model directories were found.")

    files_scanned = 0
    sample_lines_scanned = 0
    rows_written = 0
    missing_scorer_rows = 0
    bad_json_lines = 0

    print(f"[{ts_now()}] Building condensed score CSV: {output_csv}")

    with output_csv.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for model_dir in model_dirs:
            model_name = model_dir.name
            hint_dirs = sorted(
                [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("hint_fraction_")],
                key=lambda p: p.name,
            )
            for hint_dir in hint_dirs:
                hint_value = parse_hint_from_dirname(hint_dir.name)
                if not should_keep_hint(hint_value):
                    continue

                file_path = hint_dir / solver_file
                if not file_path.exists():
                    continue

                files_scanned += 1
                print(f"[{ts_now()}] Scanning {file_path}")

                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample_obj = json.loads(line)
                        except json.JSONDecodeError:
                            bad_json_lines += 1
                            continue

                        sample_lines_scanned += 1
                        sample_id = sample_obj.get("sample_id")
                        rollouts = sample_obj.get("rollouts")
                        if sample_id is None or not isinstance(rollouts, list):
                            continue

                        for rollout in rollouts:
                            if not isinstance(rollout, dict):
                                continue
                            score_outcomes = rollout.get("score_outcomes")
                            if not isinstance(score_outcomes, dict):
                                missing_scorer_rows += 1
                                continue
                            scorer_payload = score_outcomes.get(scorer_name)
                            if not isinstance(scorer_payload, dict):
                                missing_scorer_rows += 1
                                continue

                            writer.writerow(
                                {
                                    "model": model_name,
                                    "hint_fraction": hint_value,
                                    "sample_id": sample_id,
                                    "rollout_id": rollout.get("rollout_id", ""),
                                    "epoch": rollout.get("epoch", ""),
                                    "score_raw_value": scorer_payload.get("score_raw_value", ""),
                                    "is_correct": normalize_is_correct(scorer_payload.get("is_correct")),
                                    "extracted_answer": scorer_payload.get("extracted_answer", ""),
                                    "extraction_status": scorer_payload.get("extraction_status", ""),
                                }
                            )
                            rows_written += 1

                        # if sample_lines_scanned % PROGRESS_EVERY_SAMPLE_LINES == 0:
                        #     print(
                        #         f"[{ts_now()}] sample_lines={sample_lines_scanned:,} "
                        #         f"rows_written={rows_written:,}"
                        #     )

    print(
        f"[{ts_now()}] Condensed CSV done | files={files_scanned} "
        f"sample_lines={sample_lines_scanned:,} rows={rows_written:,} "
        f"missing_scorer_rows={missing_scorer_rows:,} bad_json={bad_json_lines:,}"
    )


def load_condensed_scores_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing condensed score CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"model", "hint_fraction", "sample_id", "is_correct"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns {required}: {csv_path}")

    df = df.copy()
    df["hint_fraction"] = pd.to_numeric(df["hint_fraction"], errors="coerce")
    df["is_correct_num"] = parse_is_correct_column(df["is_correct"])
    df = df.dropna(subset=["hint_fraction", "is_correct_num"])
    df = df[df["hint_fraction"].map(should_keep_hint)]

    if df.empty:
        raise ValueError("No usable score rows in condensed CSV after filtering.")
    return df


def filter_models(df: pd.DataFrame, model_allowlist: list[str]) -> pd.DataFrame:
    if not model_allowlist:
        raise ValueError("MODEL_ALLOWLIST must be a non-empty list.")
    allow = set(model_allowlist)
    out = df[df["model"].isin(allow)].copy()
    if out.empty:
        raise ValueError("MODEL_ALLOWLIST filtered out all rows.")
    missing = sorted(allow - set(df["model"].unique().tolist()))
    if missing:
        print("WARNING requested models not found in data:", ", ".join(missing))
    return out


def print_samples_with_more_than_threshold_rollouts(
    *, rollout_df: pd.DataFrame, threshold: int
) -> None:
    sample_counts = (
        rollout_df.groupby(["model", "hint_fraction", "sample_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_rollouts"})
    )
    combo_counts = (
        sample_counts.assign(is_gt=lambda d: d["n_rollouts"] > threshold)
        .groupby(["model", "hint_fraction"], as_index=False)["is_gt"]
        .sum()
        .rename(columns={"is_gt": f"n_samples_gt_{threshold}_rollouts"})
        .sort_values(
            [f"n_samples_gt_{threshold}_rollouts", "model", "hint_fraction"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    print()
    print(
        f"=== sample_id count with >{threshold} rollouts "
        f"(ordered most->least, by model+hint) ==="
    )
    for _, row in combo_counts.iterrows():
        print(
            f"model={row['model']} hint={float(row['hint_fraction']):.1f} "
            f"n_samples_gt_{threshold}_rollouts={int(row[f'n_samples_gt_{threshold}_rollouts'])}"
        )


def collect_bootstrap_accuracy_df(*, rollout_df: pd.DataFrame, eci_file: Path) -> pd.DataFrame:
    eci_map = load_eci_map(eci_file)
    rng = np.random.default_rng(RANDOM_SEED)

    rows: list[dict[str, Any]] = []

    grouped = rollout_df.groupby(["model", "hint_fraction"], sort=True)
    for (model, hint), group_df in grouped:
        sample_groups = group_df.groupby("sample_id")

        sample_rollout_scores: list[np.ndarray] = []
        n_samples_total = 0
        n_rollouts_eligible = 0
        n_short_rollout_samples = 0

        for sample_id, sample_df in sample_groups:
            n_samples_total += 1
            vals = sample_df["is_correct_num"].to_numpy(dtype=float)
            if vals.size < MIN_ROLLOUTS_PER_SAMPLE:
                n_short_rollout_samples += 1
                print(
                    f"WARNING short sample rollouts "
                    f"model={model} hint={float(hint):.1f} sample_id={sample_id} "
                    f"num_rollouts={vals.size} expected>={MIN_ROLLOUTS_PER_SAMPLE}"
                )
            sample_rollout_scores.append(vals)
            n_rollouts_eligible += int(vals.size)

        if not sample_rollout_scores:
            continue

        n_samples_eligible = len(sample_rollout_scores)

        # Point estimate: average sample-level rollout accuracy.
        point_accuracy = float(np.mean([arr.mean() for arr in sample_rollout_scores]))

        # Bootstrap definition requested:
        # each iteration draws one random rollout score per sample_id, then averages.
        boot_sums = np.zeros(N_BOOTSTRAP, dtype=float)
        for arr in sample_rollout_scores:
            draw_idx = rng.integers(low=0, high=arr.size, size=N_BOOTSTRAP)
            boot_sums += arr[draw_idx]
        boot_means = boot_sums / float(n_samples_eligible)
        ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])

        rows.append(
            {
                "model": model,
                "hint": float(hint),
                "accuracy": point_accuracy,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_samples_total": n_samples_total,
                "n_samples_eligible": n_samples_eligible,
                "n_rollouts_eligible": n_rollouts_eligible,
                "n_short_rollout_samples": n_short_rollout_samples,
                "eci": eci_map.get(model),
            }
        )

    if not rows:
        raise ValueError("No usable (model, hint) groups found for bootstrap.")

    agg = pd.DataFrame(rows)

    missing_eci = sorted(set(agg.loc[agg["eci"].isna(), "model"].tolist()))
    if missing_eci:
        print("WARNING missing ECI for models (dropping):", ", ".join(missing_eci))
        agg = agg.dropna(subset=["eci"])

    if agg.empty:
        raise ValueError("All rows dropped after ECI join.")

    return agg.sort_values(["eci", "model", "hint"]).reset_index(drop=True)


def plot_accuracy_vs_hint_by_model_bootstrap(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str,
    title: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_eci = df[["model", "eci"]].drop_duplicates().sort_values("eci")
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    model_cmap = plt.cm.coolwarm
    model_colors = {m: model_cmap(i / max(n_models - 1, 1)) for i, m in enumerate(models_sorted)}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        eci = float(model_df["eci"].iloc[0])

        x = model_df["hint"].to_numpy(dtype=float)
        y = model_df["accuracy"].to_numpy(dtype=float)
        low = model_df["ci_low"].to_numpy(dtype=float)
        high = model_df["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - low, high - y])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=model_colors[model],
            alpha=0.9,
            linewidth=1.5,
            markersize=4.5,
            capsize=2.0,
        )
        ax.set_title(f"{model}\\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    if not CONDENSED_CSV.exists():
        build_condensed_scores_csv(
            data_root=DATA_ROOT,
            dataset_family=DATASET_FAMILY,
            solver_file=SOLVER_FILE,
            scorer_name=SCORER_NAME,
            output_csv=CONDENSED_CSV,
            model_allowlist=MODEL_ALLOWLIST,
        )
    else:
        print(f"[{ts_now()}] Using existing condensed CSV: {CONDENSED_CSV}")

    rollout_df = load_condensed_scores_df(CONDENSED_CSV)
    rollout_df = filter_models(rollout_df, MODEL_ALLOWLIST)
    print_samples_with_more_than_threshold_rollouts(
        rollout_df=rollout_df, threshold=MORE_THAN_ROLLOUT_THRESHOLD
    )

    agg_df = collect_bootstrap_accuracy_df(
        rollout_df=rollout_df,
        eci_file=ECI_FILE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / OUT_BOOTSTRAP_CSV
    agg_df.to_csv(out_csv, index=False)

    out_plot = plot_accuracy_vs_hint_by_model_bootstrap(
        df=agg_df,
        output_dir=OUTPUT_DIR,
        out_name=OUT_BOOTSTRAP_PNG,
        title=(
            f"AIME solution / {SOLVER_FILE.removesuffix('.jsonl')}: "
            f"accuracy vs hint by model (bootstrap CI, {SCORER_NAME})"
        ),
    )

    print(f"Wrote bootstrap CSV: {out_csv}")
    print(f"Wrote bootstrap plot: {out_plot}")
    print(f"Models plotted: {agg_df['model'].nunique()}")
    print(f"Rows (model,hint): {len(agg_df)}")


if __name__ == "__main__":
    # python suze_experiments/20260321/plot_aime_solution_masked_accuracy_vs_hint.py
    main()
