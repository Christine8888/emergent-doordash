"""Sanity-check plots for AIME runs.

This script makes a raw accuracy-vs-ECI plot by hint level:
- scatter points only (no scaling-law fit),
- one color per hint level,
- top x-axis labeling which model(s) sit at each ECI,
- strict validation that each (hint, model) appears at most once.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import duckdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_eci_map(eci_file: Path) -> dict[str, float]:
    eci_df = pd.read_csv(eci_file)
    return dict(zip(eci_df["model"], eci_df["eci_fitted"]))


def _model_labels_by_eci(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    model_eci = df[["model", "eci"]].drop_duplicates()
    grouped = (
        model_eci.groupby("eci", as_index=False)["model"]
        .agg(lambda s: ", ".join(sorted(s.tolist())))
        .sort_values("eci")
    )
    ticks = [float(v) for v in grouped["eci"].tolist()]
    labels = grouped["model"].tolist()
    return ticks, labels


def load_results_df(
    *,
    base_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
) -> pd.DataFrame:
    # Import here so this script can run from repo root without install.
    from src.modelx import load_results

    df = load_results(
        base_folder=str(base_folder),
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )
    if df.empty:
        raise ValueError("No results found for the provided eval/solver/condition.")

    if all_models is not None:
        df = df[df["model"].isin(all_models)]
    if hint_fractions is not None:
        df = df[df["hint"].isin(hint_fractions)]

    eci_map = _load_eci_map(eci_file)
    df = df.copy()
    df["eci"] = df["model"].map(eci_map)
    df = df.dropna(subset=["eci", "accuracy", "hint"])

    if df.empty:
        raise ValueError("No rows remain after model/hint/ECI filtering.")

    return df


def _canonical_model_name(model: str | None, model_path: str | None) -> str:
    raw = (model_path or model or "").strip()
    if raw.startswith("vllm/"):
        raw = raw.split("/", 1)[1]
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw


def _slugify(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def load_regraded_results_df(
    *,
    db_path: Path,
    eci_file: Path,
    scorer_name: str,
    run_type: str,
    benchmark: str,
    path_hint_level: str,
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
    expected_epochs: int = 10,
    n_bootstrap: int = 1000,
    strict_epochs: bool = True,
) -> pd.DataFrame:
    from src.utils.inspect_utils import compute_bootstrap_over_epochs_from_correctness

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB cache not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df_rollout = conn.execute(
            """
            SELECT
                r.rollout_id,
                r.model,
                r.model_path,
                r.hint_fraction AS hint,
                r.sample_id,
                r.epoch,
                r.created,
                rs.score_normalized
            FROM rollouts r
            INNER JOIN rollout_scorers rs
                ON rs.rollout_id = r.rollout_id
            WHERE r.run_type = ?
              AND r.benchmark = ?
              AND r.path_hint_level = ?
              AND rs.scorer_name = ?
              AND r.hint_fraction IS NOT NULL
              AND r.sample_id IS NOT NULL
              AND r.epoch IS NOT NULL
            """,
            [run_type, benchmark, path_hint_level, scorer_name],
        ).df()
    finally:
        conn.close()

    if df_rollout.empty:
        raise ValueError(
            "No rows found for the requested scorer/filter in DuckDB cache. "
            f"scorer={scorer_name} run_type={run_type} benchmark={benchmark} "
            f"path_hint_level={path_hint_level}"
        )

    df_rollout = df_rollout.copy()
    df_rollout["model"] = df_rollout.apply(
        lambda r: _canonical_model_name(
            str(r["model"]) if pd.notna(r["model"]) else None,
            str(r["model_path"]) if pd.notna(r["model_path"]) else None,
        ),
        axis=1,
    )
    df_rollout["hint"] = df_rollout["hint"].astype(float).round(2)
    df_rollout["epoch"] = df_rollout["epoch"].astype(int)
    df_rollout["is_correct"] = (df_rollout["score_normalized"] == "C").astype(int)

    # Multiple eval runs can produce duplicate rows for (model, hint, sample_id, epoch).
    # Keep the most recent row by created timestamp, then rollout_id as tie-break.
    before = len(df_rollout)
    df_rollout = df_rollout.sort_values(["created", "rollout_id"])
    df_rollout = df_rollout.drop_duplicates(
        subset=["model", "hint", "sample_id", "epoch"],
        keep="last",
    )
    dropped = before - len(df_rollout)
    if dropped > 0:
        print(
            f"Dropped {dropped:,} duplicate rollout rows by (model,hint,sample_id,epoch) "
            "keeping latest created entry."
        )

    if all_models is not None:
        df_rollout = df_rollout[df_rollout["model"].isin(all_models)]
    if hint_fractions is not None:
        allowed = {round(float(h), 2) for h in hint_fractions}
        df_rollout = df_rollout[df_rollout["hint"].isin(allowed)]

    if df_rollout.empty:
        raise ValueError("No rows remain after model/hint filtering for regraded results.")

    records: list[dict[str, float | str | int]] = []
    bad_epoch_examples: list[dict[str, str | float | int]] = []

    for (model, hint), group in df_rollout.groupby(["model", "hint"], sort=True):
        per_sample_epoch_correct: dict[str, list[int]] = {}

        for sample_id, sample_group in group.groupby("sample_id", sort=False):
            sample_group = sample_group.sort_values("epoch")
            epoch_to_correct = {
                int(ep): int(corr)
                for ep, corr in zip(sample_group["epoch"].tolist(), sample_group["is_correct"].tolist())
            }
            expected = set(range(1, expected_epochs + 1))
            seen = set(epoch_to_correct.keys())
            if seen != expected:
                bad_epoch_examples.append(
                    {
                        "model": str(model),
                        "hint": float(hint),
                        "sample_id": str(sample_id),
                        "n_epochs_seen": len(seen),
                    }
                )
                if strict_epochs:
                    continue
                # Skip malformed samples in non-strict mode.
                continue
            per_sample_epoch_correct[str(sample_id)] = [epoch_to_correct[e] for e in range(1, expected_epochs + 1)]

        if strict_epochs and bad_epoch_examples:
            example_str = bad_epoch_examples[:10]
            raise ValueError(
                "Expected exactly 10 epochs per sample for regraded bootstrap, but found mismatches. "
                f"First examples: {example_str}"
            )

        if not per_sample_epoch_correct:
            continue

        bs = compute_bootstrap_over_epochs_from_correctness(
            per_sample_epoch_correct,
            n_bootstrap=n_bootstrap,
        )
        records.append(
            {
                "model": str(model),
                "hint": float(hint),
                "accuracy": float(bs["accuracy"]),
                "stderr": float(bs["stderr"]),
                "n_samples": int(len(per_sample_epoch_correct)),
                "epochs": int(bs["epochs"]),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No aggregated model+hint rows produced from regraded data.")

    eci_map = _load_eci_map(eci_file)
    df["eci"] = df["model"].map(eci_map)
    df = df.dropna(subset=["eci", "accuracy", "hint"])
    if df.empty:
        raise ValueError("No rows remain after ECI join for regraded data.")

    return df


def load_regraded_results_df_from_enriched_sidecar(
    *,
    sidecar_file: Path,
    eci_file: Path,
    scorer_name: str,
    run_type: str,
    benchmark: str,
    path_hint_level: str,
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
    expected_epochs: int = 10,
    n_bootstrap: int = 1000,
    strict_epochs: bool = True,
) -> pd.DataFrame:
    from src.utils.inspect_utils import compute_bootstrap_over_epochs_from_correctness

    if not sidecar_file.exists():
        raise FileNotFoundError(f"Enriched sidecar not found: {sidecar_file}")

    allowed_models = set(all_models) if all_models is not None else None
    allowed_hints = (
        {round(float(h), 2) for h in hint_fractions}
        if hint_fractions is not None
        else None
    )

    # (model, hint) -> sample_id -> (scores_by_epoch, created_by_epoch)
    grouped: dict[tuple[str, float], dict[str, tuple[list[int | None], list[str]]]] = {}
    bad_json_lines = 0
    filtered_out = 0
    kept_rows = 0
    bad_rows = 0

    print(f"amount of lines in sidecar: {sidecar_file.stat().st_size}")

    with sidecar_file.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad_json_lines += 1
                if bad_json_lines <= 5:
                    print(
                        f"WARN bad JSON in enriched sidecar line={line_num:,}",
                        flush=True,
                    )
                continue

            if str(row.get("scorer_name") or "") != scorer_name:
                filtered_out += 1
                continue
            if str(row.get("run_type") or "") != run_type:
                filtered_out += 1
                continue
            if str(row.get("benchmark") or "") != benchmark:
                filtered_out += 1
                continue
            if str(row.get("path_hint_level") or "") != path_hint_level:
                filtered_out += 1
                continue

            model = _canonical_model_name(
                str(row.get("model")) if row.get("model") is not None else None,
                str(row.get("model_path")) if row.get("model_path") is not None else None,
            )
            if not model:
                bad_rows += 1
                continue
            if allowed_models is not None and model not in allowed_models:
                filtered_out += 1
                continue

            try:
                hint = round(float(row.get("hint_fraction")), 2)
            except (TypeError, ValueError):
                bad_rows += 1
                continue
            if allowed_hints is not None and hint not in allowed_hints:
                filtered_out += 1
                continue

            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                bad_rows += 1
                continue
            try:
                epoch = int(row.get("epoch"))
            except (TypeError, ValueError):
                bad_rows += 1
                continue
            if epoch < 1 or epoch > expected_epochs:
                bad_rows += 1
                continue

            score_label = str(row.get("score_normalized") or "").strip().upper()
            if score_label == "C":
                score = 1
            elif score_label == "I":
                score = 0
            else:
                score = 1 if bool(row.get("is_correct")) else 0

            created = str(row.get("created") or "")
            combo = (model, hint)
            sample_map = grouped.setdefault(combo, {})
            entry = sample_map.get(sample_id)
            if entry is None:
                entry = ([None] * expected_epochs, [""] * expected_epochs)
                sample_map[sample_id] = entry

            scores_by_epoch, created_by_epoch = entry
            idx = epoch - 1
            # Keep most recent entry for duplicate rows.
            if created >= created_by_epoch[idx]:
                scores_by_epoch[idx] = score
                created_by_epoch[idx] = created

            kept_rows += 1

    if kept_rows == 0:
        raise ValueError(
            "No matching rows found in enriched sidecar after filtering. "
            f"scorer={scorer_name} run_type={run_type} benchmark={benchmark} "
            f"path_hint_level={path_hint_level}"
        )

    bad_samples: list[dict[str, str | float | int]] = []
    records: list[dict[str, float | str | int]] = []
    for (model, hint), sample_map in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        per_sample_epoch_correct: dict[str, list[int]] = {}
        for sample_id, (scores_by_epoch, _created_by_epoch) in sample_map.items():
            if any(v is None for v in scores_by_epoch):
                bad_samples.append(
                    {
                        "model": model,
                        "hint": hint,
                        "sample_id": sample_id,
                        "n_epochs_seen": sum(v is not None for v in scores_by_epoch),
                    }
                )
                continue
            per_sample_epoch_correct[sample_id] = [int(v) for v in scores_by_epoch if v is not None]

        if not per_sample_epoch_correct:
            continue

        bs = compute_bootstrap_over_epochs_from_correctness(
            per_sample_epoch_correct,
            n_bootstrap=n_bootstrap,
        )
        records.append(
            {
                "model": model,
                "hint": float(hint),
                "accuracy": float(bs["accuracy"]),
                "stderr": float(bs["stderr"]),
                "n_samples": int(len(per_sample_epoch_correct)),
                "epochs": int(bs["epochs"]),
            }
        )

    if bad_samples:
        strict_note = " strict_epochs=True;" if strict_epochs else ""
        print(
            f"WARN{strict_note} expected exactly {expected_epochs} epochs per sample, "
            "but enriched sidecar had incomplete samples. "
            f"n_incomplete={len(bad_samples):,} examples={bad_samples[:10]}",
            flush=True,
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No aggregated rows produced from enriched sidecar.")

    eci_map = _load_eci_map(eci_file)
    df["eci"] = df["model"].map(eci_map)
    df = df.dropna(subset=["eci", "accuracy", "hint"])
    if df.empty:
        raise ValueError("No rows remain after ECI join for enriched sidecar data.")

    print(
        "Loaded regraded enriched sidecar: "
        f"rows_kept={kept_rows:,} filtered_out={filtered_out:,} "
        f"bad_json={bad_json_lines:,} bad_rows={bad_rows:,} "
        f"incomplete_samples={len(bad_samples):,}",
        flush=True,
    )
    return df


def _combos_with_local_eval(
    *,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
) -> set[tuple[str, float]]:
    eval_root = base_folder / eval_name / solver / condition
    combos: set[tuple[str, float]] = set()
    if not eval_root.exists():
        return combos

    for model_dir in eval_root.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for eval_file in model_dir.glob("*.eval"):
            try:
                with ZipFile(eval_file) as zf:
                    start = json.loads(zf.read("_journal/start.json"))
            except (BadZipFile, KeyError, ValueError, OSError):
                continue

            eval_info = start.get("eval", {})
            hint = eval_info.get("metadata", {}).get("hint_fraction")
            if hint is None:
                continue
            combos.add((model_name, round(float(hint), 2)))
    return combos


def add_inference_owner(
    *,
    df: pd.DataFrame,
    base_folder: Path,
    eval_name: str,
    solver: str,
    condition: str,
) -> pd.DataFrame:
    local_eval_combos = _combos_with_local_eval(
        base_folder=base_folder,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
    )
    out = df.copy()
    out["inference_owner"] = out.apply(
        lambda r: (
            "mine"
            if (r["model"], round(float(r["hint"]), 2)) in local_eval_combos
            else "christine"
        ),
        axis=1,
    )
    return out


def assert_unique_model_hint(df: pd.DataFrame, *, output_dir: Path | None = None) -> None:
    counts = (
        df.groupby(["hint", "model"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    dup = counts[counts["count"] > 1].sort_values(["hint", "model"])
    if dup.empty:
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dup.to_csv(output_dir / "duplicate_model_hint_counts.csv", index=False)

    sample = dup.head(10).to_dict(orient="records")
    raise ValueError(
        "Found duplicate points for the same (hint, model). "
        f"Rows with count>1: {len(dup)}. Sample: {sample}"
    )


def plot_accuracy_vs_eci_raw_by_hint(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str = "accuracy_vs_eci_by_hint_raw_points.png",
    title: str = "AIME: accuracy vs ECI by hint (raw points only)",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    hints = sorted(df["hint"].unique())
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hints) - 1, 1)) for i, h in enumerate(hints)}

    for hint in hints:
        hint_df = df[df["hint"] == hint].sort_values("eci")
        ax.scatter(
            hint_df["eci"],
            hint_df["accuracy"],
            color=colors[hint],
            label=f"h={hint:.2f}",
            alpha=0.85,
            s=65,
        )

    ax.set_xlabel("ECI")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    ticks, labels = _model_labels_by_eci(df)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels(labels, rotation=75, ha="left", fontsize=8)
    ax_top.set_xlabel("model(s) at each ECI")

    plt.tight_layout()
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_vs_hint_by_model_raw_points(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str = "accuracy_vs_hint_by_model.png",
    title: str = "AIME: accuracy vs hint by model (raw points only)",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_eci = (
        df[["model", "eci"]]
        .drop_duplicates()
        .sort_values("eci")
    )
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

        ax.scatter(
            model_df["hint"],
            model_df["accuracy"],
            color=model_colors[model],
            alpha=0.85,
            s=45,
        )
        ax.set_title(f"{model}\neci={eci:.2f}", fontsize=8)
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


def plot_accuracy_vs_hint_by_model_raw_points_by_owner(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    out_name: str = "accuracy_vs_hint_by_model_owner_debug.png",
    title: str = "AIME: accuracy vs hint by model (owner-colored debug)",
) -> Path:
    if "inference_owner" not in df.columns:
        raise ValueError("Missing 'inference_owner' column. Call add_inference_owner(...) first.")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_eci = (
        df[["model", "eci"]]
        .drop_duplicates()
        .sort_values("eci")
    )
    models_sorted = model_eci["model"].tolist()
    n_models = len(models_sorted)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    owner_colors = {"mine": "#1f77b4", "christine": "#ff7f0e"}

    for i, model in enumerate(models_sorted):
        ax = axes[i]
        model_df = df[df["model"] == model].sort_values("hint")
        eci = float(model_df["eci"].iloc[0])

        for owner in ["mine", "christine"]:
            owner_df = model_df[model_df["inference_owner"] == owner]
            if owner_df.empty:
                continue
            ax.scatter(
                owner_df["hint"],
                owner_df["accuracy"],
                color=owner_colors[owner],
                alpha=0.85,
                s=45,
            )
        ax.set_title(f"{model}\neci={eci:.2f}", fontsize=8)
        ax.set_xlabel("hint")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=owner_colors["mine"], label="mine"),
        Line2D([], [], marker="o", linestyle="", color=owner_colors["christine"], label="christine"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_aime_raw_sanity_plot(
    *,
    base_folder: Path,
    eci_file: Path,
    eval_name: str,
    solver: str,
    condition: str,
    output_dir: Path,
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
) -> tuple[Path, Path]:
    df = load_results_df(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name=eval_name,
        solver=solver,
        condition=condition,
        all_models=all_models,
        hint_fractions=hint_fractions,
    )
    assert_unique_model_hint(df, output_dir=output_dir)

    summary = (
        df.groupby(["hint"], as_index=False)
        .agg(n_models=("model", "nunique"), mean_accuracy=("accuracy", "mean"))
        .sort_values("hint")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "raw_hint_summary.csv", index=False)

    # Detailed table: one row per model, one column per hint fraction.
    # This gives a direct sanity-check view of accuracy per model per hint level.
    per_model_hint = (
        df.pivot(index="model", columns="hint", values="accuracy")
        .sort_index()
        .sort_index(axis=1)
    )
    per_model_hint.columns = [f"hint_{float(c):.2f}" for c in per_model_hint.columns]
    per_model_hint.to_csv(output_dir / "accuracy_per_model_per_hint.csv")

    # Long-form detailed table for easy filtering/joins.
    detailed_long = df[["model", "hint", "eci", "accuracy"]].sort_values(["model", "hint"])
    detailed_long.to_csv(output_dir / "accuracy_per_model_per_hint_long.csv", index=False)

    out_eci = plot_accuracy_vs_eci_raw_by_hint(df=df, output_dir=output_dir)
    out_by_model = plot_accuracy_vs_hint_by_model_raw_points(df=df, output_dir=output_dir)
    return out_eci, out_by_model


def run_aime_raw_sanity_plot_from_regraded_cache(
    *,
    db_path: Path,
    eci_file: Path,
    output_dir: Path,
    scorer_name: str = "aime_scorer_extract_answer_fixed",
    run_type: str = "results",
    benchmark: str = "aime",
    path_hint_level: str = "solution_intext_masked/0shot",
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
    expected_epochs: int = 10,
    n_bootstrap: int = 1000,
) -> tuple[Path, Path]:
    df = load_regraded_results_df(
        db_path=db_path,
        eci_file=eci_file,
        scorer_name=scorer_name,
        run_type=run_type,
        benchmark=benchmark,
        path_hint_level=path_hint_level,
        all_models=all_models,
        hint_fractions=hint_fractions,
        expected_epochs=expected_epochs,
        n_bootstrap=n_bootstrap,
        strict_epochs=True,
    )
    assert_unique_model_hint(df, output_dir=output_dir)

    slug = _slugify(scorer_name)
    summary = (
        df.groupby(["hint"], as_index=False)
        .agg(n_models=("model", "nunique"), mean_accuracy=("accuracy", "mean"))
        .sort_values("hint")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"raw_hint_summary_{slug}.csv", index=False)

    per_model_hint = (
        df.pivot(index="model", columns="hint", values="accuracy")
        .sort_index()
        .sort_index(axis=1)
    )
    per_model_hint.columns = [f"hint_{float(c):.2f}" for c in per_model_hint.columns]
    per_model_hint.to_csv(output_dir / f"accuracy_per_model_per_hint_{slug}.csv")

    detailed_long = df[["model", "hint", "eci", "accuracy", "stderr", "n_samples", "epochs"]].sort_values(
        ["model", "hint"]
    )
    detailed_long.to_csv(output_dir / f"accuracy_per_model_per_hint_long_{slug}.csv", index=False)

    out_eci = plot_accuracy_vs_eci_raw_by_hint(
        df=df,
        output_dir=output_dir,
        out_name=f"accuracy_vs_eci_by_hint_raw_points_{slug}.png",
        title=f"AIME: accuracy vs ECI by hint ({scorer_name})",
    )
    out_by_model = plot_accuracy_vs_hint_by_model_raw_points(
        df=df,
        output_dir=output_dir,
        out_name=f"accuracy_vs_hint_by_model_{slug}.png",
        title=f"AIME: accuracy vs hint by model ({scorer_name})",
    )
    return out_eci, out_by_model


def run_aime_raw_sanity_plot_from_regraded_enriched_sidecar(
    *,
    sidecar_file: Path,
    eci_file: Path,
    output_dir: Path,
    scorer_name: str = "aime_scorer_extract_answer_fixed",
    run_type: str = "results",
    benchmark: str = "aime",
    path_hint_level: str = "solution_intext_masked/0shot",
    all_models: list[str] | None = None,
    hint_fractions: list[float] | None = None,
    expected_epochs: int = 10,
    n_bootstrap: int = 1000,
) -> tuple[Path, Path]:
    df = load_regraded_results_df_from_enriched_sidecar(
        sidecar_file=sidecar_file,
        eci_file=eci_file,
        scorer_name=scorer_name,
        run_type=run_type,
        benchmark=benchmark,
        path_hint_level=path_hint_level,
        all_models=all_models,
        hint_fractions=hint_fractions,
        expected_epochs=expected_epochs,
        n_bootstrap=n_bootstrap,
        strict_epochs=True,
    )
    print('loaded results from sidecar')
    assert_unique_model_hint(df, output_dir=output_dir)

    slug = _slugify(scorer_name)
    summary = (
        df.groupby(["hint"], as_index=False)
        .agg(n_models=("model", "nunique"), mean_accuracy=("accuracy", "mean"))
        .sort_values("hint")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"raw_hint_summary_{slug}.csv", index=False)

    per_model_hint = (
        df.pivot(index="model", columns="hint", values="accuracy")
        .sort_index()
        .sort_index(axis=1)
    )
    per_model_hint.columns = [f"hint_{float(c):.2f}" for c in per_model_hint.columns]
    per_model_hint.to_csv(output_dir / f"accuracy_per_model_per_hint_{slug}.csv")

    detailed_long = df[["model", "hint", "eci", "accuracy", "stderr", "n_samples", "epochs"]].sort_values(
        ["model", "hint"]
    )
    detailed_long.to_csv(output_dir / f"accuracy_per_model_per_hint_long_{slug}.csv", index=False)

    out_eci = plot_accuracy_vs_eci_raw_by_hint(
        df=df,
        output_dir=output_dir,
        out_name=f"accuracy_vs_eci_by_hint_raw_points_{slug}.png",
        title=f"AIME: accuracy vs ECI by hint ({scorer_name})",
    )
    out_by_model = plot_accuracy_vs_hint_by_model_raw_points(
        df=df,
        output_dir=output_dir,
        out_name=f"accuracy_vs_hint_by_model_{slug}.png",
        title=f"AIME: accuracy vs hint by model ({scorer_name})",
    )
    return out_eci, out_by_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base_folder = project_root / "christine_experiments/20251113/results"
    eci_file = project_root / "christine_experiments/20260129_fitting/eci_model_capabilities.csv"
    output_dir = Path(__file__).resolve().parent
    regraded_db = project_root / "suze_experiments/20260313/consolidated_jsonl/_viewer_cache.duckdb"
    regraded_enriched_sidecar = (
        project_root
        / "suze_experiments/20260313/consolidated_jsonl/results__aime.extract_answer_fixed.scorers.enriched.jsonl"
    )
    all_models = [
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
    ]
    hint_fractions = [round(i / 20.0, 2) for i in range(21)]

    out_eci, out_by_model = run_aime_raw_sanity_plot(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name="aime",
        solver="solution_intext_masked",
        condition="0shot",
        output_dir=output_dir,
        all_models=all_models,
        hint_fractions=hint_fractions,
    )

    debug_df = load_results_df(
        base_folder=base_folder,
        eci_file=eci_file,
        eval_name="aime",
        solver="solution_intext_masked",
        condition="0shot",
        all_models=all_models,
        hint_fractions=hint_fractions,
    )
    print('original results loaded')
    # debug_df = add_inference_owner(
    #     df=debug_df,
    #     base_folder=base_folder,
    #     eval_name="aime",
    #     solver="solution_intext_masked",
    #     condition="0shot",
    # )
    # print('inference owner added')
    # out_by_owner = plot_accuracy_vs_hint_by_model_raw_points_by_owner(
    #     df=debug_df,
    #     output_dir=output_dir,
    # )
    print('owner-colored plot created')

    print(f"Wrote: {out_eci}")
    print(f"Wrote: {out_by_model}")
    # print(f"Wrote: {out_by_owner}")

    # Regraded scorer plots, preferring enriched sidecar (standalone) and
    # falling back to DuckDB cache if enriched sidecar is unavailable.
    if regraded_enriched_sidecar.exists():
        print('regraded enriched sidecar exists')
        out_eci_regraded, out_by_model_regraded = run_aime_raw_sanity_plot_from_regraded_enriched_sidecar(
            sidecar_file=regraded_enriched_sidecar,
            eci_file=eci_file,
            output_dir=output_dir,
            scorer_name="aime_scorer_extract_answer_fixed",
            run_type="results",
            benchmark="aime",
            path_hint_level="solution_intext_masked/0shot",
            all_models=all_models,
            hint_fractions=hint_fractions,
            expected_epochs=10,
            n_bootstrap=1000,
        )
        print('regraded enriched sidecar plots created')
    else:
        print('regraded enriched sidecar does not exist; make it!')
        # out_eci_regraded, out_by_model_regraded = run_aime_raw_sanity_plot_from_regraded_cache(
        #     db_path=regraded_db,
        #     eci_file=eci_file,
        #     output_dir=output_dir,
        #     scorer_name="aime_scorer_extract_answer_fixed",
        #     run_type="results",
        #     benchmark="aime",
        #     path_hint_level="solution_intext_masked/0shot",
        #     all_models=all_models,
        #     hint_fractions=hint_fractions,
        #     expected_epochs=10,
        #     n_bootstrap=1000,
        # )
    print(f"Wrote: {out_eci_regraded}")
    print(f"Wrote: {out_by_model_regraded}")


if __name__ == "__main__":
    # python suze_experiments/20260307/aime_plots.py
    main()
