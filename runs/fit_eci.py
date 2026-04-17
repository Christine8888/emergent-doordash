from __future__ import annotations

"""Estimate per-model ECI from local inspect-backed benchmark exports.

Workflow:
1. collect benchmark scores for local models from `data/eci_scores`
2. load Epoch benchmark difficulties/slopes from `data/epoch_ai_data`
3. fit one ECI capability score per model by inverting Epoch's benchmark sigmoids

The `data/epoch_ai_data` folder in this repo comes from Epoch AI's public data:
https://epoch.ai/data/
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.storage import read_jsonl
from src.types import ECIScoreRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FOLDER = PROJECT_ROOT / "data" / "eci_scores"
EPOCH_DATA_DIR = PROJECT_ROOT / "data" / "epoch_ai_data"

EVAL_TO_ECI = {
    "hellaswag__split_validation": "HellaSwag",
    "piqa": "PIQA",
    "mmlu_5_shot__language_en_us__cot_true": "MMLU",
    "bbh__prompt_type_answer_only": "BBH",
    "arc_challenge": "ARC AI2",
    "winogrande__dataset_name_winogrande_xl__fewshot_5": "Winogrande",
    "math__levels_5__fewshot_0": "MATH level 5",
}

MIN_BENCHMARKS = len(EVAL_TO_ECI)
MIN_SCORE = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate ECI for models in data/eci_scores.")
    return parser.parse_args()


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _output_path() -> Path:
    eval_part = "--".join(sorted(_slug(name) for name in EVAL_TO_ECI.keys()))
    return PROJECT_ROOT / "data" / f"eci_model_capabilities__simple__{eval_part}.csv"
def _extract_is_correct(record: ECIScoreRecord) -> bool | None:
    for grader in record.graders:
        if isinstance(grader.is_correct, bool):
            return grader.is_correct
    return None


def load_baseline_scores() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    print("\nLoading baseline results...")

    for eval_name, benchmark_name in EVAL_TO_ECI.items():
        eval_dir = BASELINE_FOLDER / eval_name
        if not eval_dir.exists():
            raise FileNotFoundError(f"Missing baseline folder: {eval_dir}")

        n_models = 0
        for jsonl_path in sorted(eval_dir.glob("*.jsonl")):
            if not jsonl_path.is_file():
                continue

            file_rows = read_jsonl(jsonl_path, model_cls=ECIScoreRecord)
            typed_rows = [row for row in file_rows if isinstance(row, ECIScoreRecord)]
            judged = [flag for flag in (_extract_is_correct(row) for row in typed_rows) if flag is not None]
            if not judged:
                continue
            accuracy = sum(1.0 if flag else 0.0 for flag in judged) / len(judged)
            model_name = typed_rows[0].model if typed_rows else jsonl_path.stem

            rows.append(
                {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "score": accuracy,
                }
            )
            n_models += 1

        print(f"  {eval_name} -> {benchmark_name}: {n_models} models")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No baseline scores loaded.")
    return df


def load_epoch_params() -> tuple[dict[str, float], dict[str, float]]:
    csv_path = EPOCH_DATA_DIR / "additional_eci_data" / "eci_benchmark_difficulties_and_slopes.csv"
    df = pd.read_csv(csv_path)
    difficulty = dict(zip(df["benchmark_name"], df["edi"]))
    slope = dict(zip(df["benchmark_name"], df["estimated_slope_scaled"]))
    return difficulty, slope


def load_epoch_eci() -> dict[str, float]:
    csv_path = EPOCH_DATA_DIR / "epoch_capabilities_index.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ECI Score"])
    return dict(zip(df["Model version"], df["ECI Score"]))


def print_score_summary(scores_df: pd.DataFrame) -> None:
    benchmark_order = [EVAL_TO_ECI[eval_name] for eval_name in EVAL_TO_ECI]
    benchmark_alias = {
        "HellaSwag": "Hella",
        "PIQA": "PIQA",
        "MMLU": "MMLU",
        "BBH": "BBH",
        "ARC AI2": "ARC",
        "Winogrande": "Wino",
        "MATH level 5": "MATH5",
    }
    model_width = max(len("Model"), max(len(str(model)) for model in scores_df["model"].unique()))
    score_width = 8

    header = ["Model".ljust(model_width)] + [
        benchmark_alias.get(benchmark, benchmark)[:score_width].rjust(score_width)
        for benchmark in benchmark_order
    ]
    separator = ["-" * model_width] + ["-" * score_width for _ in benchmark_order]

    print("\nBenchmark scores per model:")
    print("  " + " ".join(header))
    print("  " + " ".join(separator))
    for model in sorted(scores_df["model"].unique()):
        model_df = scores_df[scores_df["model"] == model]
        score_map = dict(zip(model_df["benchmark"], model_df["score"]))
        row = [str(model).ljust(model_width)]
        for benchmark in benchmark_order:
            value = score_map.get(benchmark)
            row.append("--".rjust(score_width) if value is None else f"{float(value):.4f}".rjust(score_width))
        print("  " + " ".join(row))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def estimate_eci(scores_df: pd.DataFrame) -> dict[str, float]:
    difficulty, slope = load_epoch_params()
    valid_benchmarks = set(difficulty.keys()) & set(slope.keys())

    df = scores_df[scores_df["benchmark"].isin(valid_benchmarks)].copy()
    df = df[df["score"] >= MIN_SCORE].copy()
    df["score"] = df["score"].clip(0.001, 0.999)

    results: dict[str, float] = {}
    counts: dict[str, int] = {}
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model]
        if len(model_df) < MIN_BENCHMARKS:
            continue

        benchmarks = model_df["benchmark"].tolist()
        scores = model_df["score"].astype(float).to_numpy()
        db = np.asarray([float(difficulty[b]) for b in benchmarks], dtype=float)
        ab = np.asarray([float(slope[b]) for b in benchmarks], dtype=float)

        def loss(c_value: float) -> float:
            pred = sigmoid(ab * (c_value - db))
            return float(np.sum((pred - scores) ** 2))

        fit = minimize_scalar(loss, bounds=(50.0, 200.0), method="bounded")
        results[model] = float(fit.x)
        counts[model] = len(benchmarks)

    print("\nEstimated ECI per model:")
    for model in sorted(results.keys(), key=lambda m: results[m], reverse=True):
        print(f"  {model}: {counts[model]} benchmarks, ECI={results[model]:.1f}")

    return results


def main() -> None:
    _parse_args()

    epoch_eci = load_epoch_eci()
    print(f"Loaded {len(epoch_eci)} Epoch ECI scores for comparison")

    user_scores = load_baseline_scores()
    user_models = sorted(user_scores["model"].unique().tolist())
    print(f"\nTotal user scores: {len(user_scores)}")
    print(f"User models: {len(user_models)}")
    print_score_summary(user_scores)

    fitted_eci = estimate_eci(user_scores)

    print("\n" + "=" * 72)
    print(f"{'Model':<40} {'Our ECI':>10} {'Epoch ECI':>10}")
    print("=" * 72)
    for model in user_models:
        our_val = fitted_eci.get(model)
        epoch_val = epoch_eci.get(model)
        our_str = f"{our_val:>10.1f}" if our_val is not None else f"{'--':>10}"
        epoch_str = f"{epoch_val:>10.1f}" if epoch_val is not None else f"{'--':>10}"
        print(f"  {model:<38} {our_str} {epoch_str}")

    output_path = _output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        [
            {
                "model": model,
                "eci_our_fit": fitted_eci.get(model),
                "eci_epoch": epoch_eci.get(model),
            }
            for model in sorted(user_models, key=lambda m: fitted_eci.get(m, float("-inf")), reverse=True)
        ]
    )
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    # python -m runs.fit_eci
    main()
