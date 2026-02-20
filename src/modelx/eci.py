"""Epoch Capabilities Index (ECI) estimation from benchmark scores.

Two approaches:
1. fit_eci() - Joint fitting of Cm, Db, αb from scratch (Epoch's approach)
2. estimate_eci() - Invert using Epoch's pre-fitted Db, αb (approximation)

See ECI.md for methodology details.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# Default paths
_MODULE_DIR = Path(__file__).parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent  # src/modelx -> src -> project root
_EPOCH_DATA_DIR = _PROJECT_ROOT / "benchmark_data"
_USER_SCORES_PATH = _MODULE_DIR / "model_scores.csv"
_MANUAL_SCORES_PATH = _MODULE_DIR / "model_scores_manual.csv"


def load_epoch_benchmark_scores(
    data_dir: Path | str = _EPOCH_DATA_DIR,
    benchmarks: list[str] | None = None,
    only_eci_models: bool = True,
) -> pd.DataFrame:
    """Load Epoch's raw benchmark scores into standard format.

    Args:
        data_dir: Path to benchmark_data folder
        benchmarks: List of benchmarks to load. None = all available.
        only_eci_models: If True, only include models that have ECI scores in Epoch's data.

    Returns:
        DataFrame with columns: model, benchmark, score
    """
    data_dir = Path(data_dir)

    # Mapping: benchmark name -> (filename, score column)
    # These match Epoch's benchmark names from eci_benchmark_difficulties_and_slopes.csv
    BENCHMARK_FILES = {
        # Easy benchmarks
        "LAMBADA": ("lambada_external.csv", "Score"),
        "TriviaQA": ("trivia_qa_external.csv", "EM"),
        "PIQA": ("piqa_external.csv", "Score"),
        "HellaSwag": ("hella_swag_external.csv", "Overall accuracy"),
        "OpenBookQA": ("open_book_qa_external.csv", "Accuracy"),
        "ARC AI2": ("arc_ai2_external.csv", "Challenge score"),
        "GSM8K": ("gsm8k_external.csv", "EM"),
        "VideoMME": ("video_mme_external.csv", "Overall (no subtitles)"),
        "Lech Mazur Writing": ("lech_mazur_writing_external.csv", "Mean score"),
        "Winogrande": ("wino_grande_external.csv", "Accuracy"),
        "MMLU": ("mmlu_external.csv", "EM"),
        "ScienceQA": ("science_qa_external.csv", "Score"),
        "BBH": ("bbh_external.csv", "Average"),
        "GeoBench": ("geobench_external.csv", "Photos Avg Score"),
        "MATH level 5": ("math_level_5.csv", "mean_score"),
        "ANLI": ("adversarial_nli_external.csv", "Score"),
        "Fiction.LiveBench": ("fictionlivebench_external.csv", "120k token score"),
        "GPQA diamond": ("gpqa_diamond.csv", "mean_score"),
        "OTIS Mock AIME 2024-2025": ("otis_mock_aime_2024_2025.csv", "mean_score"),
        "Aider polyglot": ("aider_polyglot_external.csv", "Percent correct"),
        "CadEval": ("cad_eval_external.csv", "Overall pass (%)"),
        "SWE-Bench Verified (Bash Only)": ("swe_bench_bash.csv", "% Resolved"),
        "WeirdML": ("weirdml_external.csv", "Accuracy"),
        "VPCT": ("vpct_external.csv", "Correct"),
        "ARC-AGI": ("arc_agi_external.csv", "Score"),
        "OSWorld": ("os_world_external.csv", "Score"),
        "The Agent Company": ("the_agent_company_external.csv", "% Score"),
        "Cybench": ("cybench_external.csv", "Unguided % Solved"),
        "DeepResearch Bench": ("deepresearchbench_external.csv", "Average score"),
        "SimpleBench": ("simplebench_external.csv", "Score (AVG@5)"),
        "SimpleQA Verified": ("simpleqa_verified.csv", "mean_score"),
        "Terminal Bench": ("terminalbench_external.csv", "Accuracy mean"),
        "FrontierMath-2025-02-28-Private": ("frontiermath.csv", "mean_score"),
        "Chess Puzzles": ("chess_puzzles.csv", "mean_score"),
        "Balrog": ("balrog_external.csv", "Average progress"),
        "FrontierMath-Tier-4-2025-07-01-Private": ("frontiermath_tier_4.csv", "mean_score"),
        "GSO-Bench": ("gso_external.csv", "Score OPT@1"),
    }

    if benchmarks is None:
        benchmarks = list(BENCHMARK_FILES.keys())

    # Benchmarks that need score normalization (divisor to get 0-1 range)
    SCORE_DIVISORS = {
        "Lech Mazur Writing": 10.0,  # 0-10 scale
        "Aider polyglot": 100.0,     # percentage
        "OSWorld": 100.0,            # percentage
        "The Agent Company": 1.0,    # already 0-1
        "CadEval": 100.0,            # percentage
        "SWE-Bench Verified (Bash Only)": 100.0,  # percentage
        "Cybench": 100.0,            # percentage
    }

    # Benchmarks to skip (scores not convertible to 0-1)
    SKIP_BENCHMARKS = {"GeoBench"}  # Distance-based scores

    rows = []
    for bench in benchmarks:
        if bench in SKIP_BENCHMARKS:
            continue
        if bench not in BENCHMARK_FILES:
            logger.warning(f"Unknown benchmark: {bench}")
            continue

        filename, score_col = BENCHMARK_FILES[bench]
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        df = pd.read_csv(filepath)
        if score_col not in df.columns:
            logger.warning(f"Score column '{score_col}' not in {filename}")
            continue

        divisor = SCORE_DIVISORS.get(bench, 1.0)

        for _, row in df.iterrows():
            model = row.get("Model version") or row.get("Name")
            score = row.get(score_col)
            if pd.notna(model) and pd.notna(score):
                normalized_score = float(score) / divisor
                # Clamp to valid range
                normalized_score = max(0.001, min(0.999, normalized_score))
                rows.append({
                    "model": str(model).strip(),
                    "benchmark": bench,
                    "score": normalized_score,
                })

    result_df = pd.DataFrame(rows)

    # Filter to only ECI models + user's models
    if only_eci_models and len(result_df) > 0:
        allowed_models = set(load_epoch_eci(data_dir).keys())
        # Also include user's models (from both auto and manual scores)
        if _USER_SCORES_PATH.exists():
            auto_df = pd.read_csv(_USER_SCORES_PATH)
            allowed_models |= set(auto_df["model"].dropna().unique())
        if _MANUAL_SCORES_PATH.exists():
            manual_df = pd.read_csv(_MANUAL_SCORES_PATH)
            allowed_models |= set(manual_df["model"].dropna().unique())
        result_df = result_df[result_df["model"].isin(allowed_models)]

    return result_df


def load_epoch_params(
    data_dir: Path | str = _EPOCH_DATA_DIR,
) -> dict:
    """Load Epoch's fitted benchmark parameters (Db, αb).

    Returns dict with:
        - difficulty: {benchmark_name: Db}
        - slope: {benchmark_name: αb}
        - benchmarks: list of benchmark names
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "additional_eci_data" / "eci_benchmark_difficulties_and_slopes.csv"

    df = pd.read_csv(csv_path)

    return {
        "difficulty": dict(zip(df["benchmark_name"], df["edi"])),
        "slope": dict(zip(df["benchmark_name"], df["estimated_slope_scaled"])),
        "benchmarks": df["benchmark_name"].tolist(),
    }


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def fit_eci(
    scores_df: pd.DataFrame,
    anchor_benchmark: str = "Winogrande",
    reg_strength: float = 0.1,
    exclude_benchmarks: list[str] | None = None,
    min_benchmarks: int = 5,
    min_score: float = 0.05,
) -> dict:
    """Fit ECI model using Epoch's exact approach (benchmark anchoring).

    Fits: score = σ(αb × (Cm - Db))

    Uses benchmark anchoring per Epoch's paper:
    1. Initialize C, D at 0, α at 1
    2. Fit with L2 regularization (default 0.1)
    3. Fix anchor benchmark's α=1, then shift so D_anchor=0

    Args:
        scores_df: DataFrame with columns [model, benchmark, score]
        anchor_benchmark: Benchmark to anchor (α=1, D=0 after shifting)
        reg_strength: L2 regularization strength (default 0.1 per Epoch)
        exclude_benchmarks: List of benchmark names to exclude from fitting
        min_benchmarks: Minimum number of benchmark scores required per model
        min_score: Exclude scores below this threshold (likely erroneous)

    Returns:
        dict with:
            - Cm: {model: capability}
            - Db: {benchmark: difficulty}
            - ab: {benchmark: slope}
            - predictions: DataFrame with predicted scores
            - rmse: root mean squared error
    """
    df = scores_df.dropna(subset=["score"]).copy()

    # Filter out suspiciously low scores (likely erroneous, e.g. 0% MMLU)
    n_before = len(df)
    df = df[df["score"] >= min_score]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} scores below {min_score}")

    # Exclude specified benchmarks
    if exclude_benchmarks:
        df = df[~df["benchmark"].isin(exclude_benchmarks)]
        logger.info(f"Excluded benchmarks: {exclude_benchmarks}")

    # Filter models with too few benchmarks
    if min_benchmarks > 1:
        model_counts = df.groupby("model").size()
        valid_models = model_counts[model_counts >= min_benchmarks].index
        n_before = df["model"].nunique()
        df = df[df["model"].isin(valid_models)]
        n_after = df["model"].nunique()
        if n_before > n_after:
            logger.info(f"Filtered models with <{min_benchmarks} benchmarks: {n_before} -> {n_after}")

    # Clip scores to valid range
    df["score"] = df["score"].clip(0.001, 0.999)

    models = df["model"].unique().tolist()
    benchmarks = df["benchmark"].unique().tolist()

    n_models = len(models)
    n_benchmarks = len(benchmarks)

    model_idx = {m: i for i, m in enumerate(models)}
    bench_idx = {b: i for i, b in enumerate(benchmarks)}

    # Validate anchor benchmark exists
    if anchor_benchmark not in bench_idx:
        logger.warning(f"Anchor benchmark '{anchor_benchmark}' not in data, using first benchmark")
        anchor_benchmark = benchmarks[0]
    anchor_bench_idx = bench_idx[anchor_benchmark]

    # Pre-compute indices and scores as arrays
    m_indices = np.array([model_idx[m] for m in df["model"]])
    b_indices = np.array([bench_idx[b] for b in df["benchmark"]])
    scores_arr = df["score"].values

    def split_params(params):
        """Unpack params: all C, all D, free α (anchor α fixed at 1)."""
        C = params[:n_models]
        D = params[n_models:n_models + n_benchmarks]
        alpha_free = params[n_models + n_benchmarks:]

        # Reconstruct full alpha with anchor fixed at 1
        alpha = np.ones(n_benchmarks)
        free_idx = 0
        for i in range(n_benchmarks):
            if i == anchor_bench_idx:
                alpha[i] = 1.0
            else:
                alpha[i] = alpha_free[free_idx]
                free_idx += 1
        return C, D, alpha

    def residuals(params):
        C, D, alpha = split_params(params)
        pred = sigmoid(alpha[b_indices] * (C[m_indices] - D[b_indices]))
        resid = pred - scores_arr

        # L2 regularization (Epoch style: C, D toward 0; α toward 1)
        if reg_strength > 0:
            n_params = n_models + n_benchmarks + (n_benchmarks - 1)
            reg_term = reg_strength * (
                np.sum(C ** 2) +
                np.sum(D ** 2) +
                np.sum((alpha - 1) ** 2)
            ) / n_params
            reg_penalty = np.sqrt(reg_term) if reg_term > 0 else 0
            return np.append(resid, reg_penalty)

        return resid

    # Initialize per Epoch: C=0, D=0, α=1
    x0 = np.concatenate([
        np.zeros(n_models),           # C
        np.zeros(n_benchmarks),       # D
        np.ones(n_benchmarks - 1),    # α_free (anchor is fixed at 1)
    ])

    # Bound alpha to be positive
    lower = np.concatenate([
        np.full(n_models, -np.inf),
        np.full(n_benchmarks, -np.inf),
        np.full(n_benchmarks - 1, 0.001),
    ])
    upper = np.concatenate([
        np.full(n_models, np.inf),
        np.full(n_benchmarks, np.inf),
        np.full(n_benchmarks - 1, np.inf),
    ])

    result = least_squares(residuals, x0, bounds=(lower, upper), method='trf')
    Cm, Db, ab = split_params(result.x)

    # Shift so anchor benchmark has D=0 (per Epoch's identifiability fix)
    shift = Db[anchor_bench_idx]
    Cm = Cm - shift
    Db = Db - shift

    # Compute predictions
    pred = sigmoid(ab[b_indices] * (Cm[m_indices] - Db[b_indices]))
    pred_df = pd.DataFrame({
        "model": df["model"].values,
        "benchmark": df["benchmark"].values,
        "score": scores_arr,
        "predicted": pred,
        "error": pred - scores_arr,
    })
    rmse = np.sqrt((pred_df["error"] ** 2).mean())

    return {
        "Cm": dict(zip(models, Cm)),
        "Db": dict(zip(benchmarks, Db)),
        "ab": dict(zip(benchmarks, ab)),
        "predictions": pred_df,
        "rmse": rmse,
        "anchor_benchmark": anchor_benchmark,
    }


def load_user_scores(
    csv_path: Path | str = _USER_SCORES_PATH,
    manual_path: Path | str = _MANUAL_SCORES_PATH,
    include_manual: bool = True,
) -> pd.DataFrame:
    """Load user's model benchmark scores (programmatic + manual).

    Merges two sources:
    1. model_scores.csv - auto-generated from baseline evals
    2. model_scores_manual.csv - hard-coded values (e.g., from system cards)

    Manual scores take precedence if there's overlap.

    Returns DataFrame with columns: model, benchmark, score
    """
    dfs = []

    # Load programmatic scores
    csv_path = Path(csv_path)
    if csv_path.exists():
        dfs.append(pd.read_csv(csv_path)[["model", "benchmark", "score"]])

    # Load manual scores (takes precedence)
    if include_manual:
        manual_path = Path(manual_path)
        if manual_path.exists():
            manual_df = pd.read_csv(manual_path)[["model", "benchmark", "score"]]
            dfs.append(manual_df)

    if not dfs:
        logger.warning("No score files found")
        return pd.DataFrame(columns=["model", "benchmark", "score"])

    # Concat and dedupe (later entries = manual take precedence)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "benchmark"], keep="last")

    # Filter out placeholder rows
    combined = combined[~combined["model"].str.startswith("_", na=False)]

    return combined


def load_epoch_eci(
    data_dir: Path | str = _EPOCH_DATA_DIR,
) -> dict[str, float]:
    """Load pre-computed ECI scores from Epoch's data.

    Returns dict mapping model version string -> ECI score.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "epoch_capabilities_index.csv"

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ECI Score"])

    return dict(zip(df["Model version"], df["ECI Score"]))


def list_benchmarks() -> list[str]:
    """List available benchmark names from Epoch's data."""
    params = load_epoch_params()
    return params["benchmarks"]


def estimate_eci_from_epoch_params(
    scores_df: pd.DataFrame,
    min_benchmarks: int = 3,
    min_score: float = 0.05,
) -> dict[str, float]:
    """Estimate ECI for models using Epoch's pre-fitted benchmark parameters.

    This is the principled approach when you want ECIs consistent with Epoch's scale.
    Instead of re-fitting D and α, we use Epoch's values and only solve for C.

    For each model, finds C that minimizes:
        Σ_b (score_mb - σ(αb × (C - Db)))²

    Args:
        scores_df: DataFrame with columns [model, benchmark, score]
        min_benchmarks: Minimum benchmarks required per model
        min_score: Exclude scores below this threshold (likely erroneous)

    Returns:
        Dict mapping model -> ECI score (on Epoch's scale, ~100-160)
    """
    from scipy.optimize import minimize_scalar

    # Load Epoch's benchmark parameters
    params = load_epoch_params()
    D = params["difficulty"]  # benchmark -> difficulty
    alpha = params["slope"]   # benchmark -> slope

    # Filter to benchmarks we have params for
    valid_benchmarks = set(D.keys())
    df = scores_df[scores_df["benchmark"].isin(valid_benchmarks)].copy()

    # Filter out suspiciously low scores (likely erroneous, e.g. 0% MMLU)
    n_before = len(df)
    df = df[df["score"] >= min_score]
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} scores below {min_score}")

    df["score"] = df["score"].clip(0.001, 0.999)

    results = {}
    benchmark_counts = {}

    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        if len(model_df) < min_benchmarks:
            logger.info(f"Skipping {model}: only {len(model_df)} benchmarks (need {min_benchmarks})")
            continue

        benchmarks = model_df["benchmark"].values
        scores = model_df["score"].values

        # Get D and α for this model's benchmarks
        Db = np.array([D[b] for b in benchmarks])
        ab = np.array([alpha[b] for b in benchmarks])

        def loss(C):
            pred = sigmoid(ab * (C - Db))
            return np.sum((pred - scores) ** 2)

        # Optimize - search in reasonable range
        result = minimize_scalar(loss, bounds=(50, 200), method='bounded')
        results[model] = result.x
        benchmark_counts[model] = len(benchmarks)

    # Print benchmark counts for each model
    print(f"\nECI estimation - benchmarks used per model:")
    for model in sorted(results.keys(), key=lambda m: results[m], reverse=True):
        print(f"  {model}: {benchmark_counts[model]} benchmarks, ECI={results[model]:.1f}")

    return results


# Mapping from your eval names to ECI benchmark names
EVAL_TO_ECI = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "mmlu_5_shot_cot": "MMLU",
    "math_level_5": "MATH level 5",
    "bbh": "BBH",
    "arc_challenge": "ARC AI2",  # Epoch only uses Challenge score
    "winogrande": "Winogrande",
    # Excluded or not directly mappable:
    # "gpqa" - excluded per user request
    # "mmlu_0_shot" - using 5_shot instead
    # "math" - use math_level_5 instead
    # "arc_easy" - Epoch only uses Challenge, not Easy
    # "bbeh" - different from BBH
    # "aime", "hle", "ifeval", "niah", "commonsense_qa" - not in ECI
}


def refresh_model_scores(
    baseline_folder: str,
    output_path: Path | str | None = None,
    eval_mapping: dict[str, str] | None = None,
    exclude_evals: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> pd.DataFrame:
    """Refresh model_scores.csv from baseline results.

    Args:
        baseline_folder: Path to baseline results (e.g., .../baseline)
        output_path: Where to save CSV (default: src/modelx/model_scores.csv)
        eval_mapping: Override eval name -> ECI benchmark mapping
        exclude_evals: Evals to exclude (default: none beyond unmapped)
        exclude_models: Model name patterns to exclude (substring match)

    Returns:
        DataFrame with columns: model, benchmark, score
    """
    from .results import load_baseline

    if output_path is None:
        output_path = _USER_SCORES_PATH
    output_path = Path(output_path)

    if eval_mapping is None:
        eval_mapping = EVAL_TO_ECI

    if exclude_evals is None:
        exclude_evals = []

    if exclude_models is None:
        exclude_models = []

    baseline_folder = Path(baseline_folder)
    rows = []

    for eval_name, eci_benchmark in eval_mapping.items():
        if eval_name in exclude_evals:
            continue

        eval_folder = baseline_folder / eval_name
        if not eval_folder.exists():
            logger.info(f"Skipping {eval_name}: folder not found")
            continue

        df = load_baseline(str(baseline_folder), eval_name)
        if df.empty:
            logger.info(f"Skipping {eval_name}: no results")
            continue

        for _, row in df.iterrows():
            model_name = row["model"]
            if any(pat in model_name for pat in exclude_models):
                continue
            if pd.notna(row.get("accuracy")):
                rows.append({
                    "model": model_name,
                    "benchmark": eci_benchmark,
                    "score": row["accuracy"],
                })

        logger.info(f"Loaded {len(df)} models from {eval_name} -> {eci_benchmark}")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["model", "benchmark"]).reset_index(drop=True)

    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(result_df)} rows to {output_path}")

    return result_df
