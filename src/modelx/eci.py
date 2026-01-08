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
_EPOCH_DATA_DIR = Path("/Users/christineye/emergent-doordash/benchmark_data")
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
    anchor_benchmark: str | None = None,
    reg_lambda: float = 0.1,
) -> dict:
    """Fit ECI model jointly (Epoch's approach).

    Fits: score = σ(αb × (Cm - Db))

    Args:
        scores_df: DataFrame with columns [model, benchmark, score]
        anchor_benchmark: Benchmark to anchor (D=0, α=1). If None, uses first.
        reg_lambda: L2 regularization strength

    Returns:
        dict with:
            - Cm: {model: capability}
            - Db: {benchmark: difficulty}
            - ab: {benchmark: slope}
            - predictions: DataFrame with predicted scores
            - rmse: root mean squared error
    """
    df = scores_df.dropna(subset=["score"]).copy()

    models = df["model"].unique().tolist()
    benchmarks = df["benchmark"].unique().tolist()

    if anchor_benchmark is None:
        anchor_benchmark = benchmarks[0]
    if anchor_benchmark not in benchmarks:
        raise ValueError(f"Anchor benchmark '{anchor_benchmark}' not in data")

    anchor_idx = benchmarks.index(anchor_benchmark)

    n_models = len(models)
    n_benchmarks = len(benchmarks)
    n_obs = len(df)

    model_idx = {m: i for i, m in enumerate(models)}
    bench_idx = {b: i for i, b in enumerate(benchmarks)}

    # Pre-compute indices and scores as arrays for vectorization
    m_indices = np.array([model_idx[m] for m in df["model"]])
    b_indices = np.array([bench_idx[b] for b in df["benchmark"]])
    scores_arr = df["score"].values

    def unpack(params):
        Cm = params[:n_models].copy()
        Db = params[n_models:n_models + n_benchmarks].copy()
        ab = params[n_models + n_benchmarks:].copy()
        # Fix anchor
        Db[anchor_idx] = 0.0
        ab[anchor_idx] = 1.0
        return Cm, Db, ab

    def residuals(params):
        Cm, Db, ab = unpack(params)

        # Vectorized prediction
        pred = sigmoid(ab[b_indices] * (Cm[m_indices] - Db[b_indices]))
        resid = pred - scores_arr

        # L2 regularization (exclude anchor from Db, ab regularization)
        reg_Cm = np.sqrt(reg_lambda) * Cm
        reg_Db = np.sqrt(reg_lambda) * np.delete(Db, anchor_idx)
        reg_ab = np.sqrt(reg_lambda) * (np.delete(ab, anchor_idx) - 1)

        return np.concatenate([resid, reg_Cm, reg_Db, reg_ab])

    # Initialize
    x0 = np.concatenate([
        np.zeros(n_models),       # Cm
        np.zeros(n_benchmarks),   # Db
        np.ones(n_benchmarks),    # αb
    ])

    result = least_squares(residuals, x0, method='trf')
    Cm, Db, ab = unpack(result.x)

    # Linear rescale so Claude 3.5 Sonnet (June) = 130, GPT-5 = 150
    # Transform: Cm' = a*Cm + b, Db' = a*Db + b, ab' = ab/a
    # This preserves predictions: σ(ab'*(Cm'-Db')) = σ((ab/a)*a*(Cm-Db)) = σ(ab*(Cm-Db))
    ref_models = {
        "claude-3-5-sonnet-20240620": 130.0,
        "gpt-5-2025-08-07_medium": 150.0,
    }
    ref_vals = []
    ref_targets = []
    for m, target in ref_models.items():
        if m in model_idx:
            ref_vals.append(Cm[model_idx[m]])
            ref_targets.append(target)

    if len(ref_vals) >= 2:
        # Solve for a, b: target = a * val + b
        A = np.array([[ref_vals[0], 1], [ref_vals[1], 1]])
        B = np.array(ref_targets)
        a, b = np.linalg.solve(A, B)
    elif len(ref_vals) == 1:
        # Just shift
        a, b = 1.0, ref_targets[0] - ref_vals[0]
    else:
        a, b = 1.0, 100.0  # fallback

    Cm = a * Cm + b
    Db = a * Db + b
    ab = ab / a  # Adjust slopes to preserve predictions

    # Vectorized predictions
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
        "anchor": anchor_benchmark,
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


def estimate_eci_single(
    score: float,
    benchmark: str,
    params: dict,
) -> float | None:
    """Estimate ECI for a single (score, benchmark) pair.

    Formula: Cm = Db + logit(score) / αb

    Args:
        score: Benchmark score (0 to 1)
        benchmark: Benchmark name (must match Epoch's naming)
        params: Output of load_epoch_params()

    Returns:
        Estimated ECI score, or None if benchmark not found
    """
    if benchmark not in params["difficulty"]:
        logger.warning(f"Benchmark '{benchmark}' not in Epoch's data")
        return None

    # Clamp score to avoid infinite logit
    score = np.clip(score, 0.001, 0.999)

    Db = params["difficulty"][benchmark]
    ab = params["slope"][benchmark]

    logit_score = np.log(score / (1 - score))
    Cm = Db + logit_score / ab

    return float(Cm)


def estimate_eci(
    scores: dict[str, float],
    params: dict | None = None,
) -> float | None:
    """Estimate ECI from multiple benchmark scores (averaged).

    Args:
        scores: {benchmark_name: score} mapping
        params: Output of load_epoch_params(), loaded if None

    Returns:
        Average estimated ECI across benchmarks, or None if no valid estimates
    """
    if params is None:
        params = load_epoch_params()

    estimates = []
    for benchmark, score in scores.items():
        Cm = estimate_eci_single(score, benchmark, params)
        if Cm is not None:
            estimates.append(Cm)

    if not estimates:
        return None

    return float(np.mean(estimates))


def get_eci(
    model: str,
    params: dict | None = None,
    user_scores: pd.DataFrame | None = None,
    epoch_eci: dict[str, float] | None = None,
    fallback_to_epoch: bool = True,
) -> float | None:
    """Get ECI for a model, using user scores or falling back to Epoch's data.

    Priority:
        1. Estimate from user's benchmark scores (if available)
        2. Lookup in Epoch's pre-computed ECI (if fallback_to_epoch=True)
        3. Return None

    Args:
        model: Model name
        params: Epoch parameters (loaded if None)
        user_scores: User's scores DataFrame (loaded if None)
        epoch_eci: Epoch's pre-computed ECI (loaded if fallback_to_epoch and None)
        fallback_to_epoch: Whether to use Epoch's data if user scores unavailable

    Returns:
        ECI score or None
    """
    if params is None:
        params = load_epoch_params()
    if user_scores is None:
        user_scores = load_user_scores()

    # Try user's scores first
    model_scores = user_scores[user_scores["model"] == model]
    if len(model_scores) > 0:
        scores_dict = dict(zip(model_scores["benchmark"], model_scores["score"]))
        eci = estimate_eci(scores_dict, params)
        if eci is not None:
            return eci

    # Fallback to Epoch's pre-computed values
    if fallback_to_epoch:
        if epoch_eci is None:
            epoch_eci = load_epoch_eci()

        # Try exact match first
        if model in epoch_eci:
            return epoch_eci[model]

        # Try partial match (model name contained in version string)
        for version, eci in epoch_eci.items():
            if model.lower() in version.lower():
                return eci

    return None


def list_benchmarks() -> list[str]:
    """List available benchmark names from Epoch's data."""
    params = load_epoch_params()
    return params["benchmarks"]


# Mapping from your eval names to ECI benchmark names
EVAL_TO_ECI = {
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "mmlu_5_shot": "MMLU",
    "math_level_5": "MATH level 5",
    "bbh": "BBH",
    # Excluded or not directly mappable:
    # "gpqa" - excluded per user request
    # "mmlu_0_shot" - using 5_shot instead
    # "math" - use math_level_5 instead
    # "arc_challenge" - ECI's ARC AI2 combines easy+challenge
    # "arc_easy" - ECI's ARC AI2 combines easy+challenge
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
