"""Load evaluation results into DataFrames."""

import json
import logging
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from .size import size

log = logging.getLogger(__name__)


def _get_nested(d: dict, path: str, default=None):
    """Get nested value from dict using dot notation (e.g., 'manual_bootstrap.accuracy')."""
    keys = path.split(".")
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is default:
            return default
    return d


def load_results(
    base_folder: str,
    eval_name: str,
    solver: str,
    condition: str = "0shot",
    filename_pattern: str = "{eval}_{solver}_{condition}_{hint}.json",
    metrics: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load hint-based experiment results into a DataFrame.

    Scans: {base_folder}/{eval_name}/{solver}/{condition}/{model}/*.json

    Args:
        base_folder: Base results directory (e.g., .../results/gpqa)
        eval_name: Evaluation name (e.g., 'gpqa', 'aime')
        solver: Solver name (e.g., 'solution_prefill_sequential')
        condition: Condition (e.g., '0shot')
        filename_pattern: Pattern for result filenames. Supports {eval}, {solver},
            {condition}, {hint} placeholders.
        metrics: Dict mapping output column names to JSON paths.
            Default: {"accuracy": "manual_bootstrap.accuracy", "stderr": "manual_bootstrap.stderr"}

    Returns:
        DataFrame with columns: model, model_size, hint, + metric columns
    """
    if metrics is None:
        metrics = {
            "accuracy": "manual_bootstrap.accuracy",
            "stderr": "manual_bootstrap.stderr",
        }

    folder = Path(base_folder) / eval_name / solver / condition
    if not folder.exists():
        log.warning(f"Folder not found: {folder}")
        return pd.DataFrame()

    rows = []
    # Build regex from filename pattern to extract hint
    pattern_regex = filename_pattern.format(
        eval=re.escape(eval_name),
        solver=re.escape(solver),
        condition=re.escape(condition),
        hint=r"([\d.]+)",
    )
    hint_re = re.compile(pattern_regex)

    for model_dir in folder.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for json_file in model_dir.glob("*.json"):
            match = hint_re.match(json_file.name)
            if not match:
                continue

            hint = float(match.group(1))

            with open(json_file) as f:
                data = json.load(f)

            row = {
                "model": model_name,
                "model_size": size(model_name),
                "hint": hint,
            }
            for col_name, json_path in metrics.items():
                row[col_name] = _get_nested(data, json_path)

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model_size", "hint"]).reset_index(drop=True)
    return df


def load_baseline(
    base_folder: str,
    eval_name: str,
    metrics: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load baseline evaluation results into a DataFrame.

    Scans: {base_folder}/{eval_name}/{model}/{eval_name}.json

    Args:
        base_folder: Base baseline directory (e.g., .../baseline)
        eval_name: Evaluation name (e.g., 'ifeval', 'math')
        metrics: Dict mapping output column names to JSON paths.
            Default extracts from common scorer fields.

    Returns:
        DataFrame with columns: model, model_size, + metric columns
    """
    if metrics is None:
        # Will be auto-detected per file
        metrics = {}

    folder = Path(base_folder) / eval_name
    if not folder.exists():
        log.warning(f"Folder not found: {folder}")
        return pd.DataFrame()

    rows = []
    for model_dir in folder.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        json_file = model_dir / f"{eval_name}.json"
        if not json_file.exists():
            continue

        with open(json_file) as f:
            data = json.load(f)

        row = {
            "model": model_name,
            "model_size": size(model_name),
        }

        if metrics:
            for col_name, json_path in metrics.items():
                row[col_name] = _get_nested(data, json_path)
        else:
            # Auto-extract: find scorer fields (dicts with 'accuracy' or similar)
            for key, value in data.items():
                if isinstance(value, dict) and "accuracy" in value:
                    row["accuracy"] = value.get("accuracy")
                    row["stderr"] = value.get("stderr")
                    row["scorer"] = key
                    break

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("model_size").reset_index(drop=True)
    return df


def load_all_baselines(
    base_folder: str,
    eval_configs: list[dict],
) -> pd.DataFrame:
    """Load multiple baseline evaluations into a single DataFrame.

    Args:
        base_folder: Base baseline directory
        eval_configs: List of dicts with keys:
            - eval_name: Evaluation name
            - metrics: Optional dict mapping column names to JSON paths
            - label: Optional display label (defaults to eval_name)

    Returns:
        DataFrame with columns: model, model_size, eval, accuracy, stderr, ...
    """
    dfs = []
    for config in eval_configs:
        eval_name = config["eval_name"]
        label = config.get("label", eval_name)
        metrics = config.get("metrics")

        df = load_baseline(base_folder, eval_name, metrics=metrics)
        if not df.empty:
            df["eval"] = label
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def add_derived_columns(
    df: pd.DataFrame,
    columns: dict[str, Callable[[pd.DataFrame], pd.Series]],
) -> pd.DataFrame:
    """Add derived columns to DataFrame.

    Args:
        df: Input DataFrame
        columns: Dict mapping new column names to functions that compute them.
            Each function takes the DataFrame and returns a Series.

    Returns:
        DataFrame with new columns added.

    Example:
        df = add_derived_columns(df, {
            "error_rate": lambda d: 1 - d["accuracy"],
            "log_size": lambda d: np.log(d["model_size"]),
        })
    """
    df = df.copy()
    for col_name, func in columns.items():
        df[col_name] = func(df)
    return df
