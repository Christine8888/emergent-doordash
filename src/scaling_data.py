from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.hinted_accuracy import load_results_with_ci_for_combo


def canonicalize_model_name(model: str) -> str:
    if "/" in model:
        return model
    if model.startswith("Qwen"):
        return f"Qwen/{model}"
    if model.startswith("Llama-"):
        return f"meta-llama/{model}"
    if model.startswith("gemma-"):
        return f"google/{model}"
    return model


def infer_model_family(model: str) -> str:
    canonical_model = canonicalize_model_name(str(model))
    if canonical_model.startswith("Qwen/Qwen3-"):
        return "Qwen3"
    if canonical_model.startswith("Qwen/Qwen2.5-"):
        return "Qwen2.5"
    if canonical_model.startswith("google/gemma-3-"):
        return "gemma-3"
    if canonical_model.startswith("meta-llama/Llama-3.1-"):
        return "Llama-3.1"
    provider, _, remainder = canonical_model.partition("/")
    if provider and remainder:
        family = remainder.split("-", 1)[0].strip()
        if family:
            return family
    return canonical_model


def load_eci_map(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "model" not in reader.fieldnames or "eci_our_fit" not in reader.fieldnames:
            raise ValueError(f"Expected columns 'model' and 'eci_our_fit' in {path}")

        out: dict[str, float] = {}
        for row in reader:
            model = str(row.get("model", "")).strip()
            eci_raw = row.get("eci_our_fit")
            if not model or eci_raw in (None, ""):
                continue
            out[canonicalize_model_name(model)] = float(eci_raw)
    return out


def eci_benchmark_label(path: Path) -> str:
    stem = path.stem
    prefix = "eci_model_capabilities__simple__"
    if not stem.startswith(prefix):
        return "unknown"
    encoded = stem[len(prefix) :]
    if not encoded:
        return "unknown"
    benchmark_names: list[str] = []
    for part in encoded.split("--"):
        name = part.split("__", 1)[0].strip()
        if name and name not in benchmark_names:
            benchmark_names.append(name)
    return ", ".join(benchmark_names) if benchmark_names else "unknown"


def load_canonical_combo_results(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
) -> tuple[dict[str, dict[float, dict[str, float]]], list[str]]:
    combo_results = load_results_with_ci_for_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    )
    canonical_combo_results = {
        canonicalize_model_name(str(model)): stats
        for model, stats in combo_results.items()
    }
    return canonical_combo_results, sorted(canonical_combo_results.keys())


def resolve_models_to_use(
    *,
    available_models: list[str],
    benchmark: str,
    preferred_models: list[str] | None,
) -> list[str]:
    canonical_available_models = [
        canonicalize_model_name(str(model)) for model in available_models
    ]
    if preferred_models is None:
        return canonical_available_models

    canonical_models_to_use = [
        canonicalize_model_name(str(model)) for model in preferred_models
    ]
    missing_models = sorted(set(canonical_models_to_use) - set(canonical_available_models))
    if missing_models:
        raise ValueError(
            f"Configured preferred models missing for benchmark={benchmark}: {missing_models}. "
            f"Available models: {sorted(canonical_available_models)}"
        )
    return canonical_models_to_use


def build_base_rows(
    *,
    combo_results: dict[str, dict[float, dict[str, float]]],
    models: list[str],
    fractioner: str,
    benchmark: str,
) -> list[dict[str, Any]]:
    base_rows: list[dict[str, Any]] = []
    for model in models:
        if model not in combo_results:
            raise ValueError(
                f"Configured model missing combo results for benchmark={benchmark}: "
                f"model={model}"
            )
        for hint_fraction, stats in sorted(combo_results[model].items()):
            base_rows.append(
                {
                    "model": model,
                    "fractioner": fractioner,
                    "hint_fraction": float(hint_fraction),
                    "accuracy": float(stats["accuracy"]),
                    "ci_low": float(stats["ci_low"]),
                    "ci_high": float(stats["ci_high"]),
                }
            )
    return base_rows


def build_x_rows(
    *,
    base_rows: list[dict[str, Any]],
    x_map: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "x_value": float(x_map[str(row["model"])]),
            "model_family": infer_model_family(str(row["model"])),
        }
        for row in base_rows
        if str(row["model"]) in x_map
    ]
