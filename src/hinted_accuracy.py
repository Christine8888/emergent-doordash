from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DATA_ROOT = Path("data")
EXTERNAL_RESULTS_WITH_CI_PATHS = {
    "mask_word": DATA_ROOT / "results_with_ci_mask_word.json",
    "truncate_word": DATA_ROOT / "results_with_ci_truncate_word.json",
}
EXPECTED_FRACTIONS = [i / 10 for i in range(11)]
N_BOOTSTRAP = 5000
RANDOM_SEED = 0


def safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_fraction_from_filename(name: str) -> float:
    match = re.match(r"^fraction_(.+)\.jsonl$", name)
    if not match:
        raise ValueError(f"Unexpected fraction filename: {name}")
    return float(match.group(1))


def checkpoint_path_for_fraction(path: Path) -> Path:
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected .jsonl path, got: {path}")
    return path.with_suffix(".ckpt.json")


def is_complete_fraction(path: Path) -> tuple[bool, str | None]:
    ckpt_path = checkpoint_path_for_fraction(path)
    if not ckpt_path.exists():
        return False, f"missing checkpoint {ckpt_path}"

    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
    except Exception as exc:
        return False, f"failed to read checkpoint {ckpt_path}: {exc}"

    if not isinstance(ckpt, dict):
        return False, f"invalid checkpoint payload {ckpt_path}"

    total_candidates = ckpt.get("total_candidates")
    processed_this_run = ckpt.get("processed_this_run")
    skipped_existing = ckpt.get("skipped_existing")
    remaining = ckpt.get("remaining")

    if not isinstance(total_candidates, int) or total_candidates < 0:
        return False, f"invalid total_candidates in {ckpt_path}"
    if not isinstance(processed_this_run, int) or processed_this_run < 0:
        return False, f"invalid processed_this_run in {ckpt_path}"
    if not isinstance(skipped_existing, int) or skipped_existing < 0:
        return False, f"invalid skipped_existing in {ckpt_path}"
    if not isinstance(remaining, int) or remaining < 0:
        return False, f"invalid remaining in {ckpt_path}"

    completed_total = processed_this_run + skipped_existing
    if remaining != 0:
        return False, f"remaining={remaining}"
    if completed_total < total_candidates:
        return False, (
            f"incomplete completed_total={completed_total} total_candidates={total_candidates}"
        )
    return True, None


def extract_is_correct(row: dict[str, Any]) -> bool | None:
    graders = row.get("graders")
    if not isinstance(graders, list):
        return None

    for grader in graders:
        if not isinstance(grader, dict):
            continue
        is_correct = grader.get("is_correct")
        if isinstance(is_correct, bool):
            return is_correct
    return None


def bootstrap_accuracy(
    *,
    sample_to_scores: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    sample_arrays = list(sample_to_scores.values())
    if not sample_arrays:
        raise ValueError("No sample arrays available for bootstrap.")

    point_accuracy = float(np.mean([arr.mean() for arr in sample_arrays]))

    boot_sums = np.zeros(N_BOOTSTRAP, dtype=float)
    for arr in sample_arrays:
        draw_idx = rng.integers(low=0, high=arr.size, size=N_BOOTSTRAP)
        boot_sums += arr[draw_idx]
    boot_means = boot_sums / float(len(sample_arrays))
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    return point_accuracy, float(ci_low), float(ci_high)


def collect_stats_for_fraction(
    *,
    path: Path,
    rng: np.random.Generator,
) -> dict[str, float | int] | None:
    sample_to_scores: dict[str, list[float]] = {}
    rows_total = 0
    rows_with_known_label = 0
    rows_without_known_label = 0

    for row in iter_jsonl(path):
        rows_total += 1

        if not isinstance(row, dict):
            rows_without_known_label += 1
            continue

        problem_id = str(row.get("problem_id", "")).strip()
        if not problem_id:
            rows_without_known_label += 1
            continue

        is_correct = extract_is_correct(row)
        if is_correct is None:
            rows_without_known_label += 1
            continue

        rows_with_known_label += 1
        sample_to_scores.setdefault(problem_id, []).append(1.0 if is_correct else 0.0)

    sample_to_arrays = {
        sample_id: np.asarray(values, dtype=float)
        for sample_id, values in sample_to_scores.items()
        if len(values) > 0
    }
    if not sample_to_arrays:
        return None

    point_accuracy, ci_low, ci_high = bootstrap_accuracy(
        sample_to_scores=sample_to_arrays,
        rng=rng,
    )
    n_rollouts = int(sum(arr.size for arr in sample_to_arrays.values()))

    return {
        "accuracy": point_accuracy,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_samples": int(len(sample_to_arrays)),
        "n_rollouts": n_rollouts,
        "rows_total": int(rows_total),
        "rows_with_known_label": int(rows_with_known_label),
        "rows_without_known_label": int(rows_without_known_label),
    }


def load_external_results_with_ci(
    path: Path,
) -> dict[str, dict[float, dict[str, float]]] | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {path}, got {type(payload).__name__}")

    out: dict[str, dict[float, dict[str, float]]] = {}
    for model, model_payload in payload.items():
        if not isinstance(model, str) or not isinstance(model_payload, dict):
            continue
        fraction_map: dict[float, dict[str, float]] = {}
        for hint_fraction_raw, stats in model_payload.items():
            if not isinstance(stats, dict):
                continue
            try:
                hint_fraction = float(hint_fraction_raw)
            except (TypeError, ValueError):
                continue
            mean = stats.get("mean")
            ci_lower = stats.get("ci_lower")
            ci_upper = stats.get("ci_upper")
            if not all(isinstance(value, (float, int)) for value in (mean, ci_lower, ci_upper)):
                continue
            fraction_map[float(hint_fraction)] = {
                "accuracy": float(mean),
                "ci_low": float(ci_lower),
                "ci_high": float(ci_upper),
            }
        if fraction_map:
            out[model] = fraction_map
    return out


def discover_models_for_benchmark(benchmark: str, *, data_root: Path = DATA_ROOT) -> list[str]:
    benchmark_dir = data_root / "hinted_inference" / safe_component(benchmark)
    if not benchmark_dir.exists():
        return []
    return sorted(path.name for path in benchmark_dir.iterdir() if path.is_dir())


def discover_fractioners(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
    data_root: Path = DATA_ROOT,
) -> list[str]:
    benchmark_name = safe_component(benchmark)
    model_name = safe_component(model)
    hint_prefix = f"{safe_component(hint_type)}__"
    model_dir = data_root / "hinted_inference" / benchmark_name / model_name
    if not model_dir.exists():
        return []

    fractioners: list[str] = []
    for path in model_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(hint_prefix):
            continue
        parts = path.name.split("__", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        fractioners.append(parts[1])
    return sorted(set(fractioners))


def discover_fraction_files(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
    fractioner: str,
    data_root: Path = DATA_ROOT,
) -> list[tuple[float, Path]]:
    benchmark_name = safe_component(benchmark)
    model_name = safe_component(model)
    hint_fractioner = f"{safe_component(hint_type)}__{safe_component(fractioner)}"
    combo_dir = data_root / "hinted_inference" / benchmark_name / model_name / hint_fractioner

    if not combo_dir.exists():
        return []

    out: list[tuple[float, Path]] = []
    for path in combo_dir.glob("fraction_*.jsonl"):
        try:
            fraction = parse_fraction_from_filename(path.name)
        except ValueError:
            continue
        out.append((fraction, path))
    return sorted(out, key=lambda pair: pair[0])


def collect_complete_fraction_stats(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
    fractioner: str,
    expected_fractions: list[float] | None = None,
    data_root: Path = DATA_ROOT,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[str]]:
    expected = EXPECTED_FRACTIONS if expected_fractions is None else expected_fractions
    expected_fraction_set = {float(f"{value:.6f}") for value in expected}
    rng = np.random.default_rng(random_seed)

    fraction_files = discover_fraction_files(
        benchmark=benchmark,
        model=model,
        hint_type=hint_type,
        fractioner=fractioner,
        data_root=data_root,
    )
    if not fraction_files:
        return [], [f"no files for model={model} fractioner={fractioner}"]

    complete_fraction_files: list[tuple[float, Path]] = []
    incomplete_fraction_reasons: list[str] = []
    for hint_fraction, path in fraction_files:
        complete, reason = is_complete_fraction(path)
        if complete:
            complete_fraction_files.append((float(hint_fraction), path))
        else:
            incomplete_fraction_reasons.append(
                f"{float(hint_fraction):.1f}:{reason or 'incomplete'}"
            )

    by_fraction = {float(f"{frac:.6f}"): path for frac, path in complete_fraction_files}
    available_fraction_set = set(by_fraction.keys())
    missing = sorted(expected_fraction_set - available_fraction_set)
    if missing:
        return [], [
            f"missing_fractions={missing} incomplete_points={incomplete_fraction_reasons}"
        ]

    rows: list[dict[str, Any]] = []
    warnings = list(incomplete_fraction_reasons)
    for hint_fraction in expected:
        path = by_fraction[float(f"{hint_fraction:.6f}")]
        stats = collect_stats_for_fraction(path=path, rng=rng)
        if stats is None:
            warnings.append(
                f"unusable fraction rows model={model} fractioner={fractioner} "
                f"fraction={hint_fraction} path={path}"
            )
            continue
        rows.append(
            {
                "model": model,
                "fractioner": fractioner,
                "hint_fraction": float(hint_fraction),
                **stats,
                "path": str(path),
            }
        )

    return rows, warnings


def rows_to_results_with_ci_payload(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    payload: dict[str, dict[str, dict[str, float]]] = {}
    rows_sorted = sorted(
        rows,
        key=lambda row: (str(row["model"]), float(row["hint_fraction"])),
    )
    for row in rows_sorted:
        model = str(row["model"])
        fraction_key = f"{float(row['hint_fraction']):.1f}"
        payload.setdefault(model, {})[fraction_key] = {
            "mean": float(row["accuracy"]),
            "ci_lower": float(row["ci_low"]),
            "ci_upper": float(row["ci_high"]),
        }
    return payload


def external_results_to_payload(
    external_results: dict[str, dict[float, dict[str, float]]] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    payload: dict[str, dict[str, dict[str, float]]] = {}
    if not external_results:
        return payload

    for model, fraction_map in sorted(external_results.items()):
        for hint_fraction, stats in sorted(fraction_map.items()):
            fraction_key = f"{float(hint_fraction):.1f}"
            payload.setdefault(model, {})[fraction_key] = {
                "mean": float(stats["accuracy"]),
                "ci_lower": float(stats["ci_low"]),
                "ci_upper": float(stats["ci_high"]),
            }
    return payload

