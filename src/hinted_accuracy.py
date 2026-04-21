from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DATA_ROOT = Path("data")
LUKE_AIME_RESULTS_ROOT = DATA_ROOT / "luke_aime2025_2026_results"
LUKE_AIME_BENCHMARK = "aime2025_2026"
LUKE_SUPPORTED_HINT_TYPE = "answer_not_revealed"
LUKE_SUPPORTED_FRACTIONERS = {"mask_word", "truncate_word"}
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


def _is_luke_aime_combo(*, benchmark: str, hint_type: str, fractioner: str) -> bool:
    return (
        benchmark == LUKE_AIME_BENCHMARK
        and hint_type == LUKE_SUPPORTED_HINT_TYPE
        and fractioner in LUKE_SUPPORTED_FRACTIONERS
    )


def _expected_luke_problem_ids() -> list[str]:
    return [f"{LUKE_AIME_BENCHMARK}_{idx:04d}" for idx in range(1, 61)]


def _discover_luke_models_for_fractioner(fractioner: str) -> list[str]:
    fractioner_dir = LUKE_AIME_RESULTS_ROOT / safe_component(fractioner)
    if not fractioner_dir.exists():
        return []
    return sorted(
        path.name
        for path in fractioner_dir.iterdir()
        if path.is_dir() and path.name != "submitit_logs"
    )


def _discover_luke_fraction_files(
    *,
    model: str,
    fractioner: str,
) -> list[tuple[float, Path]]:
    model_dir = LUKE_AIME_RESULTS_ROOT / safe_component(fractioner) / safe_component(model)
    if not model_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    for path in model_dir.glob("fraction_*.jsonl"):
        try:
            fraction = parse_fraction_from_filename(path.name)
        except ValueError:
            continue
        out.append((fraction, path))
    return sorted(out, key=lambda pair: pair[0])


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


def is_complete_luke_fraction(path: Path) -> tuple[bool, str | None]:
    expected_problem_ids = set(_expected_luke_problem_ids())
    problem_counts: dict[str, int] = {}
    rows_total = 0

    try:
        for row in iter_jsonl(path):
            rows_total += 1
            if not isinstance(row, dict):
                return False, f"non-dict row at index={rows_total - 1}"
            problem_id = str(row.get("problem_id", "")).strip()
            if problem_id not in expected_problem_ids:
                return False, f"unexpected problem_id={problem_id!r}"
            hint_fraction = row.get("hint_fraction")
            if not isinstance(hint_fraction, (float, int)):
                return False, f"missing/invalid hint_fraction at row={rows_total - 1}"
            if extract_is_correct(row) is None:
                return False, f"missing/invalid correct label at row={rows_total - 1}"
            problem_counts[problem_id] = problem_counts.get(problem_id, 0) + 1
    except Exception as exc:
        return False, f"failed to read {path}: {exc}"

    if rows_total != 600:
        return False, f"expected 600 rows, found {rows_total}"
    if set(problem_counts.keys()) != expected_problem_ids:
        missing = sorted(expected_problem_ids - set(problem_counts.keys()))
        extra = sorted(set(problem_counts.keys()) - expected_problem_ids)
        return False, f"problem_id coverage mismatch missing={missing} extra={extra}"
    bad_counts = sorted(
        problem_id for problem_id, count in problem_counts.items() if count != 10
    )
    if bad_counts:
        preview = [(problem_id, problem_counts[problem_id]) for problem_id in bad_counts[:5]]
        return False, f"expected 10 rows per problem, bad_counts={preview}"
    return True, None


def extract_is_correct(row: dict[str, Any]) -> bool | None:
    direct_correct = row.get("correct")
    if isinstance(direct_correct, bool):
        return direct_correct

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


def load_luke_results_with_ci_for_combo(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
) -> dict[str, dict[float, dict[str, float]]]:
    if not _is_luke_aime_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    ):
        return {}

    rows: list[dict[str, Any]] = []
    for model in _discover_luke_models_for_fractioner(fractioner):
        model_rows, _warnings = collect_complete_fraction_stats_from_luke(
            benchmark=benchmark,
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
        )
        rows.extend(model_rows)
    return parse_results_with_ci_payload(rows_to_results_with_ci_payload(rows))


def parse_results_with_ci_payload(
    payload: dict[str, Any] | None,
) -> dict[str, dict[float, dict[str, float]]]:
    out: dict[str, dict[float, dict[str, float]]] = {}
    if not isinstance(payload, dict):
        return out

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
    models: set[str] = set()
    benchmark_dir = data_root / "hinted_inference" / safe_component(benchmark)
    if benchmark_dir.exists():
        models.update(path.name for path in benchmark_dir.iterdir() if path.is_dir())
    return sorted(models)


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


def collect_complete_fraction_stats_from_luke(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
    fractioner: str,
    expected_fractions: list[float] | None = None,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not _is_luke_aime_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    ):
        return [], [f"unsupported_luke_combo benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}"]

    expected = EXPECTED_FRACTIONS if expected_fractions is None else expected_fractions
    expected_fraction_set = {float(f"{value:.6f}") for value in expected}
    rng = np.random.default_rng(random_seed)

    fraction_files = _discover_luke_fraction_files(
        model=model,
        fractioner=fractioner,
    )
    if not fraction_files:
        return [], [f"no luke files for model={model} fractioner={fractioner}"]

    complete_fraction_files: list[tuple[float, Path]] = []
    incomplete_fraction_reasons: list[str] = []
    for hint_fraction, path in fraction_files:
        complete, reason = is_complete_luke_fraction(path)
        if complete:
            complete_fraction_files.append((float(hint_fraction), path))
        else:
            incomplete_fraction_reasons.append(
                f"{float(hint_fraction):.1f}:{reason or 'incomplete'}"
            )

    by_fraction = {float(f"{frac:.6f}"): path for frac, path in complete_fraction_files}
    missing = sorted(expected_fraction_set - set(by_fraction.keys()))
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
                f"unusable luke fraction rows model={model} fractioner={fractioner} "
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


def load_local_results_with_ci_for_combo(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    data_root: Path = DATA_ROOT,
) -> dict[str, dict[float, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    for model in discover_models_for_benchmark(benchmark, data_root=data_root):
        model_rows, _warnings = collect_complete_fraction_stats(
            benchmark=benchmark,
            model=model,
            hint_type=hint_type,
            fractioner=fractioner,
            data_root=data_root,
        )
        rows.extend(model_rows)

    local_payload = rows_to_results_with_ci_payload(rows)
    return parse_results_with_ci_payload(local_payload)


def _merge_results_with_ci_maps(
    *,
    base: dict[str, dict[float, dict[str, float]]],
    override: dict[str, dict[float, dict[str, float]]],
) -> dict[str, dict[float, dict[str, float]]]:
    out: dict[str, dict[float, dict[str, float]]] = {
        model: dict(fraction_map) for model, fraction_map in base.items()
    }
    for model, fraction_map in override.items():
        out.setdefault(model, {}).update(fraction_map)
    return out


def load_results_with_ci_for_combo(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    data_root: Path = DATA_ROOT,
) -> dict[str, dict[float, dict[str, float]]]:
    local_results = load_local_results_with_ci_for_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        data_root=data_root,
    )
    luke_results = load_luke_results_with_ci_for_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    )
    return _merge_results_with_ci_maps(base=luke_results, override=local_results)


def discover_models_for_combo(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    data_root: Path = DATA_ROOT,
) -> list[str]:
    payload = load_results_with_ci_for_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
        data_root=data_root,
    )
    return sorted(payload.keys())
