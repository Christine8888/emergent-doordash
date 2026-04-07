from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/accuracy_vs_hint")
N_BOOTSTRAP = 5000
RANDOM_SEED = 0
EXPECTED_FRACTIONS = [i / 10 for i in range(11)]


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hinted inference accuracy vs hint fraction with bootstrap CIs."
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'.")
    return parser.parse_args()


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_is_correct(row: dict[str, Any]) -> bool | None:
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


def _discover_models(benchmark: str) -> list[str]:
    benchmark_dir = DATA_ROOT / "hinted_inference" / _safe_component(benchmark)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Missing benchmark directory: {benchmark_dir}")
    return sorted(path.name for path in benchmark_dir.iterdir() if path.is_dir())


def _parse_fraction_from_filename(name: str) -> float:
    match = re.match(r"^fraction_(.+)\.jsonl$", name)
    if not match:
        raise ValueError(f"Unexpected fraction filename: {name}")
    return float(match.group(1))


def _discover_fraction_files(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
    fractioner: str,
) -> list[tuple[float, Path]]:
    benchmark_name = _safe_component(benchmark)
    model_name = _safe_component(model)
    hint_fractioner = f"{_safe_component(hint_type)}__{_safe_component(fractioner)}"
    combo_dir = DATA_ROOT / "hinted_inference" / benchmark_name / model_name / hint_fractioner

    if not combo_dir.exists():
        return []

    out: list[tuple[float, Path]] = []
    for path in combo_dir.glob("fraction_*.jsonl"):
        try:
            fraction = _parse_fraction_from_filename(path.name)
        except ValueError:
            continue
        out.append((fraction, path))
    return sorted(out, key=lambda pair: pair[0])


def _discover_fractioners(
    *,
    benchmark: str,
    model: str,
    hint_type: str,
) -> list[str]:
    benchmark_name = _safe_component(benchmark)
    model_name = _safe_component(model)
    hint_prefix = f"{_safe_component(hint_type)}__"
    model_dir = DATA_ROOT / "hinted_inference" / benchmark_name / model_name
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


def _bootstrap_accuracy(
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


def _collect_stats_for_fraction(
    *,
    path: Path,
    rng: np.random.Generator,
) -> dict[str, float | int] | None:
    sample_to_scores: dict[str, list[float]] = {}
    rows_total = 0
    rows_with_known_label = 0
    rows_without_known_label = 0

    for row in _iter_jsonl(path):
        rows_total += 1

        if not isinstance(row, dict):
            rows_without_known_label += 1
            continue

        problem_id = str(row.get("problem_id", "")).strip()
        if not problem_id:
            rows_without_known_label += 1
            continue

        is_correct = _extract_is_correct(row)
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

    point_accuracy, ci_low, ci_high = _bootstrap_accuracy(
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


def _plot(results: list[dict[str, Any]], *, output_png: Path, title: str) -> None:
    models = sorted({row["model"] for row in results})
    n_models = len(models)

    if n_models == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        axes = [ax]
    else:
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows))
        axes = axes_obj.flatten() if hasattr(axes_obj, "flatten") else [axes_obj]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_rows = [row for row in results if row["model"] == model]
        fractioners = sorted({str(row["fractioner"]) for row in model_rows})
        for fractioner in fractioners:
            series_rows = sorted(
                [row for row in model_rows if row["fractioner"] == fractioner],
                key=lambda row: float(row["hint_fraction"]),
            )
            x = np.asarray([float(row["hint_fraction"]) for row in series_rows], dtype=float)
            y = np.asarray([float(row["accuracy"]) for row in series_rows], dtype=float)
            low = np.asarray([float(row["ci_low"]) for row in series_rows], dtype=float)
            high = np.asarray([float(row["ci_high"]) for row in series_rows], dtype=float)
            yerr = np.vstack([y - low, high - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                alpha=0.9,
                linewidth=1.5,
                markersize=3.8,
                capsize=2.0,
                label=fractioner,
            )
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Hint Fraction")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    models_available = _discover_models(args.benchmark)
    if not models_available:
        raise ValueError(f"No models found for benchmark={args.benchmark!r}.")

    if args.model == "all":
        models_to_plot = models_available
    else:
        if args.model not in models_available:
            raise ValueError(
                f"Requested model {args.model!r} not found. "
                f"Available: {models_available}"
            )
        models_to_plot = [args.model]

    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []

    expected_fraction_set = {float(f"{value:.6f}") for value in EXPECTED_FRACTIONS}

    for model in models_to_plot:
        fractioners = _discover_fractioners(
            benchmark=args.benchmark,
            model=model,
            hint_type=args.hint_type,
        )
        if not fractioners:
            print(
                f"[plot_hinted_accuracy_vs_hint][WARN] no fractioners for "
                f"model={model} hint_type={args.hint_type}"
            )
            continue

        for fractioner in fractioners:
            fraction_files = _discover_fraction_files(
                benchmark=args.benchmark,
                model=model,
                hint_type=args.hint_type,
                fractioner=fractioner,
            )
            if not fraction_files:
                continue

            by_fraction = {float(f"{frac:.6f}"): path for frac, path in fraction_files}
            available_fraction_set = set(by_fraction.keys())
            missing = sorted(expected_fraction_set - available_fraction_set)
            if missing:
                print(
                    f"[plot_hinted_accuracy_vs_hint][WARN] skipping incomplete fractioner "
                    f"model={model} fractioner={fractioner} missing_fractions={missing}"
                )
                continue

            fraction_rows: list[dict[str, Any]] = []
            usable = True
            for hint_fraction in EXPECTED_FRACTIONS:
                key = float(f"{hint_fraction:.6f}")
                path = by_fraction[key]
                stats = _collect_stats_for_fraction(path=path, rng=rng)
                if stats is None:
                    print(
                        f"[plot_hinted_accuracy_vs_hint][WARN] skipping fractioner due to "
                        f"unusable fraction rows model={model} fractioner={fractioner} "
                        f"fraction={hint_fraction} path={path}"
                    )
                    usable = False
                    break
                fraction_rows.append(
                    {
                        "model": model,
                        "fractioner": fractioner,
                        "hint_fraction": float(hint_fraction),
                        **stats,
                        "path": str(path),
                    }
                )

            if not usable:
                continue

            rows.extend(fraction_rows)
            print(
                f"[plot_hinted_accuracy_vs_hint] included model={model} "
                f"fractioner={fractioner} n_points={len(fraction_rows)}"
            )

    if not rows:
        raise ValueError("No usable rows collected. Check benchmark/model/hint_type/fractioner.")

    rows_sorted = sorted(
        rows,
        key=lambda r: (str(r["model"]), str(r["fractioner"]), float(r["hint_fraction"])),
    )

    stem = (
        f"{_safe_component(args.benchmark)}__{_safe_component(args.hint_type)}__"
        f"all_complete_fractioners__{_safe_component(args.model)}"
    )
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = PLOTS_ROOT / f"{stem}__bootstrap.csv"
    json_path = PLOTS_ROOT / f"{stem}__bootstrap.json"
    png_path = PLOTS_ROOT / f"{stem}__bootstrap.png"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "fractioner",
                "hint_fraction",
                "accuracy",
                "ci_low",
                "ci_high",
                "n_samples",
                "n_rollouts",
                "rows_total",
                "rows_with_known_label",
                "rows_without_known_label",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": args.benchmark,
                "hint_type": args.hint_type,
                "fractioners_mode": "all_complete",
                "model": args.model,
                "n_bootstrap": N_BOOTSTRAP,
                "random_seed": RANDOM_SEED,
                "rows": rows_sorted,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _plot(
        rows_sorted,
        output_png=png_path,
        title=(
            f"Hinted Accuracy vs Hint Fraction\n"
            f"benchmark={args.benchmark} hint_type={args.hint_type} "
            f"(all complete fractioners)"
        ),
    )

    print(f"[plot_hinted_accuracy_vs_hint] wrote_csv={csv_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_json={json_path}")
    print(f"[plot_hinted_accuracy_vs_hint] wrote_plot={png_path}")


if __name__ == "__main__":
    # python -m runs.plot_hinted_accuracy_vs_hint --benchmark aime2025_2026 --hint-type answer_not_revealed --model Qwen3-4B
    main()
