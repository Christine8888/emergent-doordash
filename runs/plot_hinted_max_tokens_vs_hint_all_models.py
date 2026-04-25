from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.model_config import is_model_excluded_for_fractioner


DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/hinted_max_tokens_vs_hint")
EXPECTED_FRACTIONS = [i / 10 for i in range(11)]


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot histograms of per-fraction max output_token_count for all models with hinted-inference data."
        )
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument(
        "--hint-type",
        type=str,
        default=None,
        help="Optional hint type filter. If omitted, plot every discovered hint type.",
    )
    parser.add_argument(
        "--fractioner",
        type=str,
        required=True,
        help="Fractioner to plot.",
    )
    parser.add_argument(
        "--hint-fraction",
        type=float,
        default=None,
        help="Optional specific hint fraction to include (e.g. 0.3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_ROOT,
        help="Directory for plots and summary CSV.",
    )
    return parser.parse_args()


def _benchmark_dir(benchmark: str) -> Path:
    return DATA_ROOT / "hinted_inference" / _safe_component(benchmark)


def _parse_fraction_from_filename(name: str) -> float:
    match = re.match(r"^fraction_(.+)\.jsonl$", name)
    if not match:
        raise ValueError(f"Unexpected fraction filename: {name}")
    return float(match.group(1))


def _split_combo_name(combo_name: str) -> tuple[str, str]:
    parts = combo_name.split("__", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Unexpected combo directory name: {combo_name}")
    return parts[0], parts[1]


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_completion_stats(path: Path) -> dict[str, Any]:
    record_count = 0
    error_count = 0
    max_output_tokens: int | None = None

    for row in _iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        record_count += 1
        if bool(row.get("is_error")):
            error_count += 1
        output_token_count = row.get("output_token_count")
        if isinstance(output_token_count, int):
            if max_output_tokens is None or output_token_count > max_output_tokens:
                max_output_tokens = output_token_count

    return {
        "record_count": record_count,
        "error_count": error_count,
        "max_output_tokens": max_output_tokens,
    }


def _read_checkpoint_complete(ckpt_path: Path) -> bool | None:
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    total_candidates = payload.get("total_candidates")
    processed_this_run = payload.get("processed_this_run")
    skipped_existing = payload.get("skipped_existing")
    remaining = payload.get("remaining")
    if not all(isinstance(v, int) for v in (total_candidates, processed_this_run, skipped_existing, remaining)):
        return None
    return remaining == 0 and (processed_this_run + skipped_existing) >= total_candidates


def collect_rows(
    *,
    benchmark: str,
    hint_type_filter: str | None,
    fractioner_filter: str | None,
    hint_fraction_filter: float | None,
) -> list[dict[str, Any]]:
    benchmark_dir = _benchmark_dir(benchmark)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Missing benchmark directory: {benchmark_dir}")

    rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        if is_model_excluded_for_fractioner(model, fractioner_filter):
            continue
        for combo_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            try:
                hint_type, fractioner = _split_combo_name(combo_dir.name)
            except ValueError:
                continue
            if hint_type_filter is not None and hint_type != _safe_component(hint_type_filter):
                continue
            if fractioner_filter is not None and fractioner != _safe_component(fractioner_filter):
                continue

            for jsonl_path in sorted(combo_dir.glob("fraction_*.jsonl")):
                try:
                    hint_fraction = _parse_fraction_from_filename(jsonl_path.name)
                except ValueError:
                    continue
                if hint_fraction_filter is not None and not math.isclose(
                    hint_fraction,
                    hint_fraction_filter,
                    abs_tol=1e-9,
                ):
                    continue
                stats = _read_completion_stats(jsonl_path)
                ckpt_complete = _read_checkpoint_complete(jsonl_path.with_suffix(".ckpt.json"))
                rows.append(
                    {
                        "benchmark": _safe_component(benchmark),
                        "model": model,
                        "hint_type": hint_type,
                        "fractioner": fractioner,
                        "hint_fraction": hint_fraction,
                        "record_count": stats["record_count"],
                        "error_count": stats["error_count"],
                        "max_output_tokens": stats["max_output_tokens"],
                        "is_complete": ckpt_complete,
                        "jsonl_path": str(jsonl_path),
                    }
                )
                print(
                    f"collected model={model} hint_type={hint_type} fractioner={fractioner} "
                    f"fraction={hint_fraction:.1f} rows={stats['record_count']} "
                    f"max_output_tokens={stats['max_output_tokens']}",
                    flush=True,
                )

    if not rows:
        raise ValueError("No hinted-inference rows found for the requested filters.")
    return rows


def _group_rows_by_hint_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["hint_type"]), []).append(row)
    return grouped


def _sort_models(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["model"]) for row in rows})


def _model_series(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    model_rows = [row for row in rows if row["model"] == model and row["max_output_tokens"] is not None]
    return sorted(model_rows, key=lambda row: float(row["hint_fraction"]))


def _load_output_token_lengths(row: dict[str, Any]) -> list[int]:
    jsonl_path = Path(str(row["jsonl_path"]))
    lengths: list[int] = []
    for record in _iter_jsonl(jsonl_path):
        if not isinstance(record, dict):
            continue
        output_token_count = record.get("output_token_count")
        if isinstance(output_token_count, int):
            lengths.append(output_token_count)
    return lengths


def _plot_hint_type_histogram(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    hint_fraction_filter: float | None,
) -> Path:
    models = [model for model in _sort_models(rows) if _model_series(rows, model)]
    if not models:
        raise ValueError(f"No plottable rows for hint_type={hint_type} fractioner={fractioner}")

    n_cols = 5
    n_rows = max(4, math.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), squeeze=False)
    axes_flat = list(axes.flatten())
    max_hist_count = 0.0

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        series = _model_series(rows, model)
        complete_values: list[int] = []
        incomplete_values: list[int] = []
        for row in series:
            target = complete_values if row["is_complete"] is True else incomplete_values
            target.extend(_load_output_token_lengths(row))

        if complete_values:
            complete_counts, _, _ = ax.hist(
                complete_values,
                bins=20,
                color="#1f77b4",
                alpha=0.45,
                edgecolor="none",
                label="complete",
            )
            if len(complete_counts) > 0:
                max_hist_count = max(max_hist_count, float(max(complete_counts)))
        if incomplete_values:
            incomplete_counts, _, _ = ax.hist(
                incomplete_values,
                bins=20,
                color="#ff7f0e",
                alpha=0.45,
                edgecolor="none",
                label="partial/unknown",
            )
            if len(incomplete_counts) > 0:
                max_hist_count = max(max_hist_count, float(max(incomplete_counts)))

        ax.set_title(model, fontsize=9)
        ax.set_xlabel("output token count")
        ax.set_ylabel("count")
        ax.axvline(32000, color="#cc4444", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    shared_y_max = max(1.0, max_hist_count * 1.05)
    for idx in range(len(models)):
        axes_flat[idx].set_ylim(0.0, shared_y_max)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        (
            f"{benchmark}: output token length histogram by model\n"
            f"hint_type={hint_type} fractioner={fractioner}"
            + (
                f" hint_fraction={hint_fraction_filter:.1f}"
                if hint_fraction_filter is not None
                else ""
            )
        ),
        fontsize=13,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"{_safe_component(benchmark)}__{_safe_component(hint_type)}__"
        f"{_safe_component(fractioner)}__"
        f"{_safe_component(f'hint_fraction_{hint_fraction_filter:.1f}') if hint_fraction_filter is not None else 'all_hint_fractions'}__"
        f"max_tokens_hist_by_model.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_csv(rows: list[dict[str, Any]], output_dir: Path, benchmark: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_component(benchmark)}__hinted_max_tokens_vs_hint_all_models.csv"
    fieldnames = [
        "benchmark",
        "model",
        "hint_type",
        "fractioner",
        "hint_fraction",
        "record_count",
        "error_count",
        "max_output_tokens",
        "is_complete",
        "jsonl_path",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["hint_type"]),
                str(item["fractioner"]),
                str(item["model"]),
                float(item["hint_fraction"]),
            ),
        ):
            writer.writerow(row)
    return out_path


def main() -> None:
    args = _parse_args()
    rows = collect_rows(
        benchmark=args.benchmark,
        hint_type_filter=args.hint_type,
        fractioner_filter=args.fractioner,
        hint_fraction_filter=args.hint_fraction,
    )
    csv_path = _write_csv(rows, args.output_dir, args.benchmark)

    hint_type_to_rows = _group_rows_by_hint_type(rows)
    plot_paths: list[Path] = []
    for hint_type, hint_type_rows in sorted(hint_type_to_rows.items()):
        plot_path = _plot_hint_type_histogram(
            benchmark=_safe_component(args.benchmark),
            hint_type=hint_type,
            fractioner=_safe_component(args.fractioner),
            rows=hint_type_rows,
            output_dir=args.output_dir,
            hint_fraction_filter=args.hint_fraction,
        )
        plot_paths.append(plot_path)
        model_count = len({row["model"] for row in hint_type_rows if row["max_output_tokens"] is not None})
        print(
            f"wrote plot={plot_path} models={model_count} rows={len(hint_type_rows)}",
            flush=True,
        )

    print(f"wrote csv={csv_path}", flush=True)
    print(f"plots_written={len(plot_paths)}", flush=True)


if __name__ == "__main__":
    # python -m runs.plot_hinted_max_tokens_vs_hint_all_models --benchmark aime2025_2026 --fractioner mask_word
    # python -m runs.plot_hinted_max_tokens_vs_hint_all_models --benchmark aime2025_2026 --fractioner mask_word --hint-fraction 0.0
    main()
