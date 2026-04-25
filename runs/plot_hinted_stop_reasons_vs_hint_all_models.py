from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.model_config import is_model_excluded_for_fractioner, models_excluded_from_selection


DATA_ROOT = Path("data")
PLOTS_ROOT = Path("plots/hinted_stop_reasons_vs_hint")
EXPECTED_FRACTIONS = [i / 10 for i in range(11)]
STOP_REASON_COLORS = {
    "stop": "#2ca02c",
    "token_limit": "#d62728",
    "error": "#7f7f7f",
    "missing": "#9467bd",
}
TOKEN_LIMIT_STOP_REASONS = {
    "length",
    "max_tokens",
    "model_length",
    "max_output_tokens",
    "max_completion_tokens",
}


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-model stop-reason percentages vs hint fraction for hinted-inference data."
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
        "--model",
        type=str,
        nargs="+",
        default=["all"],
        help="One-or-more model names, or 'all'.",
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


def _extract_stop_reason(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    stop_reason = None
    if isinstance(metadata, dict):
        stop_reason = metadata.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        stop_reason = row.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason.strip():
        cleaned = stop_reason.strip()
        if cleaned in TOKEN_LIMIT_STOP_REASONS:
            return "token_limit"
        return cleaned
    if bool(row.get("is_error")):
        return "error"
    return "missing"


def _read_stop_reason_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        counts[_extract_stop_reason(row)] += 1
    return counts


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
    fractioner_filter: str,
    model_filter: set[str] | None,
) -> list[dict[str, Any]]:
    benchmark_dir = _benchmark_dir(benchmark)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Missing benchmark directory: {benchmark_dir}")

    rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        if is_model_excluded_for_fractioner(model, fractioner_filter):
            continue
        if model_filter is not None and model not in model_filter:
            continue
        for combo_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            try:
                hint_type, fractioner = _split_combo_name(combo_dir.name)
            except ValueError:
                continue
            if hint_type_filter is not None and hint_type != _safe_component(hint_type_filter):
                continue
            if fractioner != _safe_component(fractioner_filter):
                continue

            for jsonl_path in sorted(combo_dir.glob("fraction_*.jsonl")):
                try:
                    hint_fraction = _parse_fraction_from_filename(jsonl_path.name)
                except ValueError:
                    continue

                counts = _read_stop_reason_counts(jsonl_path)
                record_count = int(sum(counts.values()))
                is_complete = _read_checkpoint_complete(jsonl_path.with_suffix(".ckpt.json"))
                for stop_reason, count in sorted(counts.items()):
                    rows.append(
                        {
                            "benchmark": _safe_component(benchmark),
                            "model": model,
                            "hint_type": hint_type,
                            "fractioner": fractioner,
                            "hint_fraction": hint_fraction,
                            "stop_reason": stop_reason,
                            "count": int(count),
                            "record_count": record_count,
                            "percentage": 100.0 * float(count) / float(record_count) if record_count else 0.0,
                            "is_complete": is_complete,
                            "jsonl_path": str(jsonl_path),
                        }
                    )
                print(
                    f"collected model={model} hint_type={hint_type} fractioner={fractioner} "
                    f"fraction={hint_fraction:.1f} rows={record_count} stop_reasons={dict(sorted(counts.items()))}",
                    flush=True,
                )

    if not rows:
        raise ValueError("No hinted-inference rows found for the requested filters.")
    return rows


def _complete_models_for_rows(rows: list[dict[str, Any]]) -> set[str]:
    by_model_fraction: dict[str, dict[float, int]] = {}
    for row in rows:
        model = str(row["model"])
        hint_fraction = float(row["hint_fraction"])
        by_model_fraction.setdefault(model, {})[hint_fraction] = int(row["record_count"])

    expected_fractions = {float(f"{fraction:.6f}") for fraction in EXPECTED_FRACTIONS}
    complete_models: set[str] = set()
    for model, fraction_counts in sorted(by_model_fraction.items()):
        normalized_counts = {
            float(f"{fraction:.6f}"): count
            for fraction, count in fraction_counts.items()
        }
        if set(normalized_counts) != expected_fractions:
            continue
        if all(normalized_counts[fraction] == 600 for fraction in expected_fractions):
            complete_models.add(model)
    return complete_models


def _filter_complete_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete_models = _complete_models_for_rows(rows)
    skipped_models = sorted({str(row["model"]) for row in rows} - complete_models)
    if skipped_models:
        print(
            "[plot_hinted_stop_reasons_vs_hint_all_models] skipping incomplete models: "
            f"{skipped_models}",
            flush=True,
        )
    filtered_rows = [row for row in rows if str(row["model"]) in complete_models]
    if not filtered_rows:
        raise ValueError("No complete models found with all 6600 responses.")
    return filtered_rows


def _group_rows_by_hint_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["hint_type"]), []).append(row)
    return grouped


def _sort_models(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["model"]) for row in rows})


def _sort_stop_reasons(rows: list[dict[str, Any]]) -> list[str]:
    reasons = {str(row["stop_reason"]) for row in rows}
    preferred = [reason for reason in ("stop", "token_limit", "error", "missing") if reason in reasons]
    return preferred + sorted(reasons - set(preferred))


def _series_for_model(
    *,
    rows: list[dict[str, Any]],
    model: str,
    stop_reasons: list[str],
) -> tuple[list[float], dict[str, list[float]]]:
    fractions = sorted({float(row["hint_fraction"]) for row in rows if str(row["model"]) == model})
    by_key = {
        (float(row["hint_fraction"]), str(row["stop_reason"])): float(row["percentage"])
        for row in rows
        if str(row["model"]) == model
    }
    reason_to_percentages = {
        reason: [by_key.get((fraction, reason), 0.0) for fraction in fractions]
        for reason in stop_reasons
    }
    return fractions, reason_to_percentages


def _plot_hint_type(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    model_component: str,
) -> Path:
    models = _sort_models(rows)
    stop_reasons = _sort_stop_reasons(rows)
    if not models:
        raise ValueError(f"No plottable rows for hint_type={hint_type} fractioner={fractioner}")

    n_cols = 5
    n_rows = max(1, math.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.3 * n_rows), squeeze=False)
    axes_flat = list(axes.flatten())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    fallback_colors = {
        reason: color_cycle[idx % len(color_cycle)] if color_cycle else None
        for idx, reason in enumerate(stop_reasons)
    }

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        fractions, reason_to_percentages = _series_for_model(
            rows=rows,
            model=model,
            stop_reasons=stop_reasons,
        )
        values = reason_to_percentages.get("token_limit", [0.0 for _ in fractions])
        bar_width = 0.075
        ax.bar(
            fractions,
            values,
            width=bar_width,
            label="token_limit",
            color=STOP_REASON_COLORS["token_limit"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.35,
            align="center",
        )
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Hint Fraction")
        ax.set_ylabel("Token Limit %")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 100.0)
        ax.set_xticks(EXPECTED_FRACTIONS)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.grid(True, alpha=0.3)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            fontsize=9,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle(
        (
            f"{benchmark}: token-limit percentage vs hint fraction\n"
            f"hint_type={hint_type} fractioner={fractioner}"
        ),
        fontsize=13,
        y=1.035 if handles else 0.995,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"{_safe_component(benchmark)}__{_safe_component(hint_type)}__"
        f"{_safe_component(fractioner)}__{model_component}__"
        f"token_limit_vs_hint_by_model.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_csv(rows: list[dict[str, Any]], output_dir: Path, benchmark: str, model_component: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        output_dir
        / f"{_safe_component(benchmark)}__{model_component}__hinted_stop_reasons_vs_hint_all_models.csv"
    )
    fieldnames = [
        "benchmark",
        "model",
        "hint_type",
        "fractioner",
        "hint_fraction",
        "stop_reason",
        "count",
        "record_count",
        "percentage",
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
                str(item["stop_reason"]),
            ),
        ):
            writer.writerow(row)
    return out_path


def main() -> None:
    args = _parse_args()
    requested_models = list(args.model)
    if requested_models == ["all"]:
        model_filter = None
        model_component = "all"
    else:
        if "all" in requested_models:
            raise ValueError(
                "When passing specific models, do not include 'all'. "
                f"Requested: {requested_models}"
            )
        excluded_requested_models = models_excluded_from_selection(
            requested_models,
            args.fractioner,
        )
        if excluded_requested_models:
            raise ValueError(
                f"Requested model(s) excluded for fractioner={args.fractioner!r}: "
                f"{excluded_requested_models}"
            )
        model_filter = set(requested_models)
        model_component = "__".join(_safe_component(model) for model in sorted(model_filter))

    rows = collect_rows(
        benchmark=args.benchmark,
        hint_type_filter=args.hint_type,
        fractioner_filter=args.fractioner,
        model_filter=model_filter,
    )
    rows = _filter_complete_models(rows)
    found_models = {str(row["model"]) for row in rows}
    if model_filter is not None:
        missing_models = sorted(model_filter - found_models)
        if missing_models:
            raise ValueError(
                f"Requested model(s) not found for filters: {missing_models}. "
                f"Found models: {sorted(found_models)}"
            )

    csv_path = _write_csv(rows, args.output_dir, args.benchmark, model_component)
    plot_paths: list[Path] = []
    for hint_type, hint_type_rows in sorted(_group_rows_by_hint_type(rows).items()):
        plot_path = _plot_hint_type(
            benchmark=_safe_component(args.benchmark),
            hint_type=hint_type,
            fractioner=_safe_component(args.fractioner),
            rows=hint_type_rows,
            output_dir=args.output_dir,
            model_component=model_component,
        )
        plot_paths.append(plot_path)
        print(
            f"wrote plot={plot_path} models={len(_sort_models(hint_type_rows))} rows={len(hint_type_rows)}",
            flush=True,
        )

    print(f"wrote csv={csv_path}", flush=True)
    print(f"plots_written={len(plot_paths)}", flush=True)


if __name__ == "__main__":
    # python -m runs.plot_hinted_stop_reasons_vs_hint_all_models --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word
    main()
