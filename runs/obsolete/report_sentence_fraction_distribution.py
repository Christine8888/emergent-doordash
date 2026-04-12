from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
EXPANDED_HINT_ROOT = DEFAULT_DATA_ROOT / "expanded_hinted_prompts"
SENTENCE_FRACTIONERS = ("truncate_sentence", "mask_sentence")


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned or "unknown"


def _parse_fraction_from_filename(path: Path) -> float:
    match = re.match(r"^fraction_(.+)\.jsonl$", path.name)
    if match is None:
        raise ValueError(f"Unexpected fraction filename: {path.name}")
    return float(match.group(1))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _sentence_count(text: str) -> int:
    count = 0
    for match in re.finditer(r"[^.!?\n]+(?:[.!?](?=\s|$))?", text):
        if match.group(0).strip():
            count += 1
    return count


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None


def _resolve_hints_root(
    *,
    hints: str,
    benchmark_name: str | None,
    data_root: str | Path,
) -> Path:
    candidate = Path(hints)
    if candidate.exists():
        if candidate.is_file():
            return candidate.parent
        return candidate

    if benchmark_name is None:
        raise ValueError(
            "benchmark_name is required when hints is not an existing path. "
            "Expected hints to be either a path or a hint_type selector."
        )

    return (
        Path(data_root)
        / "expanded_hinted_prompts"
        / _safe_component(benchmark_name)
        / _safe_component(hints)
    )


def _resolve_fractioner_dirs(root: Path, requested_fractioners: list[str] | None) -> list[Path]:
    if root.is_file():
        root = root.parent

    requested = set(requested_fractioners or SENTENCE_FRACTIONERS)

    if root.name in requested:
        return [root]

    dirs: list[Path] = []
    for fractioner in SENTENCE_FRACTIONERS:
        if fractioner not in requested:
            continue
        candidate = root / fractioner
        if candidate.exists() and candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _counter_payload(counter: Counter[int], denominator: int) -> list[dict[str, Any]]:
    if denominator <= 0:
        return []
    return [
        {
            "sentences": sentence_count,
            "count": count,
            "pct": round(100.0 * count / denominator, 3),
        }
        for sentence_count, count in sorted(counter.items())
    ]


def _quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    lower = float(sorted_values[lower_idx])
    upper = float(sorted_values[upper_idx])
    weight = position - lower_idx
    return lower + (upper - lower) * weight


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def compute_sentence_fraction_distribution(
    *,
    hints: str,
    benchmark_name: str | None = None,
    fractioners: list[str] | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    resolved_root = _resolve_hints_root(
        hints=hints,
        benchmark_name=benchmark_name,
        data_root=data_root,
    )
    if not resolved_root.exists():
        raise ValueError(f"Expanded hints path does not exist: {resolved_root}")

    fractioner_dirs = _resolve_fractioner_dirs(resolved_root, fractioners)
    if not fractioner_dirs:
        raise ValueError(
            f"No sentence fractioner folders found under {resolved_root}. "
            f"Checked: {list(SENTENCE_FRACTIONERS)}"
        )

    fractioners_payload: dict[str, Any] = {}
    for fractioner_dir in sorted(fractioner_dirs):
        fraction_files = sorted(
            fractioner_dir.glob("fraction_*.jsonl"),
            key=_parse_fraction_from_filename,
        )
        if not fraction_files:
            continue

        unique_total_counter: Counter[int] = Counter()
        seen_hint_ids: set[str] = set()
        per_fraction_payload: list[dict[str, Any]] = []
        missing_metadata_rows = 0

        for fraction_file in fraction_files:
            fraction_value = _parse_fraction_from_filename(fraction_file)
            rows = _read_jsonl(fraction_file)

            visible_counter: Counter[int] = Counter()
            total_counter: Counter[int] = Counter()
            visible_values: list[int] = []
            total_values: list[int] = []

            for row in rows:
                fraction_meta = row.get("fraction_metadata")
                if not isinstance(fraction_meta, dict):
                    fraction_meta = {}

                total = _maybe_int(fraction_meta.get("units_total"))
                visible = _maybe_int(fraction_meta.get("units_visible"))
                masked = _maybe_int(fraction_meta.get("units_masked"))

                if total is None:
                    hint_obj = row.get("hint")
                    if isinstance(hint_obj, dict):
                        full_hint = str(hint_obj.get("full_hint", ""))
                        total = _sentence_count(full_hint)
                if visible is None and total is not None and masked is not None:
                    visible = total - masked

                if total is None or visible is None:
                    missing_metadata_rows += 1
                    continue

                total_counter[total] += 1
                visible_counter[visible] += 1
                visible_values.append(visible)
                total_values.append(total)

                hint_id = row.get("hint_id")
                if hint_id is None:
                    hint_obj = row.get("hint")
                    if isinstance(hint_obj, dict):
                        hint_id = hint_obj.get("hint_id")
                if hint_id is None:
                    hint_id = row.get("prompt_id")
                if isinstance(hint_id, str) and hint_id not in seen_hint_ids:
                    unique_total_counter[total] += 1
                    seen_hint_ids.add(hint_id)

            row_count = sum(visible_counter.values())
            per_fraction_payload.append(
                {
                    "hint_fraction": fraction_value,
                    "rows": row_count,
                    "mean_visible_sentences": _round_or_none(mean(visible_values)) if visible_values else None,
                    "min_visible_sentences": min(visible_values) if visible_values else None,
                    "max_visible_sentences": max(visible_values) if visible_values else None,
                    "mean_total_sentences": _round_or_none(mean(total_values)) if total_values else None,
                    "median_total_sentences": _round_or_none(median(total_values)) if total_values else None,
                    "p25_total_sentences": _round_or_none(_quantile(total_values, 0.25)),
                    "p75_total_sentences": _round_or_none(_quantile(total_values, 0.75)),
                    "min_total_sentences": min(total_values) if total_values else None,
                    "max_total_sentences": max(total_values) if total_values else None,
                    "visible_sentence_distribution": _counter_payload(visible_counter, row_count),
                    "total_sentence_distribution": _counter_payload(total_counter, row_count),
                    "source_file": str(fraction_file),
                }
            )

        unique_hint_count = sum(unique_total_counter.values())
        fractioners_payload[fractioner_dir.name] = {
            "source_dir": str(fractioner_dir),
            "unique_hints": unique_hint_count,
            "missing_metadata_rows": missing_metadata_rows,
            "original_sentence_distribution": _counter_payload(unique_total_counter, unique_hint_count),
            "fractions": per_fraction_payload,
        }

    if not fractioners_payload:
        raise ValueError(f"No fraction_*.jsonl files found under {resolved_root}")

    return {
        "benchmark_name": benchmark_name,
        "hints": hints,
        "resolved_hints_root": str(resolved_root),
        "fractioners": fractioners_payload,
    }


def _format_distribution_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"    sentences={int(row['sentences'])} count={int(row['count'])} pct={float(row['pct']):.1f}%"
        for row in rows
    ]


def print_sentence_fraction_distribution(report: dict[str, Any]) -> None:
    print(f"hints={report['hints']}")
    print(f"resolved_hints_root={report['resolved_hints_root']}")

    benchmark_name = report.get("benchmark_name")
    if benchmark_name:
        print(f"benchmark={benchmark_name}")

    for fractioner_name, payload in sorted(report["fractioners"].items()):
        print()
        print(f"fractioner={fractioner_name}")
        print(f"source_dir={payload['source_dir']}")
        print(f"unique_hints={payload['unique_hints']}")
        print(f"missing_metadata_rows={payload['missing_metadata_rows']}")

        fractions = sorted(payload["fractions"], key=lambda row: float(row["hint_fraction"]))
        fraction_one = next((row for row in fractions if abs(float(row["hint_fraction"]) - 1.0) < 1e-9), None)
        if fraction_one is not None:
            print("fraction_1_total_sentences:")
            print(f"  rows={fraction_one['rows']}")
            print(f"  mean={fraction_one['mean_total_sentences']}")
            print(f"  median={fraction_one['median_total_sentences']}")
            print(f"  p25={fraction_one['p25_total_sentences']}")
            print(f"  p75={fraction_one['p75_total_sentences']}")
            print(f"  min={fraction_one['min_total_sentences']}")
            print(f"  max={fraction_one['max_total_sentences']}")

        print("fraction_0.1_to_0.9_mean_visible_sentences:")
        for fraction_payload in fractions:
            fraction_value = float(fraction_payload["hint_fraction"])
            if fraction_value <= 0.0 or fraction_value >= 1.0:
                continue
            print(
                "  "
                f"fraction={fraction_value:.1f} "
                f"mean_visible_sentences={fraction_payload['mean_visible_sentences']}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report sentence-count distributions from expanded hinted prompt files for "
            "truncate_sentence and mask_sentence."
        )
    )
    parser.add_argument(
        "--hints",
        type=str,
        required=True,
        help=(
            "Hint selector or path. If this is a path, it can point to a hint root, "
            "fractioner dir, or fraction_*.jsonl file. Otherwise it is treated as a hint_type."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark name. Required when --hints is a hint_type selector rather than an existing path.",
    )
    parser.add_argument(
        "--fractioner",
        type=str,
        choices=["all", *SENTENCE_FRACTIONERS],
        default="all",
        help="Optional sentence fractioner filter.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Data root containing expanded_hinted_prompts.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    fractioners = None if args.fractioner == "all" else [args.fractioner]
    report = compute_sentence_fraction_distribution(
        hints=args.hints,
        benchmark_name=args.benchmark,
        fractioners=fractioners,
        data_root=args.data_root,
    )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_sentence_fraction_distribution(report)


if __name__ == "__main__":
    # python -m runs.report_sentence_fraction_distribution --benchmark aime2025_2026 --hints answer_not_revealed
    main()
