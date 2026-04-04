from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from src.hint_fractioners import fraction_hint
from src.storage import build_hint_generation_path, read_jsonl
from src.types import HintGenerationRecord


HINT_FRACTIONER_COMBOS: list[tuple[str, str]] = [
    ("answer_not_revealed", "truncate_sentence"),
    ("answer_not_revealed", "truncate_word"),
    ("answer_not_revealed", "mask_sentence"),
    ("answer_not_revealed", "mask_word"),

    ("basic_hint", "truncate_sentence"),
    ("basic_hint", "truncate_word"),
    ("basic_hint", "mask_sentence"),
    ("basic_hint", "mask_word"),
    
    ("bag_of_hints", "bag_count"),
]

FRACTIONS: list[float] = [round(i * 0.05, 2) for i in range(21)]


def target_is_spoiled(text: str, target: str) -> bool:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(target) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, text))


def compute_spoilage_curve(
    rows: list[HintGenerationRecord],
    *,
    fractioner: str,
    fractions: list[float],
) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    for fraction in fractions:
        spoiled = 0
        total = 0
        transform_errors = 0

        for row in rows:
            try:
                hint_text, _ = fraction_hint(
                    hint_record=row,
                    fractioner_name=fractioner,
                    hint_fraction=float(fraction),
                )
            except Exception:
                transform_errors += 1
                continue

            target = str(row.answer).strip()
            if not target:
                continue

            total += 1
            if target_is_spoiled(hint_text, target):
                spoiled += 1

        rate = (spoiled / total) if total > 0 else 0.0
        points.append(
            {
                "fraction": float(fraction),
                "spoilage_rate": float(rate),
                "spoiled_count": int(spoiled),
                "total_count": int(total),
                "transform_errors": int(transform_errors),
            }
        )
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot regex-based answer spoilage curves for configured hint/fractioner combos.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=False, default="data")
    parser.add_argument("--plots-root", type=str, required=False, default="plots")
    return parser


def _combo_key(hint_type: str, fractioner: str) -> str:
    return f"{hint_type}__{fractioner}".replace("/", "_")


def main() -> None:
    args = build_parser().parse_args()

    plots_dir = Path(args.plots_root) / "spoilage_regex"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plots_dir / f"{args.benchmark}__summary.json"
    plot_path_pdf = plots_dir / f"{args.benchmark}__spoilage_regex.pdf"

    all_payload: dict[str, object] = {
        "benchmark": args.benchmark,
        "fractions": FRACTIONS,
        "combos": [],
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted_any = False

    for hint_type, fractioner in HINT_FRACTIONER_COMBOS:
        hint_path = build_hint_generation_path(
            benchmark_name=args.benchmark,
            hint_type=hint_type,
            data_root=args.data_root,
        )
        rows = read_jsonl(hint_path, model_cls=HintGenerationRecord)
        if not rows:
            print(f"[spoilage_regex][WARN] skipping combo={hint_type}+{fractioner} missing_or_empty={hint_path}")
            continue

        points = compute_spoilage_curve(
            rows=rows,
            fractioner=fractioner,
            fractions=FRACTIONS,
        )
        x = [p["fraction"] for p in points]
        y = [p["spoilage_rate"] for p in points]
        label = f"{hint_type} + {fractioner}"
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=label)
        plotted_any = True

        all_payload["combos"].append(
            {
                "hint_type": hint_type,
                "fractioner": fractioner,
                "hint_path": str(hint_path),
                "num_rows": len(rows),
                "points": points,
            }
        )

        print(f"[spoilage_regex] combo={hint_type}+{fractioner} rows={len(rows)}")

    if not plotted_any:
        raise ValueError("No curves were plotted. Check benchmark and configured HINT_FRACTIONER_COMBOS.")

    ax.set_xlabel("Hint Fraction")
    ax.set_ylabel("Regex Spoilage Rate")
    ax.set_title(f"Regex Spoilage Curves ({args.benchmark})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path_pdf)
    plt.close(fig)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_payload, f, ensure_ascii=False, indent=2)

    print(f"[spoilage_regex] plot_pdf={plot_path_pdf}")
    print(f"[spoilage_regex] summary={summary_path}")


if __name__ == "__main__":
    # python -m runs.plot_spoilage_regex --benchmark aime2025_2026
    main()
