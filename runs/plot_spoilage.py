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

    # ("basic_hint", "truncate_sentence"),
    # ("basic_hint", "truncate_word"),
    # ("basic_hint", "mask_sentence"),
    # ("basic_hint", "mask_word"),
    
    # ("bag_of_hints", "bag_count"),
]

FRACTIONS: list[float] = [round(i * 0.05, 2) for i in range(21)]


def _exact_answer_regex(target: str) -> str:
    target = target.strip()
    chunks = [chunk for chunk in re.split(r"\s+", target) if chunk]
    if not chunks:
        return ""
    pattern = r"\s+".join(re.escape(chunk) for chunk in chunks)
    if target[0].isalnum():
        pattern = r"(?<![A-Za-z0-9])" + pattern
    if target[-1].isalnum():
        pattern = pattern + r"(?![A-Za-z0-9])"
    return pattern


def _token_answer_regex(target: str) -> str:
    return r"(?<![A-Za-z0-9])" + re.escape(target) + r"(?![A-Za-z0-9])"


def _resolve_match_mode(match_mode: str, benchmark: str) -> str:
    if match_mode != "auto":
        return match_mode
    return "both" if benchmark == "hle" else "token"


def spoilage_matches(text: str, target: str, *, match_mode: str) -> dict[str, bool]:
    token_spoiled = False
    exact_regex_spoiled = False

    if match_mode in {"token", "both"}:
        pattern = _token_answer_regex(target)
        token_spoiled = bool(pattern and re.search(pattern, text))
    if match_mode in {"exact_regex", "both"}:
        pattern = _exact_answer_regex(target)
        exact_regex_spoiled = bool(pattern and re.search(pattern, text))
    if match_mode not in {"token", "exact_regex", "both"}:
        raise ValueError(f"Unsupported match_mode={match_mode!r}")

    return {
        "token": token_spoiled,
        "exact_regex": exact_regex_spoiled,
        "either": token_spoiled or exact_regex_spoiled,
        "both": token_spoiled and exact_regex_spoiled,
    }


def target_is_spoiled(text: str, target: str, *, match_mode: str) -> bool:
    return spoilage_matches(text, target, match_mode=match_mode)["either"]


def _primary_spoilage_rate_label(match_mode: str) -> str:
    if match_mode == "both":
        return "Either Matcher Spoilage Rate"
    if match_mode == "exact_regex":
        return "Exact Regex Spoilage Rate"
    return "Token Spoilage Rate"


def compute_spoilage_curve(
    rows: list[HintGenerationRecord],
    *,
    fractioner: str,
    fractions: list[float],
    match_mode: str,
) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    for fraction in fractions:
        token_spoiled = 0
        exact_regex_spoiled = 0
        either_spoiled = 0
        both_spoiled = 0
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
            matches = spoilage_matches(hint_text, target, match_mode=match_mode)
            if matches["token"]:
                token_spoiled += 1
            if matches["exact_regex"]:
                exact_regex_spoiled += 1
            if matches["either"]:
                either_spoiled += 1
            if matches["both"]:
                both_spoiled += 1

        either_rate = (either_spoiled / total) if total > 0 else 0.0
        token_rate = (token_spoiled / total) if total > 0 else 0.0
        exact_regex_rate = (exact_regex_spoiled / total) if total > 0 else 0.0
        both_rate = (both_spoiled / total) if total > 0 else 0.0
        points.append(
            {
                "fraction": float(fraction),
                "spoilage_rate": float(either_rate),
                "spoiled_count": int(either_spoiled),
                "token_spoilage_rate": float(token_rate),
                "token_spoiled_count": int(token_spoiled),
                "exact_regex_spoilage_rate": float(exact_regex_rate),
                "exact_regex_spoiled_count": int(exact_regex_spoiled),
                "either_spoilage_rate": float(either_rate),
                "either_spoiled_count": int(either_spoiled),
                "both_spoilage_rate": float(both_rate),
                "both_spoiled_count": int(both_spoiled),
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
    parser.add_argument("--hint-type", type=str, required=False, default=None, help="Optional hint type filter (e.g. answer_not_revealed).")
    parser.add_argument(
        "--fractioner",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Optional one-or-more fractioner filters (e.g. mask_word truncate_word).",
    )
    parser.add_argument(
        "--match-mode",
        choices=["auto", "token", "exact_regex", "both"],
        default="auto",
        help=(
            "Answer spoilage matcher. auto uses both token and exact_regex for --benchmark hle "
            "and token matching otherwise. exact_regex escapes the answer literally and "
            "allows whitespace in the answer to match any whitespace."
        ),
    )
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
        "match_mode": _resolve_match_mode(args.match_mode, args.benchmark),
        "combos": [],
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted_any = False
    y_max_global = 0.0

    selected_combos: list[tuple[str, str]] = HINT_FRACTIONER_COMBOS
    if args.hint_type is not None:
        selected_combos = [combo for combo in selected_combos if combo[0] == args.hint_type]
    if args.fractioner is not None:
        selected_fractioners = set(args.fractioner)
        selected_combos = [combo for combo in selected_combos if combo[1] in selected_fractioners]
    match_mode = _resolve_match_mode(args.match_mode, args.benchmark)

    for hint_type, fractioner in selected_combos:
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
            match_mode=match_mode,
        )
        x = [p["fraction"] for p in points]
        y = [p["spoilage_rate"] for p in points]
        n_hints = len(rows)
        label = f"{hint_type} + {fractioner} (n={n_hints})"
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=label)
        plotted_any = True
        if y:
            y_max_global = max(y_max_global, max(y))

        all_payload["combos"].append(
            {
                "hint_type": hint_type,
                "fractioner": fractioner,
                "hint_path": str(hint_path),
                "num_rows": len(rows),
                "match_mode": match_mode,
                "points": points,
            }
        )

        print(f"[spoilage_regex] combo={hint_type}+{fractioner} rows={len(rows)} match_mode={match_mode}")

    if not plotted_any:
        raise ValueError("No curves were plotted. Check benchmark and configured HINT_FRACTIONER_COMBOS.")

    ax.set_xlabel("Hint Fraction")
    ax.set_ylabel(_primary_spoilage_rate_label(match_mode))
    ax.set_title(f"Regex Spoilage Curves ({args.benchmark})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max_global * 1.05 if y_max_global > 0 else 1.0)
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
    # python -m runs.plot_spoilage --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word truncate_word
    # python -m runs.plot_spoilage --benchmark hle --hint-type answer_not_revealed --fractioner mask_word truncate_word
    main()
