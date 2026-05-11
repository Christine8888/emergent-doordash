from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/nlp/scr/suzeva/tmp/matplotlib")

import matplotlib.pyplot as plt


DATA = Path("data")
OUT = Path("plots/accuracy_vs_hint/llama_7b_13b_hinted_preview_incomplete.png")
MODELS = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"]
BENCHMARKS = ["hle", "aime2025_2026"]
FRACTIONERS = ["mask_word", "truncate_word"]
HINT_TYPE = "answer_not_revealed"


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def frac(path: Path) -> float:
    return float(re.match(r"fraction_(.+)\.jsonl$", path.name).group(1))


def is_correct(row: dict) -> bool | None:
    for key in ("correct", "is_correct"):
        if isinstance(row.get(key), bool):
            return row[key]
    for grader in row.get("graders", []):
        if isinstance(grader, dict) and isinstance(grader.get("is_correct"), bool):
            return grader["is_correct"]
    return None


def stats(path: Path) -> dict | None:
    by_problem: dict[str, list[float]] = {}
    rows = known = 0
    for row in iter_jsonl(path):
        rows += 1
        problem_id = str(row.get("problem_id", "")).strip()
        correct = is_correct(row) if isinstance(row, dict) else None
        if not problem_id or correct is None:
            continue
        known += 1
        by_problem.setdefault(problem_id, []).append(float(correct))
    if not by_problem:
        return None

    problem_accs = [sum(values) / len(values) for values in by_problem.values()]
    mean = sum(problem_accs) / len(problem_accs)
    if len(problem_accs) > 1:
        var = sum((value - mean) ** 2 for value in problem_accs) / (len(problem_accs) - 1)
        half_width = 1.96 * math.sqrt(var / len(problem_accs))
    else:
        half_width = 0.0
    return {
        "accuracy": mean,
        "ci_low": max(0.0, mean - half_width),
        "ci_high": min(1.0, mean + half_width),
        "rows": rows,
        "known": known,
        "problems": len(problem_accs),
    }


def files_for(benchmark: str, model: str, fractioner: str) -> list[Path]:
    combo = f"{HINT_TYPE}__{fractioner}"
    root = DATA / ("hinted_grades" if benchmark == "hle" else "hinted_inference") / benchmark / model / combo
    if not root.exists() and benchmark == "hle":
        root = DATA / "hinted_inference" / benchmark / model / combo
    return sorted(root.glob("fraction_*.jsonl"), key=frac) if root.exists() else []


def main() -> None:
    fig, axes = plt.subplots(len(BENCHMARKS), len(FRACTIONERS), figsize=(10, 6), sharex=True, sharey=True)
    plotted = 0

    for row, benchmark in enumerate(BENCHMARKS):
        for col, fractioner in enumerate(FRACTIONERS):
            ax = axes[row][col]
            ax.set_title(f"{benchmark} / {fractioner}", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)

            for model in MODELS:
                points = []
                for path in files_for(benchmark, model, fractioner):
                    point = stats(path)
                    if point is None:
                        print(f"skip no labels: {path}")
                        continue
                    points.append((frac(path), point))
                    print(
                        f"{benchmark} {fractioner} {model} f={frac(path):.1f} "
                        f"acc={point['accuracy']:.3f} labeled={point['known']}/{point['rows']}"
                    )
                if not points:
                    print(f"skip missing/unlabeled: {benchmark} {fractioner} {model}")
                    continue

                xs = [x for x, _ in points]
                ys = [p["accuracy"] for _, p in points]
                lo = [p["ci_low"] for _, p in points]
                hi = [p["ci_high"] for _, p in points]
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=model.replace("Llama-2-", ""))
                ax.fill_between(xs, lo, hi, alpha=0.12)
                for x, point in points:
                    ax.annotate(
                        str(point["known"]),
                        (x, point["accuracy"]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                        fontsize=6,
                    )
                plotted += 1

            if row == len(BENCHMARKS) - 1:
                ax.set_xlabel("hint fraction")
            if col == 0:
                ax.set_ylabel("accuracy")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=8)

    fig.suptitle("Llama-2 hinted accuracy preview (incomplete fractions allowed)", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT} ({plotted} plotted series)")


if __name__ == "__main__":
    main()
