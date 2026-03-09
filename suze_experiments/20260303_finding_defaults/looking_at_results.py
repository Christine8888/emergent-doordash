import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile, ZipFile
from inspect_ai.log import read_eval_log


# file that errored: 14767776_0 (Llama-3.1-8B-Instruct | hint=0.10)
# want to check; what were the really long samples?

# [03/09 11:42:23] WARNING: estimated input ~28073 tokens (86% of max_model_len=32768); only ~4695 tokens left for output sample_id='2011-II-3' epoch=10
# [03/09 11:42:23] WARNING: estimated input ~32123 tokens (98% of max_model_len=32768); only ~645 tokens left for output sample_id='2010-I-4' epoch=10

# christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Llama-3.1-8B-Instruct/2026-03-09T16-53-41+00-00_aime_GzpuYpZw6bmxazYANAuHsH.eval

EVAL_FILE = "christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Llama-3.1-8B-Instruct/2026-03-09T16-53-41+00-00_aime_GzpuYpZw6bmxazYANAuHsH.eval"
RESULTS_ROOT = "christine_experiments/20251113/results/aime/solution_intext_masked/0shot"


def to_dict(x):
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x


def print_sample(samples, id, epoch):
    matches = [s for s in samples if s.id == "2012-I-5" and s.epoch == 8]
    s = matches[0]
    # print(s.messages[0].content)                  # prompt
    print(s.output.choices[0].message.content)    # output


def _parse_iso(ts):
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _fmt_hint(hint):
    if hint is None:
        return "unknown"
    try:
        return f"{float(hint):.2f}"
    except (TypeError, ValueError):
        return str(hint)


def _eval_model_hint_and_created(eval_path: Path):
    try:
        with ZipFile(eval_path) as zf:
            start = json.loads(zf.read("_journal/start.json"))
    except (BadZipFile, KeyError, json.JSONDecodeError, OSError):
        return None

    eval_info = start.get("eval", {})
    model = eval_info.get("model", "unknown")
    hint = eval_info.get("metadata", {}).get("hint_fraction")
    created_at = _parse_iso(eval_info.get("created"))
    if created_at is None:
        return None
    return model, hint, created_at


def print_eval_completion_by_model_and_hint(
    results_root=RESULTS_ROOT,
    include_all_runs=False,
    progress_every=200,
    output_dir: Path | None = None,
):
    eval_files = sorted(Path(results_root).rglob("*.eval"))
    grouped = defaultdict(list)
    skipped = 0
    start_time = perf_counter()

    for idx, eval_file in enumerate(eval_files, start=1):
        parsed = _eval_model_hint_and_created(eval_file)
        if parsed is None:
            skipped += 1
            if progress_every and idx % progress_every == 0:
                elapsed = perf_counter() - start_time
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] {idx}/{len(eval_files)} scanned | "
                    f"skipped={skipped} | elapsed={elapsed:.1f}s | {rate:.1f} files/s"
                )
            continue
        model, hint, created_at = parsed
        grouped[(model, _fmt_hint(hint))].append((created_at, eval_file))
        if progress_every and idx % progress_every == 0:
            elapsed = perf_counter() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] {idx}/{len(eval_files)} scanned | "
                f"skipped={skipped} | elapsed={elapsed:.1f}s | {rate:.1f} files/s"
            )

    elapsed_total = perf_counter() - start_time
    print(
        f"[done] scanned={len(eval_files)} kept={len(eval_files)-skipped} "
        f"skipped={skipped} elapsed={elapsed_total:.1f}s"
    )

    grouped_lines: list[str] = []
    recency_rows: list[tuple[datetime, datetime, str, str, int]] = []
    for model, hint in sorted(grouped.keys()):
        runs = sorted(grouped[(model, hint)], key=lambda x: x[0])
        latest_time = runs[-1][0].isoformat()
        line = f"{model} | hint={hint} | latest_created={latest_time} | runs={len(runs)}"
        grouped_lines.append(line)
        print(line)
        recency_rows.append((runs[0][0], runs[-1][0], model, hint, len(runs)))
        if include_all_runs:
            for created_at, eval_file in runs:
                run_line = f"  - {created_at.isoformat()}  {eval_file}"
                grouped_lines.append(run_line)
                print(run_line)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_out = output_dir / "eval_completion_by_model_and_hint.txt"
    recency_out = output_dir / "eval_completion_by_model_and_hint_oldest_to_most_recent.txt"

    grouped_header = (
        f"[done] scanned={len(eval_files)} kept={len(eval_files)-skipped} "
        f"skipped={skipped} elapsed={elapsed_total:.1f}s"
    )
    grouped_out.write_text(grouped_header + "\n" + "\n".join(grouped_lines) + "\n")

    recency_lines: list[str] = [grouped_header]
    for first_created, latest_created, model, hint, n_runs in sorted(recency_rows, key=lambda r: r[1]):
        recency_lines.append(
            f"{model} | hint={hint} | first_created={first_created.isoformat()} "
            f"| latest_created={latest_created.isoformat()} | runs={n_runs}"
        )
    recency_out.write_text("\n".join(recency_lines) + "\n")
    print(f"Wrote: {grouped_out}")
    print(f"Wrote: {recency_out}")


def main():
    # python suze_experiments/20260303_finding_defaults/looking_at_results.py
    path = Path(EVAL_FILE)
    if not path.exists():
        raise FileNotFoundError(path)
    
    log = read_eval_log(str(path))
    samples = to_dict(log.samples)
    print_sample(samples=samples, id='2011-II-3', epoch=10)


    # this actually does not matter because only suze has eval files from suze, so yeah
    print_eval_completion_by_model_and_hint(include_all_runs=True)


if __name__ == "__main__":
    main()
