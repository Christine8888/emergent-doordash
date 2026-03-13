from __future__ import annotations

import os
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO
from collections import defaultdict
from zipfile import ZipFile, BadZipFile
from inspect_ai.log import read_eval_log

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------
# Settings
# --------------------------
EVAL_ROOTS = {
    # Suze results (local checkout)
    "suzeva": PROJECT_ROOT / "christine_experiments/20251113",
    # Christine results (external path)
    "christine": Path("/sphinx/u/cye/emergent-doordash/christine_experiments/20251113"),
}

BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "corpus"
OVERWRITE_OUTPUTS = False
MAX_EVAL_FILES: int | None = None  # Set small int for quick testing.
PROGRESS_EVERY_EVALS = 100
NUM_SHARDS = 1
SHARD_ID = 0



@dataclass(frozen=True)
class IngestConfig:
    project_root: Path
    eval_roots: dict[str, Path]
    output_dir: Path
    overwrite_outputs: bool = True
    max_eval_files: int | None = None
    progress_every_evals: int = 10
    shard_id: int = 0
    num_shards: int = 1

def ts_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def discover_eval_files(config: IngestConfig) -> list[tuple[str, Path, int]]:
    found: list[tuple[str, Path, int]] = []
    owner_totals: dict[str, int] = {}
    for owner, root in config.eval_roots.items():
        print(f"[{ts_now()}] scanning root: owner={owner} root={root}", flush=True)
        owner_count = 0
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".eval"):
                    eval_path = Path(dirpath) / filename
                    size_in_bytes = eval_path.stat().st_size
                    found.append((owner, eval_path, size_in_bytes))
                    owner_count += 1
                    if owner_count % 5000 == 0:
                        print(
                            f"[{ts_now()}] discovered {owner_count} eval files for owner={owner}",
                            flush=True,
                        )
        print(f"[{ts_now()}] finished scan: owner={owner} eval_files={owner_count}", flush=True)
        owner_totals[owner] = owner_count

    print(f"[{ts_now()}] discovery summary:", flush=True)
    for owner in sorted(owner_totals.keys()):
        print(f"  {owner}: {owner_totals[owner]}", flush=True)
    print(f"  total: {sum(owner_totals.values())}", flush=True)

    found.sort(key=lambda x: (x[0], str(x[1])))

    if config.max_eval_files is not None:
        print(
            f"[{ts_now()}] applying max_eval_files={config.max_eval_files} -> "
            f"processing {min(len(found), config.max_eval_files)}",
            flush=True,
        )
        found = found[: config.max_eval_files]
    return found

def human_size(n: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024 or unit == "TB":
                return f"{n:.1f} {unit}"
            n /= 1024

def as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", item)))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)

def summarize_relevant_fields_from_read_eval_log(eval_path: Path) -> tuple[float, float, dict[str, Any]]:
    t0 = time.time()
    log = read_eval_log(str(eval_path))
    parse_elapsed = time.time() - t0

    t1 = time.time()
    sample_count = 0
    scored_count = 0
    prompt_chars = 0
    output_chars = 0
    score_entries = 0

    for sample in log.samples:
        sample_count += 1

        # prompt text
        prompt_text = None
        if getattr(sample, "messages", None):
            prompt_text = as_text(sample.messages[0].content)
        if prompt_text is not None:
            prompt_chars += len(prompt_text)

        # output text
        output_text = None
        if getattr(sample, "output", None) and getattr(sample.output, "choices", None):
            output_text = as_text(sample.output.choices[0].message.content)
        if output_text is not None:
            output_chars += len(output_text)

        # scores
        sample_scores = getattr(sample, "scores", None)
        if isinstance(sample_scores, dict) and len(sample_scores) > 0:
            scored_count += 1
            score_entries += len(sample_scores)

        # touch id/epoch/sample_idx/target to keep extraction workload comparable
        _ = getattr(sample, "id", None)
        _ = getattr(sample, "epoch", None)
        _ = getattr(sample, "sample_idx", None)
        _ = getattr(sample, "target", None)

    extract_elapsed = time.time() - t1
    return parse_elapsed, extract_elapsed, {
        "samples": sample_count,
        "scored_samples": scored_count,
        "score_entries": score_entries,
        "prompt_chars": prompt_chars,
        "output_chars": output_chars,
    }

def summarize_relevant_fields_from_zip(eval_path: Path) -> tuple[float, dict[str, Any]]:
    t0 = time.time()
    with ZipFile(eval_path) as zf:
        try:
            start_payload = json.loads(zf.read("_journal/start.json"))
        except KeyError:
            start_payload = None

        sample_names = [
            name
            for name in zf.namelist()
            if name.startswith("samples/") and name.endswith(".json")
        ]
        sample_names.sort()

        sample_count = 0
        scored_count = 0
        prompt_chars = 0
        output_chars = 0
        score_entries = 0

        for sample_name in sample_names:
            sample = json.loads(zf.read(sample_name))
            sample_count += 1

            # prompt text
            prompt_text = None
            messages = sample.get("messages")
            if isinstance(messages, list) and len(messages) > 0:
                first_message = messages[0]
                if isinstance(first_message, dict):
                    prompt_text = as_text(first_message.get("content"))
                else:
                    prompt_text = as_text(first_message)
            elif sample.get("input") is not None:
                prompt_text = as_text(sample.get("input"))
            if prompt_text is not None:
                prompt_chars += len(prompt_text)

            # output text
            output_text = None
            output = sample.get("output")
            if isinstance(output, dict):
                choices = output.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if content is not None:
                                output_text = as_text(content)
                if output_text is None and output.get("completion") is not None:
                    output_text = as_text(output.get("completion"))
            if output_text is not None:
                output_chars += len(output_text)

            scores = sample.get("scores")
            if isinstance(scores, dict) and len(scores) > 0:
                scored_count += 1
                score_entries += len(scores)

            # touch id/epoch/sample_idx/target to keep extraction workload comparable
            _ = sample.get("id")
            _ = sample.get("epoch")
            _ = sample.get("sample_idx")
            _ = sample.get("target")

    elapsed = time.time() - t0
    task_name = None
    if isinstance(start_payload, dict):
        eval_obj = start_payload.get("eval", {})
        if isinstance(eval_obj, dict):
            task_name = eval_obj.get("task")
    return elapsed, {
        "task_name": task_name,
        "samples": sample_count,
        "scored_samples": scored_count,
        "score_entries": score_entries,
        "prompt_chars": prompt_chars,
        "output_chars": output_chars,
    }

def load_file_time(eval_files):
    # Group by owner
    owner_to_files = defaultdict(list)
    for owner, eval_path, size_in_bytes in eval_files:
        owner_to_files[owner].append((owner, eval_path, size_in_bytes))

    for owner in ("christine", "suzeva"):
        files = owner_to_files[owner]
        # Sort by size for min/max
        files_sorted = sorted(files, key=lambda x: x[2])
        target_successes = min(5, len(files_sorted))
        selected: list[tuple[str, Path, int]] = []
        step = max(1, len(files_sorted) // target_successes)
        for i in range(0, len(files_sorted), step):
            selected.append(files_sorted[i])
            if len(selected) >= target_successes:
                break
        if files_sorted[-1] not in selected:
            selected.append(files_sorted[-1])

        # If a selected file is corrupted, continue with additional files.
        seen = set(selected)
        fallback = [f for f in files_sorted if f not in seen]
        candidates = selected + fallback

        successes = 0
        for owner, eval_path, size_in_bytes in candidates:
            try:
                log_parse_s, log_extract_s, log_summary = summarize_relevant_fields_from_read_eval_log(eval_path)
                zip_elapsed_s, zip_summary = summarize_relevant_fields_from_zip(eval_path)

                same_samples = log_summary["samples"] == zip_summary["samples"]
                same_scored = log_summary["scored_samples"] == zip_summary["scored_samples"]
                same_score_entries = log_summary["score_entries"] == zip_summary["score_entries"]

                print(
                    f"compare ({owner}) {eval_path} [{human_size(size_in_bytes)}]\n"
                    f"  read_eval_log: parse={log_parse_s:.2f}s extract_relevant={log_extract_s:.2f}s "
                    f"total={log_parse_s + log_extract_s:.2f}s\n"
                    f"  zip_direct:    extract_relevant={zip_elapsed_s:.2f}s\n"
                    f"  parity: samples={same_samples} scored_samples={same_scored} "
                    f"score_entries={same_score_entries}\n"
                    f"  counts: samples(log/zip)={log_summary['samples']}/{zip_summary['samples']} "
                    f"scored(log/zip)={log_summary['scored_samples']}/{zip_summary['scored_samples']} "
                    f"score_entries(log/zip)={log_summary['score_entries']}/{zip_summary['score_entries']}",
                    flush=True,
                )
                successes += 1
                if successes >= target_successes:
                    break
            except Exception as exc:
                print(f"skip corrupted/unreadable file ({owner}) {eval_path}: {exc}")


def main():

    config = IngestConfig(
        project_root=PROJECT_ROOT,
        eval_roots=EVAL_ROOTS,
        output_dir=BASE_OUTPUT_DIR,
        overwrite_outputs=OVERWRITE_OUTPUTS,
        max_eval_files=MAX_EVAL_FILES,
        progress_every_evals=PROGRESS_EVERY_EVALS,
        shard_id=SHARD_ID,
        num_shards=NUM_SHARDS,
    )

    # how many files are there?
    eval_files = discover_eval_files(config)
    # print(f"[{ts_now()}] total eval files discovered: {len(eval_files)}", flush=True)
    """
    christine: 5174
    suzeva: 16145
    total: 21319
    """
   

    # How big are files on average?
    # print(eval_files[0]) # ('christine', PosixPath('/sphinx/u/cye/emergent-doordash/christine_experiments/20251113/baseline/aime/Llama-3.1-70B-Instruct/2025-12-10T13-30-44-08-00_aime_esSEeeDniSQQxfSpESyrM6.eval'))
    sizes = [size_in_bytes for (_, _, size_in_bytes) in eval_files]  # (owner, eval_path, size_in_bytes)
    byte_size = sum(sizes) // len(sizes)
    print(f"average size: {byte_size} bytes ({human_size(byte_size)})")
    print(f"total size: {human_size(sum(sizes))}")
    """
    average size: 11593651 bytes (11.1 MB)
    total size: 230.5 GB
    """

    # how long does it take to load files?
    load_file_time(eval_files)
    

    """
    skip corrupted/unreadable file (christine) /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/baseline/mmlu_5_shot/Qwen2.5-0.5B-Instruct/2025-12-05T15-29-38-08-00_mmlu-5-shot_eqqYnREBFVC9j5JymoURDj.eval: EOCD not found
    read (christine) /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/gpqa/solution_intext_masked/0shot/Qwen2.5-14B-Instruct/2025-12-03T08-42-31-08-00_gpqa-diamond_UQEFYBzpPZa7utVv8VFEzg.eval [27.6 MB] in 16.61s
    read (christine) /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/gpqa/solution_intext_sequential/0shot/Llama-3.1-70B-Instruct/2026-01-16T07-39-27-08-00_gpqa-diamond_Vysn8EGzQKgqqK5a4bjpmK.eval [30.2 MB] in 15.55s
    read (christine) /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/hellaswag/solution_intext_masked/0shot/Qwen2.5-7B-Instruct/2026-03-09T08-54-15-07-00_hellaswag-task_J5ouDCCfWUSupZMrB3f5L9.eval [41.1 MB] in 21.56s
    read (christine) /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/mmlu_0_shot/solution_intext_sequential/0shot/Qwen3-1.7B/2026-03-01T22-15-12-08-00_mmlu-0-shot-task_fw3xR2qDv4WNvZvtPb54im.eval [55.1 MB] in 25.45s
    read (christine) /sphinx/u/cye/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_sequential/0shot/Qwen3-1.7B/2026-01-07T17-36-23-08-00_aime_gmApzo5HowZL9CjcB5yTRF.eval [418.7 MB] in 34.22s
    
    skip corrupted/unreadable file (suzeva) /juice5b/scr5b/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Llama-3.1-70B-Instruct/2026-02-27T21-04-12+00-00_aime_5qYNoKJCt2qBWZ8s8KDooo.eval: EOCD not found
    read (suzeva) /juice5b/scr5b/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Llama-3.1-70B-Instruct/2026-02-23T19-17-24+00-00_aime_WWGSNM5qGJb7Cc4pNqZJYA.eval [196.2 KB] in 0.20s
    read (suzeva) /juice5b/scr5b/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-14B/2026-02-20T13-28-39+00-00_aime_Dsg7kwp6v7yfqmVLEuw8FM.eval [420.4 KB] in 0.25s
    read (suzeva) /juice5b/scr5b/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-14B/2026-02-20T20-22-41+00-00_aime_gcwyjGs3wzrrs76jRzhJy3.eval [711.1 KB] in 0.25s
    read (suzeva) /juice5b/scr5b/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-32B/2026-02-23T16-51-10+00-00_aime_Sif2bhtD2NEgKdkk8uQiC7.eval [1.0 MB] in 0.32s
    read (suzeva) /juice5b/scr5b/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-0.6B/2026-02-15T22-02-12+00-00_aime_dCmxTY5AqhBFMZqLfQ7bbc.eval [253.3 MB] in 31.02s
    """

    # napkin math: 25s for 55mb; 230.5 GB/55mb * 25 s = 4 190 * 25 = 104,750s = 29 hours...


if __name__ == "__main__":
    # python suze_experiments/20260312/measure_problems.py
    main()
