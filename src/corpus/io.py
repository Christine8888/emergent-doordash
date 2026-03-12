"""I/O and naming helpers for corpus files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator
from zipfile import BadZipFile, ZipFile


BENCHMARKS_DIRNAME = "benchmarks"
EVAL_MANIFEST_FILENAME = "eval_manifest.jsonl"
ROLLOUTS_FILENAME = "rollouts.jsonl"
LOGGED_GRADER_RESULTS_FILENAME = "grader_results_logged.jsonl"
REGRADED_DIRNAME = "grader_results_regraded"
RUN_SUMMARY_INGEST_FILENAME = "run_summary_ingest.json"
RUN_SUMMARY_REGRADE_FILENAME = "run_summary_regrade.json"
BENCHMARK_INDEX_FILENAME = "benchmark_index.json"


def jsonl_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str) + "\n"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def benchmark_name_from_eval(eval_obj: dict[str, Any]) -> str:
    name = eval_obj.get("task") or eval_obj.get("task_id") or eval_obj.get("task_display_name")
    if name is None:
        return "unknown"
    name = str(name).strip()
    return name if name else "unknown"


def benchmark_slug(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("._-")
    return slug or "unknown"


def extract_scorer_names(eval_obj: dict[str, Any]) -> list[str]:
    scorers = eval_obj.get("scorers")
    names: list[str] = []
    if isinstance(scorers, list):
        for scorer in scorers:
            if isinstance(scorer, str):
                names.append(scorer)
            elif isinstance(scorer, dict):
                name = scorer.get("name") or scorer.get("scorer") or scorer.get("id")
                if name is not None:
                    names.append(str(name))
    return names


def to_dict(x):
    """Convert pydantic/dataclass-ish objects to plain dict when possible."""
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x


def as_text(content) -> str:
    """Convert inspect message content payloads to plain text."""
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


def sample_to_record(sample) -> dict:
    """Convert an Inspect sample object into a script-friendly dictionary."""
    prompt = None
    if getattr(sample, "messages", None):
        prompt = as_text(sample.messages[0].content)

    output = None
    if getattr(sample, "output", None) and getattr(sample.output, "choices", None):
        output = as_text(sample.output.choices[0].message.content)

    scores = getattr(sample, "scores", None)
    if scores is not None:
        if isinstance(scores, dict):
            scores = {k: to_dict(v) for k, v in scores.items()}
        else:
            scores = to_dict(scores)

    return {
        "id": getattr(sample, "id", None),
        "epoch": getattr(sample, "epoch", None),
        "target": getattr(sample, "target", None),
        "sample_idx": getattr(sample, "sample_idx", None),
        "prompt": prompt,
        "output": output,
        "scores": scores,
    }


def read_eval_start(eval_path: Path) -> dict | None:
    """Read _journal/start.json from an .eval file."""
    try:
        with ZipFile(eval_path) as zf:
            return json.loads(zf.read("_journal/start.json"))
    except (BadZipFile, KeyError, json.JSONDecodeError, OSError):
        return None


def get_hint_fraction(eval_path: Path) -> float | None:
    """Best-effort hint fraction lookup from eval metadata."""
    start = read_eval_start(eval_path)
    if not start:
        return None
    return start.get("eval", {}).get("metadata", {}).get("hint_fraction")
