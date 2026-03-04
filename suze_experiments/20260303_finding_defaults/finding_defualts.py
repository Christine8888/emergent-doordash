#!/usr/bin/env python3
"""
Very simple .eval inspector for generation defaults.
"""

import json
from pathlib import Path

from inspect_ai.log import read_eval_log


DEFAULT_EVAL_FILE = "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-0.6B/2026-02-21T13-06-20+00-00_aime_7UTGLmmaLMrND6oGn5QXMt.eval"
#"/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/aime/solution_intext_masked/0shot/Qwen3-0.6B/2026-02-14T00-42-28+00-00_aime_TZyzRjaCu3es2vy276y7DW.eval"
#^ this one has no results

def to_dict(x):
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return x


def main():
    # python suze_experiments/20260303_finding_defaults/finding_defualts.py
    path = Path(DEFAULT_EVAL_FILE)
    if not path.exists():
        raise FileNotFoundError(path)

    log = read_eval_log(str(path))

    out_path = Path(__file__).with_name(f"{path.stem}_dump.txt")

    chunks = []
    chunks.append(f"file: {path}")
    chunks.append(f"version: {getattr(log, 'version', None)}")
    chunks.append(f"status: {getattr(log, 'status', None)}")
    chunks.append(f"task: {getattr(log.eval, 'task', None)}")
    chunks.append(f"model: {getattr(log.eval, 'model', None)}")
    chunks.append(f"created: {getattr(log.eval, 'created', None)}")

    chunks.append("\nplan:")
    chunks.append(json.dumps(to_dict(log.plan), indent=2, default=str))

    chunks.append("\neval:")
    chunks.append(json.dumps(to_dict(log.eval), indent=2, default=str))

    chunks.append("\nresults:")
    chunks.append(json.dumps(to_dict(log.results), indent=2, default=str))

    chunks.append("\nstats:")
    chunks.append(json.dumps(to_dict(log.stats), indent=2, default=str))

    chunks.append("\none eval sample:")
    chunks.append(json.dumps(to_dict(log.samples[0]), indent=2, default=str))

    chunks.append("\none reduction:")
    chunks.append(json.dumps(to_dict(log.reductions[0]), indent=2, default=str))

    out_path.write_text("\n".join(chunks))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
