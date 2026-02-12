#!/usr/bin/env python3
"""Load every benchmark dataset and print the Sample structure for each.

Usage:
    python suze_experiments/20260209/inspect_benchmarks.py
"""
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from utils.setup import setup_env
setup_env()

ENVS = [
    "gpqa",
    "aime",
    "math",
    "math_level_5",
    "hle",
    # "arc",
    "hellaswag",
    "piqa",
    "mmlu_5_shot_cot",
    "bbh",
    "arc_challenge",
    "winogrande",
]


def describe_input(inp):
    """Return a concise description of sample.input."""
    if isinstance(inp, str):
        return f"str (len={len(inp)}, preview={inp})"
    if isinstance(inp, list):
        roles = []
        for m in inp:
            role = getattr(m, "role", "?")
            content = getattr(m, "content", "")
            roles.append(f"{role}({len(content)} chars)")
        return f"list[{len(inp)} msgs: {', '.join(roles)}]"
    return f"{type(inp).__name__}"


def main():
    for env_name in ENVS:
        print(f"\n{'='*80}")
        print(f"  {env_name.upper()}")
        print(f"{'='*80}")

        import importlib
        mod = importlib.import_module(f"environments.{env_name}.config")

        print(f"Loading dataset...")
        try:
            ds = mod.get_dataset()
            samples = list(ds)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue
        print(f"Total samples: {len(samples)}")

        if not samples:
            print("  (empty dataset)")
            continue

        # Inspect first sample
        s = samples[0]
        print(f"\n--- First sample fields ---")
        print(f"  id:        {s.id!r}")
        print(f"  input:     {describe_input(s.input)}")
        print(f"  target:    {s.target!r}")
        print(f"  choices:   {s.choices!r}" if hasattr(s, "choices") else "  choices:   (not present)")
        print(f"  metadata:  {s.metadata!r}" if hasattr(s, "metadata") and s.metadata else "  metadata:  (empty/none)")
        print(f"  sandbox:   {s.sandbox!r}" if hasattr(s, "sandbox") and s.sandbox else "  sandbox:   (none)")
        print(f"  files:     {s.files!r}" if hasattr(s, "files") and s.files else "  files:     (none)")
        print(f"  setup:     {s.setup!r}" if hasattr(s, "setup") and s.setup else "  setup:     (none)")

        # Check if choices are already in the input text
        choices = getattr(s, "choices", None)
        if choices:
            input_text = s.input if isinstance(s.input, str) else ""
            if isinstance(s.input, list):
                input_text = " ".join(
                    getattr(m, "content", "") for m in s.input
                )
            choices_in_input = sum(1 for c in choices if c in input_text)
            print(f"\n  Choices in input text: {choices_in_input}/{len(choices)}")
            print(f"  First 3 choices: {choices[:3]}")

        # For list inputs, show message structure
        if isinstance(s.input, list):
            print(f"\n  --- Message structure ---")
            for i, m in enumerate(s.input):
                role = getattr(m, "role", "?")
                content = getattr(m, "content", "")
                preview = content[:100].replace("\n", "\\n")
                print(f"    [{i}] {role}: {preview!r}{'...' if len(content) > 100 else ''}")

        # Check what config exports
        print(f"\n  --- Config exports ---")
        for fn_name in ["get_dataset", "format_prompt", "extract_answer",
                        "grade_answer", "get_model_input", "extract_sample_fields",
                        "get_dataset_kwargs", "add_cli_args"]:
            has_it = hasattr(mod, fn_name)
            print(f"    {fn_name}: {'YES' if has_it else 'no'}")


if __name__ == "__main__":
    # python suze_experiments/20260209/inspect_benchmarks.py
    main()
