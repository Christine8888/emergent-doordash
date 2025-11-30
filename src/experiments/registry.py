"""Registry of all available evals."""

from typing import Callable


def get_external_evals() -> dict[str, Callable]:
    """Get external evals from inspect_evals package."""
    from inspect_evals.mmlu import mmlu_0_shot, mmlu_5_shot
    from inspect_evals.ifeval import ifeval
    return {
        "mmlu_0_shot": mmlu_0_shot,
        "mmlu_5_shot": mmlu_5_shot,
        "ifeval": ifeval,
    }


def get_internal_evals() -> dict[str, Callable]:
    """Get internal evals from environments."""
    from environments.gpqa.gpqa import gpqa_diamond
    from environments.arc.arc import arc
    from environments.aime.aime import aime
    from environments.hle.hle import hle
    from environments.math.math import math as math_task
    return {
        "gpqa": gpqa_diamond,
        "arc": arc,
        "aime": aime,
        "hle": hle,
        "math": math_task,
    }


def get_all_evals() -> dict[str, Callable]:
    """Get all available evals (external + internal)."""
    evals = {}
    evals.update(get_external_evals())
    evals.update(get_internal_evals())
    return evals


def get_eval(name: str) -> Callable:
    """Get a specific eval by name."""
    all_evals = get_all_evals()
    if name not in all_evals:
        raise ValueError(f"Unknown eval: {name}. Available: {list(all_evals.keys())}")
    return all_evals[name]
