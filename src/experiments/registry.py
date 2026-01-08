"""Registry of all available evals."""

from functools import partial
from typing import Callable


def get_external_evals() -> dict[str, Callable]:
    """Get external evals from inspect_evals package."""
    from inspect_evals.mmlu import mmlu_0_shot, mmlu_5_shot
    from inspect_evals.ifeval import ifeval
    from inspect_evals.mbpp import mbpp
    from inspect_evals.commonsense_qa import commonsense_qa
    from inspect_evals.arc import arc_easy, arc_challenge
    from inspect_evals.hellaswag import hellaswag
    from inspect_evals.piqa import piqa
    from inspect_evals.bbeh import bbeh
    from inspect_evals.bbh import bbh
    from inspect_evals.niah import niah

    return {
        "mmlu_0_shot": mmlu_0_shot,
        "mmlu_5_shot": mmlu_5_shot,
        # For models with thinking tokens (e.g., Qwen3), use cot=True to avoid max_tokens=5
        "mmlu_0_shot_cot": partial(mmlu_0_shot, cot=True),
        "mmlu_5_shot_cot": partial(mmlu_5_shot, cot=True),
        "ifeval": ifeval,
        "mbpp": mbpp,
        "commonsense_qa": commonsense_qa,
        "arc_easy": arc_easy,
        "arc_challenge": arc_challenge,
        "hellaswag": hellaswag,
        "piqa": piqa,
        "bbeh": bbeh,
        "bbh": bbh,
        "niah": partial(niah, min_context=4000, max_context=32000, n_contexts=50, n_positions=10),
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
        "math_level_5": partial(math_task, levels=[5]),
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
