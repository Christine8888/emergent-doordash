from __future__ import annotations

from dataclasses import dataclass


SMALL_MODEL_CONSTRAINT = "80G|141G|40G|48G"
MEDIUM_MODEL_CONSTRAINT = "80G|141G|48G"
LARGE_MODEL_CONSTRAINT = "80G|141G"
H200_CONSTRAINT = "141G"


@dataclass(frozen=True)
class ModelSpec:
    path: str
    tp: int = 1
    account: str | None = None
    constraint: str | None = None
    context_limit: int | None = None
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0

    @property
    def name(self) -> str:
        return self.path.split("/")[-1]

    @property
    def sampling_params(self) -> dict[str, bool | float | int]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }


QWEN3_SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
}

QWEN25_SMALL_SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.1,
}

QWEN25_SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
}

QWEN35_SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
}

LLAMA_SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

GEMMA_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "repetition_penalty": 1.0,
}

OPENAI_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
}


QWEN3_MODELS = [
    ModelSpec("Qwen/Qwen3-0.6B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-1.7B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-4B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-8B", constraint=MEDIUM_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-14B", constraint=LARGE_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-30B-A3B", constraint=H200_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-32B", constraint=H200_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-235B-A22B", tp=4, constraint=LARGE_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
]

QWEN25_MODELS = [
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", constraint=SMALL_MODEL_CONSTRAINT, **QWEN25_SMALL_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", constraint=SMALL_MODEL_CONSTRAINT, **QWEN25_SMALL_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", constraint=SMALL_MODEL_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", constraint=MEDIUM_MODEL_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", constraint=LARGE_MODEL_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", constraint=H200_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec(
        "Qwen/Qwen2.5-72B-Instruct",
        tp=2,
        constraint=H200_CONSTRAINT,
        context_limit=32768,
        **QWEN25_SAMPLING,
    ),
]

QWEN35_MODELS = [
    ModelSpec("Qwen/Qwen3.5-0.8B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-2B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-4B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-9B", constraint=MEDIUM_MODEL_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-27B", tp=2, constraint=MEDIUM_MODEL_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-35B-A3B", constraint=H200_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-122B-A10B", tp=2, constraint=H200_CONSTRAINT, **QWEN35_SAMPLING),
    ModelSpec("Qwen/Qwen3.5-397B-A17B", tp=4, constraint=LARGE_MODEL_CONSTRAINT, **QWEN35_SAMPLING),
]

LLAMA_MODELS = [
    ModelSpec(
        "meta-llama/Llama-2-7b-chat-hf",
        constraint=MEDIUM_MODEL_CONSTRAINT,
        context_limit=4096,
        **LLAMA_SAMPLING,
    ),
    ModelSpec(
        "meta-llama/Llama-2-13b-chat-hf",
        constraint=LARGE_MODEL_CONSTRAINT,
        context_limit=4096,
        **LLAMA_SAMPLING,
    ),
    ModelSpec(
        "meta-llama/Llama-2-70b-chat-hf",
        tp=2,
        constraint=H200_CONSTRAINT,
        context_limit=4096,
        **LLAMA_SAMPLING,
    ),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", constraint=MEDIUM_MODEL_CONSTRAINT, **LLAMA_SAMPLING),
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", tp=2, constraint=H200_CONSTRAINT, **LLAMA_SAMPLING),
    ModelSpec("meta-llama/Llama-3.3-70B-Instruct", tp=2, constraint=H200_CONSTRAINT, **LLAMA_SAMPLING),
]

GEMMA_MODELS = [
    ModelSpec("google/gemma-3-270m-it", constraint=SMALL_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-3-1b-it", constraint=SMALL_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-3-4b-it", constraint=SMALL_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-3-12b-it", constraint=LARGE_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-3-27b-it", tp=2, constraint=MEDIUM_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-4-E2B-it", constraint=SMALL_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-4-E4B-it", constraint=SMALL_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-4-31B-it", tp=2, constraint=MEDIUM_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-4-26B-A4B-it", tp=2, constraint=MEDIUM_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
]

OPENAI_MODELS = [
    ModelSpec("openai/gpt-oss-120b", tp=4, constraint=LARGE_MODEL_CONSTRAINT, **OPENAI_SAMPLING),
    ModelSpec("openai/gpt-oss-20b", tp=4, constraint=LARGE_MODEL_CONSTRAINT, **OPENAI_SAMPLING),
]

ALL_MODELS = QWEN3_MODELS + QWEN25_MODELS + LLAMA_MODELS + GEMMA_MODELS + OPENAI_MODELS + QWEN35_MODELS
ALL_MODEL_PATHS = [m.path for m in ALL_MODELS]

MASK_WORD_EXCLUDED_MODELS: set[str] = {
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    # "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
}
TRUNCATE_WORD_EXCLUDED_MODELS: set[str] = {
    # never use qwen 3.5 models
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    # "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
}

FRACTIONER_EXCLUDED_MODELS: dict[str, set[str]] = {
    "mask_word": MASK_WORD_EXCLUDED_MODELS,
    "truncate_word": TRUNCATE_WORD_EXCLUDED_MODELS,
}


def normalize_model_name(model: str) -> str:
    return str(model).strip().split("/")[-1]


def excluded_models_for_fractioner(fractioner: str | None) -> set[str]:
    if fractioner is None:
        return set()
    return set(FRACTIONER_EXCLUDED_MODELS.get(fractioner, set()))


def is_model_excluded_for_fractioner(model: str, fractioner: str | None) -> bool:
    excluded_model_names = {
        normalize_model_name(excluded_model)
        for excluded_model in excluded_models_for_fractioner(fractioner)
    }
    return normalize_model_name(model) in excluded_model_names


def models_excluded_from_selection(models: list[str], fractioner: str | None) -> list[str]:
    return sorted(
        model
        for model in models
        if is_model_excluded_for_fractioner(model, fractioner)
    )


def filter_models_for_fractioner(models: list[str], fractioner: str | None) -> list[str]:
    return [
        model
        for model in models
        if not is_model_excluded_for_fractioner(model, fractioner)
    ]


def filter_model_specs_for_fractioner(
    models: list[ModelSpec],
    fractioner: str | None,
) -> list[ModelSpec]:
    return [
        model
        for model in models
        if not is_model_excluded_for_fractioner(model.path, fractioner)
    ]


def get_model_spec(model_path: str) -> ModelSpec:
    for model in ALL_MODELS:
        if model.path == model_path:
            return model
    raise KeyError(f"Unknown model path: {model_path!r}")


def select_models(
    model: str,
    *,
    max_models: int | None = None,
    fractioner: str | None = None,
) -> list[ModelSpec]:
    if model == "all":
        selected = list(ALL_MODELS)
    else:
        excluded = models_excluded_from_selection([model], fractioner)
        if excluded:
            raise ValueError(
                f"Model {model!r} is excluded for fractioner={fractioner!r}"
            )
        selected = [get_model_spec(model)]

    selected = filter_model_specs_for_fractioner(selected, fractioner)

    if max_models is not None:
        if max_models < 1:
            raise ValueError("max_models must be >= 1")
        selected = selected[:max_models]
    return selected
