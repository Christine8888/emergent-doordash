from __future__ import annotations

from dataclasses import dataclass


SMALL_MODEL_CONSTRAINT = "80G|141G|40G|48G"
MEDIUM_MODEL_CONSTRAINT = "80G|141G|48G"
LARGE_MODEL_CONSTRAINT = "80G|141G"
GEMMA_12B_CONSTRAINT = "80G|141G"
H200_CONSTRAINT = "141G"


@dataclass(frozen=True)
class ModelSpec:
    path: str
    tp: int = 1
    account: str | None = None
    constraint: str | None = None
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

QWEN25_15B_SAMPLING = {
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


QWEN3_MODELS = [
    ModelSpec("Qwen/Qwen3-0.6B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-1.7B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-4B", constraint=SMALL_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-8B", constraint=MEDIUM_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-14B", constraint=LARGE_MODEL_CONSTRAINT, **QWEN3_SAMPLING),
    ModelSpec("Qwen/Qwen3-32B", constraint=H200_CONSTRAINT, **QWEN3_SAMPLING),
]

QWEN25_MODELS = [
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", constraint=SMALL_MODEL_CONSTRAINT, **QWEN25_15B_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", constraint=SMALL_MODEL_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", constraint=MEDIUM_MODEL_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", constraint=LARGE_MODEL_CONSTRAINT, **QWEN25_SAMPLING),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", constraint=H200_CONSTRAINT, **QWEN25_SAMPLING),
]

LLAMA_MODELS = [
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", constraint=MEDIUM_MODEL_CONSTRAINT, **LLAMA_SAMPLING),
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", tp=2, constraint=H200_CONSTRAINT, **LLAMA_SAMPLING),
]

GEMMA_MODELS = [
    ModelSpec("google/gemma-3-4b-it", constraint=SMALL_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-3-12b-it", constraint=GEMMA_12B_CONSTRAINT, **GEMMA_SAMPLING),
    ModelSpec("google/gemma-3-27b-it", tp=2, constraint=MEDIUM_MODEL_CONSTRAINT, **GEMMA_SAMPLING),
]

ALL_MODELS = QWEN3_MODELS + QWEN25_MODELS + LLAMA_MODELS + GEMMA_MODELS
ALL_MODEL_PATHS = [m.path for m in ALL_MODELS]


def get_model_spec(model_path: str) -> ModelSpec:
    for model in ALL_MODELS:
        if model.path == model_path:
            return model
    raise KeyError(f"Unknown model path: {model_path!r}")


def select_models(model: str, *, max_models: int | None = None) -> list[ModelSpec]:
    if model == "all":
        selected = list(ALL_MODELS)
    else:
        selected = [get_model_spec(model)]

    if max_models is not None:
        if max_models < 1:
            raise ValueError("max_models must be >= 1")
        selected = selected[:max_models]
    return selected
