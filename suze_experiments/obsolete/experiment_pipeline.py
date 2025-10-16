"""Minimal experiment scaffold for steps 1–4.

Step 1 is implemented (load checkpoint). Steps 2–4 print TODOs.
"""

from pathlib import Path
import sys
from src.utils.setup import setup_env
from src.models import load_causal_lm

openlm_3b = [
    ("openlm-research/open_llama_3b", None),
    # ("openlm-research/open_llama_3b_step_2500", None),
    # ("openlm-research/open_llama_3b_step_5000", None),
    # ("openlm-research/open_llama_3b_step_7500", None),
    # ("openlm-research/open_llama_3b_step_10000", None),
    # ("openlm-research/open_llama_3b_step_50000", None),
    # ("openlm-research/open_llama_3b_step_100000", None),    
    # ("openlm-research/open_llama_3b_step_150000", None),
    # ("openlm-research/open_llama_3b_step_200000", None),
    # ("openlm-research/open_llama_3b_step_250000", None),
]

def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()
    setup_env()

    models_name_and_revisions = openlm_3b

    for model_name, revision in models_name_and_revisions:
        # Step 1: load checkpoint
        model, tokenizer = load_causal_lm(model_name, revision=revision)
        print({
            "model_id": model_name,
            "revision": revision,
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "vocab_size": tokenizer.vocab_size,
            "n_params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        })

        # Step 2: pretraining perplexity (placeholder)
        print({"step": 2, "status": "TODO: implement C4 validation perplexity"})

        # Step 3: SFT (placeholder)
        print({"step": 3, "status": "TODO: implement SFT"})

        # Step 4: downstream eval (placeholder)
        print({"step": 4, "status": "TODO: implement downstream eval (e.g., MATH)"})


if __name__ == "__main__":
    # python suze_experiments/20251008/experiment_pipeline.py
    main()


