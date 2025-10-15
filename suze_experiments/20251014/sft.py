import copy
import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional

import os
import torch
from accelerate import PartialState
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.trainer_utils import set_seed
from trl import SFTConfig, SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
# from finetune import GenerativeAccuracyCallback
# from utils import zip_

# suze's utils file
from utils import set_seed


@dataclass
class SFTArgs:
    experiment_name: str
    seed: int
    model_id: str
    model_dir: str
    base_dir: str
    wandb_entity: Optional[str]
    wandb_project: Optional[str]
    sft_dataset_path: str

def train_sft(cfg: SFTArgs):
    if cfg.wandb_entity is not None and cfg.wandb_project is not None:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    set_seed(cfg.seed)
    save_dir = os.path.join(cfg.base_dir, cfg.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    def init_model_tokenizer():
        assert torch.cuda.is_available()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            cache_dir=cfg.model_dir,
        )
        print(f"loaded model {cfg.model_id} from {cfg.model_dir}")
        return model, tokenizer

    model, tokenizer = init_model_tokenizer()

    # Add padding token if needed
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    # Save initial model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"saved initial model and tokenizer to {save_dir}")

    # check that reasoning traces dataset exists
    if not os.path.exists(cfg.sft_dataset_path):
        raise FileNotFoundError(f"reasoning traces dataset not found at location {cfg.sft_dataset_path}")
    
    # load reasoning traces dataset
    reasoning_traces_dataset = load_dataset("json", data_files=cfg.sft_dataset_path, split="train")
    print(f"loaded reasoning traces dataset with {len(reasoning_traces_dataset)} examples")


    


def main():
    cli_args = OmegaConf.from_cli()
    print(cli_args)
    file_cfg = OmegaConf.load(cli_args["config"])
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args["config"]

    default_cfg = OmegaConf.structured(SFTArgs(**file_cfg))
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train_sft(cfg)


if __name__ == "__main__":
    # python suze_experiments/20251014/sft.py config=suze_experiments/20251014/test.yaml
    main()
