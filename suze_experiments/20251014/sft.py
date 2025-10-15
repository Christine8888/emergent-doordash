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
from finetune import GenerativeAccuracyCallback
from utils import zip_


# W&B configuration
WANDB_ENTITY = "suzevana"
WANDB_PROJECT = "emergent_doordash"

# Set defaults only if not already provided via environment
os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

@dataclass
class SFTArgs:
    pass


def train_sft(cfg: SFTArgs):
    pass


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
    # python suze_experiments/20251014/sft.py config=suze_experiments/20251014/sft.yaml
    main()
