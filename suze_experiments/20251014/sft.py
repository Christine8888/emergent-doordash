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
    # train parameters
    num_train_epochs: int
    max_seq_length: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    save_steps: int
    save_total_limit: int
    fp16: bool
    bf16: bool
    tf32: bool

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
        tokenizer.padding_side = "right" # ensures proper causal masking
        model.resize_token_embeddings(len(tokenizer))
    
    # Save initial model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"saved initial model and tokenizer to {save_dir}")

    # check that reasoning traces dataset exists
    if not os.path.exists(cfg.sft_dataset_path):
        raise FileNotFoundError(f"reasoning traces dataset not found at location {cfg.sft_dataset_path}")
    
    # --- load reasoning traces dataset ---
    # Convert each multi-turn record into a single text string.
    # The dataset stores {"messages": [{"role": ..., "content": ...}, ...]}.
    # We flatten this to:
    #     User: <message>
    #     Assistant: <response>
    # so SFTTrainer sees one "text" field. The collator later masks user text so
    # the model learns only from assistant responses.
    def format_conversation(example):
        messages = example["messages"]
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "user":
                conversation += f"User: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n"
        return {"text": conversation.strip()}
    
    train_dataset = load_dataset("json", data_files=cfg.sft_dataset_path, split="train")
    train_dataset = train_dataset.map(format_conversation)
    print(f"loaded reasoning traces dataset with {len(train_dataset)} examples")

    # TODO: not sure how to do eval from christine while training; let's start without it

    # --- Define collator to mask out user text ---
    response_template = "\nAssistant:"  # match conversation format
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # --- Training configuration ---
    training_args = SFTConfig(
        # minimal requirements
        output_dir=save_dir,
        run_name=cfg.experiment_name,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        max_seq_length=cfg.max_seq_length,
        seed=cfg.seed,
        report_to=["wandb"],
        save_strategy="steps",
        save_steps=cfg.save_steps,
        # nice to have
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        # NOTE: leaving out any eval related setup
    )

    # --- Initialize trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=training_args,
        data_collator=collator,
        max_seq_length=cfg.max_seq_length,
    )

    trainer.train()

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


# TODO: check that data collator masks correctly
# TODO: debug code 