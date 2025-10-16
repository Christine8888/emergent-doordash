from dataclasses import dataclass
from typing import Optional
import os
import torch
import time
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import wandb

import datetime
from accelerate import PartialState

from utils import (
    set_seed,
    test_collator_masking,
    format_conversation,
    create_data_collator,
    compute_token_length_stats,
    recommend_max_seq_length_from_stats,
)


@dataclass
class SFTArgs:
    experiment_name: str
    append_timestamp: bool
    seed: int
    model_id: str
    revision: str
    model_dir: str
    base_dir: str
    wandb_entity: str
    wandb_project: str
    sft_dataset_path: str
    test_data_collator: bool
    # train parameters
    dataset_text_field: str
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
    enable_gradient_checkpointing: bool
    # distributed parameters
    dataloader_num_workers: int

def init_model_and_tokenizer(cfg: SFTArgs):
    assert torch.cuda.is_available()
    # determine process-local device
    state = PartialState()
    device = state.device

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
    )
    from_pretrained_kwargs = dict(
        revision=cfg.revision,
        cache_dir=cfg.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        **from_pretrained_kwargs,
    )
    model.to(device)
    wandb.termlog(f"loaded model {cfg.model_id} from {cfg.model_dir}")
    if cfg.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Ensure padding token and alignment of special tokens between tokenizer, model config, and generation config
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    # For causal LMs we want right padding to preserve autoregressive masking
    tokenizer.padding_side = "right"

    # Align model.config and generation_config special tokens with the tokenizer to avoid alignment warnings
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if getattr(tokenizer, "bos_token_id", None) is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.use_cache = False  # Always disable cache during training
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=False) # NOTE: i think this causes a crash



    return model, tokenizer




def train_sft(cfg: SFTArgs, save_dir: str):
    state = PartialState()
    is_main_process = state.is_main_process
    model, tokenizer = init_model_and_tokenizer(cfg)
    
    # Save initial model and tokenizer (main process only)
    if is_main_process:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        wandb.termlog(f"saved initial model and tokenizer to {save_dir}")

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
    # use shared formatter
    
    train_dataset = load_dataset("json", data_files=cfg.sft_dataset_path, split="train")
    train_dataset = train_dataset.map(format_conversation)
    wandb.termlog(f"loaded reasoning traces dataset with {len(train_dataset)} examples")

    if is_main_process:
        wandb.config.update({
            "model_id": cfg.model_id,
            "num_train_examples": len(train_dataset),
            "max_seq_length": cfg.max_seq_length,
            "batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
        })
        wandb.termlog(f"wandb config updated")

    # TODO: not sure how to do eval from christine while training; let's start without it

    # --- Define collator to mask out user text ---
    collator = create_data_collator(tokenizer)
    wandb.termlog(f"defined collator to mask out user text.")

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
        dataset_text_field=cfg.dataset_text_field,
        report_to=["wandb"],
        save_strategy="steps",
        save_steps=cfg.save_steps,
        # nice to have
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        optim="adamw_torch_fused",
        dataloader_num_workers=cfg.dataloader_num_workers,

        # NOTE: leaving out any eval related setup
    )

    # --- Initialize trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer, 
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
        # max_seq_length=cfg.max_seq_length,
    )
    if is_main_process:
        wandb.termlog(f"initialized trainer")

    # --- Train the model ---
    if is_main_process:
        wandb.termlog(f"Training the model for {cfg.num_train_epochs} epochs")
    # --- Resume from last checkpoint if available ---
    last_checkpoint = None
    if os.path.isdir(save_dir):
        checkpoints = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            if is_main_process:
                wandb.termlog(f"Found checkpoint to resume from: {last_checkpoint}")

    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    if is_main_process:
        wandb.termlog(f"Model trained and saved to {save_dir}")


def test_data_collator(cfg: SFTArgs, save_dir: str):
    set_seed(cfg.seed)
    state = PartialState()
    if not state.is_main_process:
        return
    # keep this small and on CPU to avoid multi-GPU overhead for a quick mask check
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    train_dataset = load_dataset("json", data_files=cfg.sft_dataset_path, split="train")
    wandb.termlog(f"Formatting reasoning traces dataset with {len(train_dataset)} examples")
    train_dataset = train_dataset.map(format_conversation)
    wandb.termlog(f"loaded reasoning traces dataset with {len(train_dataset)} examples")

    # ---- Determining a good max_seq_length ----
    if False:
        model_context_limit = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
        length_stats = compute_token_length_stats(
            train_dataset,
            tokenizer,
            text_field="text",
            sample_size=5000,
            batch_size=256,
            add_special_tokens=True,
        )
        recommended_msl, msl_meta = recommend_max_seq_length_from_stats(length_stats, model_context_limit, target_percentile=95)
        wandb.termlog(
            f"token lengths stats (sampled {length_stats['count']}): min={length_stats['min']} mean={length_stats['mean']:.1f} "
            f"p95={length_stats['percentiles']['p95']} p99={length_stats['percentiles']['p99']} max={length_stats['max']}"
        )
        wandb.termlog(
            f"model context limit: {model_context_limit}; recommended max_seq_length: {recommended_msl} "
            f"(p{msl_meta['target_percentile']}={msl_meta['chosen_from_data']}, overflow={msl_meta['overflow_fraction_at_recommended']:.3f})"
        )
        wandb.termlog("="*100)

    # ---- Testing the data collator ----
    collator = create_data_collator(tokenizer)
    test_collator_masking(train_dataset, tokenizer, collator, cfg)

def main():
    start_time = time.time()
    state = PartialState()
    is_main_process = state.is_main_process
    if is_main_process:
        wandb.termlog(f"Starting main at {start_time}")
        wandb.termlog("="*100)
    cli_args = OmegaConf.from_cli()
    if is_main_process:
        wandb.termlog(str(cli_args))
    file_cfg = OmegaConf.load(cli_args["config"])
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args["config"]

    default_cfg = OmegaConf.structured(SFTArgs(**file_cfg))
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)

    if cfg.append_timestamp:
        cfg.experiment_name = f"{cfg.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    if is_main_process:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.experiment_name,
            settings=wandb.Settings()
        )
        wandb.termlog(f"wandb entity: {cfg.wandb_entity}, wandb project: {cfg.wandb_project}")

    set_seed(cfg.seed)

    
    save_dir = os.path.join(cfg.base_dir, cfg.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Allow analyze-only runs: python sft.py config=... analyze=true
    if getattr(cfg, "test_data_collator", False):
        test_data_collator(cfg, save_dir)

    train_sft(cfg, save_dir)

    end_time = time.time()
    if is_main_process:
        wandb.termlog(f"Ending main at {end_time}")
        wandb.termlog(f"Total time: {end_time - start_time:.2f} seconds")
        wandb.termlog("="*100)


if __name__ == "__main__":
    # python suze_experiments/20251014/sft.py config=suze_experiments/20251014/test.yaml
    main()