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

# from transformers.trainer_callback import EarlyStoppingCallback
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
    model_name_or_path: str
    save_dir: str
    run_name: str
    epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_len: int
    num_proc: int
    logging_steps: int
    eval_steps: int  # Also determines save_steps
    num_eval_samples: int
    fp16: bool
    bf16: bool
    tf32: bool
    seed: int
    eval_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    
    # Data
    train_dataset: str
    train_split: str
    eval_datasets: Optional[List[str]]
    eval_splits: Optional[List[str]]
    cache_dir: str

    # Generation config
    max_length: int  # Length of input prompt + max_new_tokens
    max_new_tokens: int
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float


"""Dataset preprocessing functions.

We need to end the prompt in precisely the `response_template_ids` that we pass to the `DataCollatorForCompletionOnlyLM`.
Then the label on which we backpropagate will be exactly the `completion` field below.
This is necessary because the formatted `prompt` field below are passed to the accuracy evaluation callback `GenerativeAccuracyCallback`.
"""

def prepare_math_local_train_dataset(example):
    return {
        "prompt": "# Question\n\n" + example["question"] + "\n\n# Solution",
        "completion": "\n\n" + example["response"] + " The answer is: " + example["target"],
    }

def prepare_math_cot_train_dataset(example):
    return {
        "prompt": "# Question\n\n" + example["problem"] + "\n\n# Solution",
        "completion": "\n\n" + example["solution"] + " The answer is: " + example["answer"],
    }

DATASET_TO_PREPROCESS_FN = {
    # math has multiple names so we just put them all in here
    "math": prepare_math_cot_train_dataset,
    "competition_math": prepare_math_cot_train_dataset,
    "MATH-lighteval": prepare_math_cot_train_dataset,
    "math_train.jsonl": prepare_math_local_train_dataset,
}

def prepare_dataset(dataset, dataset_name, num_proc): # TODO suze is here
    dataset = dataset.map(DATASET_TO_PREPROCESS_FN[dataset_name], num_proc=num_proc)
    return dataset

"""Prompt formatting and answer extraction functions."""


def formatting_prompts_func(example, eos_token=None):
    output_texts = []
    for prompt, response in zip(example['prompt'], example['completion']):
        formatted = prompt + response

        if eos_token is not None:
            formatted += eos_token

        output_texts.append(formatted)

    return output_texts


def parse_gsm8k_math_finetuned_answer(answer_str: str) -> str:
    """
    Parses the given answer string and extracts either:
    1. The expression following 'the answer is' or 'The answer is'
    2. The first number found in the text after 'the answer is'
    3. As a last resort, the last number found in the entire string

    Parameters:
        answer_str (str): The string containing the answer.

    Returns:
        str: The extracted expression/number, or an empty string if nothing is found.
    """
    # Find the last occurrence of "The answer is:"
    last_pos = -1
    for marker in ["The answer is:", "The answer is ", "the answer is:", "the answer is "]:
        pos = answer_str.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            start = pos + len(marker)

    if last_pos != -1:
        # Extract everything after that
        expression = answer_str[start:].strip()

        # First try to clean up the expression as before
        cleaned_expression = expression.rstrip(".!")
        
        # If the cleaned expression appears to be just a number, return it
        if re.match(r'^-?\d*\.?\d+$', cleaned_expression):
            return cleaned_expression
            
        # Otherwise, try to find the first number in the text
        number_match = re.search(r'-?\d*\.?\d+', cleaned_expression)
        if number_match:
            return number_match.group(0)

        return cleaned_expression
    
    # If no "answer is" phrase found, look for the last number in the entire string
    all_numbers = re.findall(r'-?\d*\.?\d+', answer_str)
    if all_numbers:
        return all_numbers[-1]
    
    return ""

def grade_answer(completion: str, ground_truth: str, answer_parser_fn: Callable) -> bool:
    return answer_parser_fn(completion) == answer_parser_fn(ground_truth)


DATASET_TO_ANSWER_GRADER_FN = {
    "math": partial(grade_answer, answer_parser_fn=parse_gsm8k_math_finetuned_answer),
    "competition_math": partial(grade_answer, answer_parser_fn=parse_gsm8k_math_finetuned_answer),
    "MATH-lighteval": partial(grade_answer, answer_parser_fn=parse_gsm8k_math_finetuned_answer),
    "math_train.jsonl": partial(grade_answer, answer_parser_fn=parse_gsm8k_math_finetuned_answer),
}

DATASET_TO_LOADER_FN = {
    "math_train.jsonl": lambda cfg: load_dataset("json", data_files="math_train.jsonl", split="train"),
    "competition_math": lambda cfg: load_dataset(cfg.train_dataset, split=cfg.train_split, cache_dir=cfg.cache_dir),
    "MATH-lighteval": lambda cfg: load_dataset(cfg.train_dataset, split=cfg.train_split, cache_dir=cfg.cache_dir),
}

def get_train_dataset_and_answer_grader_fn(cfg: SFTArgs):
    short_train_dataset_name = cfg.train_dataset.split("/")[-1]
    
    loader_fn = DATASET_TO_LOADER_FN[short_train_dataset_name]
    train_dataset = loader_fn(cfg)
    
    train_dataset = train_dataset.map(DATASET_TO_PREPROCESS_FN[short_train_dataset_name], num_proc=cfg.num_proc)
    # print one example
    print(f"one example of train_dataset: {train_dataset[0]}")

    # Drop cols that are not "prompt" or "completion"
    train_dataset = train_dataset.remove_columns(set(train_dataset.column_names) - {"prompt", "completion"})

    answer_grader_fn = DATASET_TO_ANSWER_GRADER_FN[short_train_dataset_name]

    return train_dataset, answer_grader_fn


def get_eval_datasets(cfg: SFTArgs):
    assert cfg.eval_splits is not None and len(cfg.eval_splits) == len(cfg.eval_datasets), "Must provide a split for each eval dataset"

    eval_datasets = {}
    eval_datasets_for_accuracy_callback = {}
    for eval_dataset_name, eval_split in zip_(cfg.eval_datasets, cfg.eval_splits):
        eval_dataset = load_dataset(eval_dataset_name, split=eval_split, cache_dir=cfg.cache_dir)
        short_eval_dataset_name = eval_dataset_name.split("/")[-1]
        eval_dataset = eval_dataset.map(DATASET_TO_PREPROCESS_FN[short_eval_dataset_name], num_proc=cfg.num_proc)
        eval_datasets[short_eval_dataset_name] = eval_dataset
        eval_datasets_for_accuracy_callback[short_eval_dataset_name] = copy.deepcopy(eval_dataset)

    return eval_datasets, eval_datasets_for_accuracy_callback


def get_data_collator(cfg: SFTArgs, tokenizer: AutoTokenizer):
    response_template_with_context = "\n\n# Solution"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    return collator


"""Training loop."""


def train_sft(cfg: SFTArgs):
    set_seed(cfg.seed)

    # Initialize tokenizer and model from local path with accelerate
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map={"": PartialState().process_index}
    )
    print(f"Model loaded from {cfg.model_name_or_path}")
    
    # Add padding token if needed
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Save initial model and tokenizer
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    # Get datasets; the train dataset determines the answer grader fn
    train_dataset, answer_grader_fn = get_train_dataset_and_answer_grader_fn(cfg)
    if cfg.eval_datasets is None:
        short_main_eval_dataset_name = None
        eval_datasets, eval_datasets_for_accuracy_callback = None, None
    else:
        eval_datasets, eval_datasets_for_accuracy_callback = get_eval_datasets(cfg)
        short_main_eval_dataset_name = cfg.eval_datasets[0].split("/")[-1]
    print(f"Loaded datasets {train_dataset} and {eval_datasets}")

    

    collator = get_data_collator(cfg, tokenizer)

    # Configure training
    training_args = SFTConfig(
        output_dir=cfg.save_dir,
        run_name=cfg.run_name,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        max_seq_length=cfg.max_seq_len,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        seed=cfg.seed,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        report_to=["wandb"],
        overwrite_output_dir=True,  # Overwrite existing checkpoints
        metric_for_best_model=f"eval_{short_main_eval_dataset_name}_loss" if short_main_eval_dataset_name is not None else "loss",
        greater_is_better=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        formatting_func=partial(formatting_prompts_func, eos_token=tokenizer.eos_token),
        data_collator=collator,
        args=training_args,
    )

    # Add callback for computing accuracy and log-likelihoods
    generation_config = GenerationConfig(
        max_length=cfg.max_length,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Add eval datasets to the GenerativeAccuracyCallback
    generative_accuracy_callback = GenerativeAccuracyCallback(
        answer_grader_fn=answer_grader_fn,
        eval_dataset=eval_datasets_for_accuracy_callback,
        trainer=trainer,
        generation_config=generation_config,
        num_prompts=cfg.num_eval_samples,
    )
    trainer.add_callback(generative_accuracy_callback)

    # Start training
    trainer.train()

def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgs

    @dataclass
    class LMTransformerArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate SFTArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
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
    # python suze_experiments/20251008/run_sft.py config=suze_experiments/20251008/suze_test.yaml
    main()