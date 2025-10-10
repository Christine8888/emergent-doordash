"""
Based on TRL callbacks from 
https://github.com/huggingface/trl/blob/main/trl/trainer/callbacks.py
"""
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import log_table_to_comet_experiment

from utils import zip_


def _generative_accuracy_completions_df(
    state: TrainerState, prompts: List[str], completions: List[str], is_correct: List[bool]
) -> pd.DataFrame:
    global_step = [str(state.global_step)] * len(prompts)
    data = list(zip(global_step, prompts, completions, is_correct))
    return pd.DataFrame(data, columns=["step", "prompt", "completion", "is_correct"])


def _generate_completions_with_eos(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    generation_config: Optional[GenerationConfig],
    batch_size: int = 1,
) -> list[str]:
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (list[str]): A list of input prompts for which completions are to be generated.
        model (PreTrainedModel): The pre-trained model to be used for generation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for encoding and decoding.
        accelerator (Accelerator): The accelerator to be used for model execution.
        generation_config (GenerationConfig): Configuration for text generation.
        batch_size (int, optional): The number of prompts to process in each batch. Default is 1.

    Returns:
        list[str]: A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx : idx + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
                eos_token_id=tokenizer.eos_token_id,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations):
                # Remove prompt from generation
                generation = generation[len(prompt) :]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)
                
    return completions


class GenerativeAccuracyCallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that draws samples from the model on a set of prompts and compares the generated samples to the ground truth.

    Usage:
    ```python
    trainer = DPOTrainer(...)
    generative_accuracy_callback = GenerativeAccuracyCallback(trainer=trainer)
    trainer.add_callback(generative_accuracy_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        eval_dataset (`datasets.Dataset` or `Dict[str, datasets.Dataset]`):
            The evaluation dataset to use for generating completions. Must include a `"prompt"` and `"completion"` column.
        answer_grader_fn (callable):
            A function that takes two strings and returns a boolean indicating whether the generated completion is correct.
            Usually involves extracting the answer from the generated completion and comparing it to the ground truth.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use for generating completions.
        num_prompts (`int` or `None`, *optional*, defaults to `None`):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples
            in the evaluation dataset.
    """

    def __init__(
        self,
        answer_grader_fn: Callable[[str, str], bool],
        eval_dataset: Union[Dataset, Dict[str, Dataset]],
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        num_prompts: Optional[int] = None,
    ):
        self.answer_grader_fn = answer_grader_fn
        self.trainer = trainer
        self.generation_config = generation_config
        self.eval_dataset = eval_dataset

        if num_prompts is not None:
            if isinstance(self.eval_dataset, dict):
                for dataset_name, dataset in self.eval_dataset.items():
                    self.eval_dataset[dataset_name] = dataset.select(range(num_prompts))
            else:
                self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def on_evaluate_helper(self, model, tokenizer, accelerator, args, state, eval_dataset, eval_dataset_name="eval"):
        with accelerator.split_between_processes(list(zip_(eval_dataset["prompt"], eval_dataset["completion"]))) as prompts_completions:
            prompts, completions = [elem[0] for elem in prompts_completions], [elem[1] for elem in prompts_completions]

            generated_completions = _generate_completions_with_eos(
                prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
            )

            completions = list(zip_(completions, generated_completions))

            is_correct = [self.answer_grader_fn(completion=completion, ground_truth=ground_truth) for ground_truth, completion in completions]
            is_correct = gather_object(is_correct)
            prompts = gather_object(prompts)
            completions = gather_object(completions)

        # Logging
        if self.trainer.accelerator.is_main_process:
            accuracy = sum(is_correct) / len(is_correct)
            self.trainer.log({f"eval/{eval_dataset_name}_accuracy": accuracy})
            
            if "wandb" in args.report_to:
                import wandb
                if wandb.run is not None:
                    df = _generative_accuracy_completions_df(
                        state=state,
                        prompts=prompts,
                        completions=completions,
                        is_correct=is_correct,
                    )
                    wandb.log({f"eval/{eval_dataset_name}_model_completions": wandb.Table(dataframe=df)})

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # At every evaluation step, we generate completions for the model and compare them with the ground truth.
        # Then we log this to the trainer.
        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        if isinstance(self.eval_dataset, dict):
            for dataset_name, dataset in self.eval_dataset.items():
                self.on_evaluate_helper(
                    model=model,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    args=args,
                    state=state,
                    eval_dataset=dataset,
                    eval_dataset_name=dataset_name,
                )
        else:
            self.on_evaluate_helper(
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                args=args,
                state=state,
                eval_dataset=self.eval_dataset,
                eval_dataset_name="eval",
            )