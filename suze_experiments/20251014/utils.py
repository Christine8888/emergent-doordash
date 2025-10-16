import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_collator_masking(train_dataset, tokenizer, collator, cfg, num_examples=1):
    """Test the masking behavior of the data collator on actual training examples.
    
    Args:
        train_dataset: The training dataset containing text examples
        tokenizer: The tokenizer to use for encoding/decoding
        collator: The DataCollatorForCompletionOnlyLM instance
        cfg: The configuration object
        num_examples: Number of examples to test (default: 2)
    """
    print("\n=== Verifying Collator Masking on Training Examples ===")
    
    # Get examples
    example_batch = train_dataset.select(range(num_examples))
    
    # Batch-tokenize so all sequences share the same length (avoids cat size mismatch)
    texts = [ex["text"] for ex in example_batch]
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_seq_length,
    )
    
    # Build a list of per-example feature dicts (what HF/TRL collators expect)
    features = [
        {
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
        }
        for i in range(tokenized["input_ids"].size(0))
    ]
    
    # Apply collator
    processed_batch = collator(features)
    
    # Show original and masked versions
    for idx, example in enumerate(example_batch):
        print(f"\nExample {idx + 1}:")
        print("\nOriginal text:")
        print(example["text"])
        
        print("\nTokenized and decoded with masks (padding shown as [PAD]):")
        masked_ids = processed_batch["input_ids"][idx]
        masked_mask = processed_batch["labels"][idx]
        
        # Convert -100 in labels back to pad_token_id for visualization
        vis_ids = torch.where(masked_mask == -100, 
                            tokenizer.pad_token_id, 
                            masked_ids)
        
        decoded = tokenizer.decode(vis_ids)
        print(decoded)
        print("\n" + "="*50)


def format_conversation(example: dict) -> dict[str, str]:
    # NOTE: if you change this, change create_data_collator too
    assert len(example["messages"]) == 2, "expected two messages in a conversation"
    assert example["messages"][0]["role"] == "user", "first message should be from user"
    assert example["messages"][1]["role"] == "assistant", "second message should be from assistant"
    # we don't accept multi turn yet
    user = example["messages"][0]["content"].strip()
    assistant = example["messages"][1]["content"].strip()
    text = f"User: {user}\nAssistant: {assistant}"
    return {"text": text}

def create_data_collator(tokenizer: AutoTokenizer):
    # NOTE: if you change this, change format_conversation too
    response_template = "Assistant:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        # NOTE: this will not work for multi-turn conversations
    )
    return collator


def _batch(iterable: List[str], batch_size: int) -> List[List[str]]:
    return [iterable[i : i + batch_size] for i in range(0, len(iterable), batch_size)]


def compute_token_length_stats(
    dataset,
    tokenizer,
    text_field: str = "text",
    sample_size: Optional[int] = 5000,
    batch_size: int = 256,
    add_special_tokens: bool = True,
) -> Dict[str, object]:
    n = len(dataset)
    if sample_size is not None and sample_size < n:
        # Fixed seed for reproducibility
        rng = np.random.default_rng(12345)
        indices = rng.choice(n, size=sample_size, replace=False)
        subset = dataset.select(indices.tolist())
    else:
        subset = dataset

    texts = [ex[text_field] for ex in subset]
    lengths: List[int] = []
    for chunk in _batch(texts, batch_size):
        enc = tokenizer(
            chunk,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
        # enc["input_ids"] is a list of lists
        lengths.extend(len(ids) for ids in enc["input_ids"])

    arr = np.array(lengths, dtype=np.int32)
    percentiles = {f"p{p}": int(np.percentile(arr, p)) for p in [50, 90, 95, 99]}
    stats: Dict[str, object] = {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size > 0 else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size > 0 else 0.0,
        "min": int(arr.min()) if arr.size > 0 else 0,
        "max": int(arr.max()) if arr.size > 0 else 0,
        "percentiles": percentiles,
        "lengths": lengths,  # keep for optional downstream checks
    }
    return stats


def recommend_max_seq_length_from_stats(
    stats: Dict[str, object],
    model_context_limit: Optional[int],
    target_percentile: int = 95,
) -> Tuple[int, Dict[str, object]]:
    chosen = int(stats["percentiles"][f"p{target_percentile}"])
    if model_context_limit is not None:
        recommended = min(chosen, int(model_context_limit))
    else:
        recommended = chosen

    meta = {
        "target_percentile": target_percentile,
        "chosen_from_data": chosen,
        "model_context_limit": int(model_context_limit) if model_context_limit is not None else None,
        "recommended": int(recommended),
        "overflow_fraction_at_recommended": float(
            np.mean(np.array(stats["lengths"]) > recommended) if len(stats["lengths"]) > 0 else 0.0
        ),
    }
    return recommended, meta