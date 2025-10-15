import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_collator_masking(train_dataset, tokenizer, collator, num_examples=2):
    """Test the masking behavior of the data collator on actual training examples.
    
    Args:
        train_dataset: The training dataset containing text examples
        tokenizer: The tokenizer to use for encoding/decoding
        collator: The DataCollatorForCompletionOnlyLM instance
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