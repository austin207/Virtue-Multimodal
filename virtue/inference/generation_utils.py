# virtue/inference/generation_utils.py

"""
Utility functions for text generation.
"""

import torch
from typing import Optional

def pad_sequences(sequences, pad_token_id: int = 0):
    """
    Pad list of token lists to same length.
    """
    max_len = max(len(seq) for seq in sequences)
    padded = [
        seq + [pad_token_id] * (max_len - len(seq))
        for seq in sequences
    ]
    return torch.tensor(padded, dtype=torch.long)

def prepare_batch(
    tokenizer,
    prompts: list,
    images: Optional[torch.Tensor] = None,
    device: str = "cuda",
    max_length: int = 512
):
    """
    Tokenize prompts and optionally include images.
    """
    input_encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {
        "input_ids": input_encodings.input_ids.to(device),
        "attention_mask": input_encodings.attention_mask.to(device)
    }
    if images is not None:
        inputs["images"] = images.to(device)
    return inputs
