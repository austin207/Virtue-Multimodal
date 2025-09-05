# virtue/data/loaders/collator.py

"""
Batch collation for multimodal data.
"""

from typing import List, Dict, Any
import torch

class DataCollator:
    """
    Collate batch of dictionaries into batched tensors.
    """
    def __init__(self, pad_token_id: int = 0, image_pad: int = 0):
        self.pad_token_id = pad_token_id
        self.image_pad = image_pad
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        batch: list of dicts with keys 'input_ids','attention_mask','images','labels'
        Returns: dict of batched tensors
        """
        batch_input_ids = [item["input_ids"].squeeze(0) for item in batch]
        batch_attention = [item["attention_mask"].squeeze(0) for item in batch]
        batch_images = [item["images"] for item in batch]
        batch_labels = [item["labels"].squeeze(0) for item in batch]
        
        # Pad text
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        
        # Stack images (assumes same size)
        images = torch.stack(batch_images, dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "labels": labels
        }
