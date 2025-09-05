# virtue/data/datasets/multimodal_dataset.py

"""
Vision-text multimodal dataset.
"""

import os
import json
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any

class MultimodalDataset(Dataset):
    """
    Dataset for vision+text tasks.
    Expects a JSON lines file with entries:
    {"image_path": "...", "text": "...", "label": ...}
    """
    def __init__(
        self,
        data_file: str,
        image_root: str,
        text_processor,
        image_processor,
        max_text_length: int = 512
    ):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line) for line in f]
        
        self.image_root = image_root
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_root, example["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor(image)
        
        # Process text
        text = example["text"]
        tokenized = self.text_processor(text, max_length=self.max_text_length)
        
        # Build output dict
        item = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": image_tensor,
        }
        if "label" in example:
            item["labels"] = tokenized["input_ids"].clone()
        return item
