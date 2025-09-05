# virtue/data/datasets/text_dataset.py

"""
Text-only dataset for instruction fine-tuning.
"""

import json
from torch.utils.data import Dataset
from typing import List, Dict, Any

class TextDataset(Dataset):
    """
    Dataset for text-only instruction tasks.
    Expects a JSON lines file with entries: {"text": "..."}
    """
    def __init__(
        self,
        data_file: str,
        text_processor,
        max_length: int = 1024
    ):
        self.text_processor = text_processor
        self.max_length = max_length
        with open(data_file, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line)["text"] for line in f]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.examples[idx]
        tokenized = self.text_processor(text, max_length=self.max_length)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone(),
        }
