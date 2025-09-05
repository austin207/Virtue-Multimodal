# virtue/data/processors/text_processor.py

"""
Text tokenization and preprocessing.
"""

from transformers import AutoTokenizer
from typing import Dict, Any

class TextProcessor:
    """
    Wrapper around HuggingFace tokenizer.
    """
    def __init__(
        self,
        tokenizer_name: str = "google/gemma-3-4b-it",
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    def __call__(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, Any]:
        return self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=max_length,
            return_tensors=self.return_tensors
        )
