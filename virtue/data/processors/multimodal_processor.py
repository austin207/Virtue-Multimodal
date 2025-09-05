# virtue/data/processors/multimodal_processor.py

"""
Combined processing for multimodal batches.
"""
from image_processor import ImageProcessor
from text_processor import TextProcessor
from typing import Any, Dict

class MultimodalProcessor:
    """
    Bundles text and image processors for end-to-end pipeline.
    """
    def __init__(
        self,
        text_processor: TextProcessor,
        image_processor: ImageProcessor,
        max_text_length: int = 512
    ):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_text_length = max_text_length
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        data dict with 'text' and 'image' entries.
        Returns tokenized text and processed image.
        """
        text = data["text"]
        image = data["image"]
        
        tokenized = self.text_processor(text, max_length=self.max_text_length)
        image_tensor = self.image_processor(image)
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "images": image_tensor,
            "labels": tokenized["input_ids"].squeeze(0).clone(),
        }
