# virtue/evaluation/evaluator.py

"""
Main evaluation logic for Virtue.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .metrics.text_metrics import compute_text_metrics
from .metrics.vision_metrics import compute_vision_metrics
from .metrics.multimodal_metrics import compute_multimodal_metrics

class Evaluator:
    """
    Evaluator for text, vision, and multimodal tasks.
    """
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def evaluate_text(
        self,
        dataloader: DataLoader,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate text generation tasks.
        """
        references, predictions = [], []
        for batch in dataloader:
            inputs = batch["input_ids"].to(self.device)
            masks = batch["attention_mask"].to(self.device)
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=masks,
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            preds = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            refs = [self.tokenizer.decode(l, skip_special_tokens=True) for l in batch["labels"]]
            predictions.extend(preds)
            references.extend(refs)
        
        return compute_text_metrics(predictions, references)
    
    @torch.no_grad()
    def evaluate_vision(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate vision-only tasks (e.g., image classification).
        """
        images, labels = [], []
        preds = []
        for batch in dataloader:
            imgs = batch["images"].to(self.device)
            labs = batch["labels"].to(self.device)
            output = self.model.vision_tower(imgs)
            # assume model has a classification head attribute
            logits = self.model.mm_projector(output).mean(dim=1)
            pred = logits.argmax(dim=-1)
            preds.extend(pred.cpu().tolist())
            labels.extend(labs.cpu().tolist())
        
        return compute_vision_metrics(preds, labels)
    
    @torch.no_grad()
    def evaluate_multimodal(
        self,
        dataloader: DataLoader,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate multimodal tasks (e.g., VQA).
        """
        references, predictions = [], []
        for batch in dataloader:
            inputs = batch["input_ids"].to(self.device)
            masks = batch["attention_mask"].to(self.device)
            images = batch["images"].to(self.device)
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=masks,
                images=images,
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            preds = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            refs = batch.get("answers", [""]*len(preds))
            predictions.extend(preds)
            references.extend(refs)
        
        return compute_multimodal_metrics(predictions, references)
