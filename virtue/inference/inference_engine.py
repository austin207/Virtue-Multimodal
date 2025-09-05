# virtue/inference/inference_engine.py

"""
Main inference engine for Virtue models.
"""

import torch
from typing import Optional, Dict, Any
from transformers import generation_utils

class InferenceEngine:
    """
    Wraps a Virtue or VirtueMultimodal model for inference.
    """
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt (with optional images).
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)
        
        if images is not None:
            inputs["images"] = images.to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=kwargs.get("max_length", self.max_length),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def stream_generate(
        self,
        prompt: str,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Stream generation (token by token).
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)
        
        if images is not None:
            inputs["images"] = images.to(self.device)
        
        # Use generate with return_dict_in_generate and output_scores
        outputs = self.model.generate(
            **inputs,
            max_length=kwargs.get("max_length", self.max_length),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        for token_id in outputs.sequences[0]:
            yield self.tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=True)
