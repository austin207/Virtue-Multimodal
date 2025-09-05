"""
Utility functions for teacher model management
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer

def load_teacher_model(
    model_name: str = "google/gemma-3-4b-it",
    quantization: str = "4bit",
    device: str = "cuda",
) -> 'GemmaTeacher':
    """
    Load and initialize teacher model
    """
    from .gemma_teacher import GemmaTeacher
    
    teacher = GemmaTeacher(
        model_name=model_name,
        quantization=quantization,
        device_map=device,
        freeze=True,
    )
    
    print(f"Loaded teacher model: {model_name}")
    print(f"Quantization: {quantization}")
    print(f"Memory usage: {teacher.get_memory_usage()}")
    
    return teacher

def prepare_teacher_outputs(
    teacher_outputs: Dict[str, torch.Tensor],
    temperature: float = 4.0,
    top_k: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Process teacher model outputs for distillation
    """
    
    logits = teacher_outputs["logits"]
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Optional top-k filtering
    if top_k is not None:
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
        filtered_logits = torch.full_like(scaled_logits, float('-inf'))
        filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
        scaled_logits = filtered_logits
    
    # Generate soft targets
    soft_targets = F.softmax(scaled_logits, dim=-1)
    
    return {
        "soft_targets": soft_targets,
        "teacher_logits": logits,
        "scaled_logits": scaled_logits,
        "hidden_states": teacher_outputs.get("hidden_states"),
        "attentions": teacher_outputs.get("attentions"),
    }

def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute knowledge distillation loss
    """
    
    # Soft target loss (KL divergence)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(
        student_log_probs, 
        teacher_probs, 
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard target loss (Cross entropy)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    
    return total_loss, {
        "kl_loss": kl_loss,
        "ce_loss": ce_loss,
        "total_loss": total_loss,
    }

def align_tokenizers(
    teacher_tokenizer: AutoTokenizer,
    student_tokenizer: AutoTokenizer,
    text: str,
) -> Dict[str, Any]:
    """
    Handle tokenizer differences between teacher and student
    """
    
    # Tokenize with both tokenizers
    teacher_tokens = teacher_tokenizer(text, return_tensors="pt", padding=True)
    student_tokens = student_tokenizer(text, return_tensors="pt", padding=True)
    
    return {
        "teacher_input_ids": teacher_tokens["input_ids"],
        "teacher_attention_mask": teacher_tokens["attention_mask"],
        "student_input_ids": student_tokens["input_ids"], 
        "student_attention_mask": student_tokens["attention_mask"],
        "length_diff": teacher_tokens["input_ids"].shape[1] - student_tokens["input_ids"].shape[1],
    }

class TeacherOutputCache:
    """
    Cache teacher outputs to avoid recomputation
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Dict[str, torch.Tensor]):
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            lru_key = min(self.access_count.keys(), key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        self.cache.clear()
        self.access_count.clear()
