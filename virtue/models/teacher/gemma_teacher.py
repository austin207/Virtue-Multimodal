"""
Gemma 3 4B-IT Teacher Model Interface
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class GemmaTeacher(nn.Module):
    """
    Wrapper for Gemma 3 4B-IT teacher model with quantization support
    """
    
    def __init__(
        self, 
        model_name: str = "google/gemma-3-4b-it",
        quantization: str = "4bit",
        device_map: str = "cuda",
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.quantization = quantization
        self.device_map = device_map
        
        # Setup quantization config
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            quantization_config = None
        
        # Load teacher model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        self.is_frozen = freeze
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through teacher model
        """
        
        if self.is_frozen:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None,
        }
    
    def generate_soft_targets(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate soft targets for knowledge distillation
        """
        
        outputs = self.forward(input_ids, attention_mask, **kwargs)
        logits = outputs["logits"]
        
        # Apply temperature scaling
        soft_targets = torch.softmax(logits / temperature, dim=-1)
        
        return soft_targets
    
    @property
    def config(self):
        return self.model.config
    
    @property
    def vocab_size(self):
        return self.config.vocab_size
    
    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage of teacher model
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "quantization": self.quantization,
            }
        else:
            return {"error": "CUDA not available"}
