"""
Virtue: 270M Parameter Multimodal Language Model
Distilled from Gemma 3 4B-IT with SigLIP vision encoder
"""

__version__ = "0.1.0"

from .models.virtue_model import VirtueForCausalLM, VirtueConfig
from .models.multimodal.virtue_mm import VirtueMultimodalForCausalLM

__all__ = [
    "VirtueForCausalLM",
    "VirtueConfig", 
    "VirtueMultimodalForCausalLM",
]
