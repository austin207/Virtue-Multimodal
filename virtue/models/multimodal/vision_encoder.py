"""
SigLIP Vision Encoder for Virtue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import SiglipVisionModel, SiglipVisionConfig

class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP Vision Encoder wrapper
    Uses pre-trained SigLIP from Gemma 3 4B-IT
    """
    
    def __init__(self, config, freeze_encoder: bool = True):
        super().__init__()
        
        # SigLIP configuration
        vision_config = SiglipVisionConfig(
            hidden_size=config.vision_hidden_size,
            image_size=config.vision_image_size,
            patch_size=config.vision_patch_size,
            num_hidden_layers=config.vision_num_hidden_layers,
            num_attention_heads=config.vision_num_attention_heads,
            intermediate_size=config.vision_intermediate_size,
        )
        
        # Load pre-trained SigLIP
        self.vision_tower = SiglipVisionModel(vision_config)
        
        # Freeze parameters if requested
        if freeze_encoder:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
        
        self.config = config
        self.is_frozen = freeze_encoder
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, channels, height, width]
        Returns:
            vision_features: [batch_size, num_patches, hidden_size]
        """
        
        if self.is_frozen:
            with torch.no_grad():
                vision_outputs = self.vision_tower(images, output_hidden_states=False)
        else:
            vision_outputs = self.vision_tower(images, output_hidden_states=False)
        
        # Get the last hidden state (patch embeddings)
        vision_features = vision_outputs.last_hidden_state
        
        return vision_features
    
    @property
    def hidden_size(self):
        return self.config.vision_hidden_size
    
    @property
    def num_patches(self):
        return (self.config.vision_image_size // self.config.vision_patch_size) ** 2
