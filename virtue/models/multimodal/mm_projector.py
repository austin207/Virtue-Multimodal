"""
Multimodal projection layer to bridge vision and text
"""

import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    """
    Projects vision encoder outputs to text embedding space
    """
    
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projector_type: str = "linear"):
        super().__init__()
        self.projector_type = projector_type
        
        if projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, text_hidden_size)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, text_hidden_size),
                nn.GELU(),
                nn.Linear(text_hidden_size, text_hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch_size, num_patches, vision_hidden_size]
        Returns:
            projected_features: [batch_size, num_patches, text_hidden_size]
        """
        return self.projector(vision_features)
