"""
RMSNorm implementation for Virtue
"""

import torch
import torch.nn as nn

class VirtueRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in Gemma and Virtue architectures
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Compute variance
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype)
