"""
Feed-forward network with SwiGLU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VirtueMLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation
    Similar to Gemma's MLP structure
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU requires two linear layers for the gate
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) 
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function
        if config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = self._gelu_pytorch_tanh
        else:
            self.act_fn = getattr(F, config.hidden_act)
    
    def _gelu_pytorch_tanh(self, x):
        """
        PyTorch tanh-based GELU implementation
        More numerically stable than erf-based version
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def forward(self, x):
        # SwiGLU: gate_proj(x) * silu(up_proj(x))
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
