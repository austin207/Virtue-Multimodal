"""
Token and position embeddings for Virtue
"""

import math
import torch
import torch.nn as nn
from typing import Optional

class VirtueEmbeddings(nn.Module):
    """
    Token embeddings with RoPE positional encoding
    """
    
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.config = config
        
    def forward(self, input_ids, reverse=False):
        if reverse:
            # For tied embeddings in output projection
            return nn.functional.linear(input_ids, self.embed_tokens.weight)
        else:
            return self.embed_tokens(input_ids)

class VirtueRotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cos/sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device, 
            dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x, seq_len=None):
        # x: [batch_size, num_heads, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary positional embedding to query and key tensors."""
    if position_ids is None:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    else:
        cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
