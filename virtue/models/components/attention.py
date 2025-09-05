"""
Multi-head attention with RoPE and Flash Attention support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

from .embeddings import VirtueRotaryPositionalEncoding, apply_rotary_pos_emb

class VirtueAttention(nn.Module):
    """
    Multi-head attention with rotary positional embeddings
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)  
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary positional encoding
        self.rotary_emb = VirtueRotaryPositionalEncoding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_base,
        )
        
        # Flash attention
        self.use_flash_attention = config.use_flash_attention and FLASH_ATTENTION_AVAILABLE
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary positional embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Handle past key values for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Compute attention
        if self.use_flash_attention and attention_mask is None:
            # Use Flash Attention for efficiency
            attn_output = self._flash_attention_forward(query_states, key_states, value_states)
            attn_weights = None
        else:
            # Standard scaled dot-product attention
            attn_output, attn_weights = self._standard_attention_forward(
                query_states, key_states, value_states, attention_mask, output_attentions
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _flash_attention_forward(self, query_states, key_states, value_states):
        # Reshape for Flash Attention: (batch_size, seq_len, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2) 
        value_states = value_states.transpose(1, 2)
        
        attn_output = flash_attn_func(
            query_states, key_states, value_states, 
            dropout_p=0.0, softmax_scale=None, causal=True
        )
        
        return attn_output.transpose(1, 2)
    
    def _standard_attention_forward(self, query_states, key_states, value_states, attention_mask, output_attentions):
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Causal masking
        if attn_weights.size(-1) > 1:
            causal_mask = torch.triu(
                torch.full((attn_weights.size(-2), attn_weights.size(-1)), float("-inf")),
                diagonal=1
            ).to(attn_weights.device, attn_weights.dtype)
            attn_weights = attn_weights + causal_mask
        
        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        if output_attentions:
            return attn_output, attn_weights
        else:
            return attn_output, None
