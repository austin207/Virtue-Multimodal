"""
Core transformer architecture for Virtue
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

from .attention import VirtueAttention
from .mlp import VirtueMLP
from .normalization import VirtueRMSNorm
from .embeddings import VirtueEmbeddings

class VirtueDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with pre-norm architecture
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Self-attention
        self.self_attn = VirtueAttention(config)
        
        # Feed-forward network
        self.mlp = VirtueMLP(config)
        
        # Layer norms (pre-norm architecture)
        self.input_layernorm = VirtueRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VirtueRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # Pre-norm for attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Pre-norm for MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed-forward network
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs

class VirtueTransformer(nn.Module):
    """
    Core transformer model for Virtue
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = VirtueEmbeddings(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            VirtueDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = VirtueRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        
        # Prepare attention mask for SDPA
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values
        )
        
        # Initialize hidden states
        hidden_states = inputs_embeds
        
        # Initialize caches
        if use_cache:
            if past_key_values is None:
                past_key_values = tuple([None] * len(self.layers))
            next_decoder_cache = ()
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # Pass through transformer layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values):
        # Create causal mask
        batch_size, seq_length = input_shape
        
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        
        # Create causal mask
        causal_mask = torch.full(
            (seq_length, seq_length + past_length), 
            fill_value=float("-inf"), 
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype
        )
        
        if seq_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=past_length + 1)
        
        # Expand for batch and heads
        causal_mask = causal_mask[None, None, :, :].expand(
            batch_size, 1, seq_length, seq_length + past_length
        )
        
        # Apply padding mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            causal_mask = causal_mask.masked_fill(attention_mask == float("-inf"), float("-inf"))
        
        return causal_mask
