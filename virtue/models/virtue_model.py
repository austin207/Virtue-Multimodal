"""
Main Virtue 270M Parameter Language Model
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .components.transformer import VirtueTransformer
from .components.embeddings import VirtueEmbeddings
from ..utils.model_utils import VirtueModelOutput

class VirtueConfig:
    """Configuration for Virtue 270M model"""
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=1024,
        intermediate_size=2736,
        num_hidden_layers=16,
        num_attention_heads=16,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        rope_base=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
        use_flash_attention=True,
        gradient_checkpointing=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.rope_base = rope_base
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.use_flash_attention = use_flash_attention
        self.gradient_checkpointing = gradient_checkpointing
        
        # Derived parameters
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Model info
        self.model_type = "virtue"
        self.architectures = ["VirtueForCausalLM"]

class VirtueForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Virtue 270M Causal Language Model
    
    Architecture based on Gemma 3 with:
    - 270M parameters
    - 32K context window 
    - RMSNorm + RoPE + SwiGLU
    - Flash Attention support
    """
    
    config_class = VirtueConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VirtueDecoderLayer"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Core transformer
        self.model = VirtueTransformer(config)
        
        # Language modeling head
        if config.tie_word_embeddings:
            self.lm_head = None  # Use tied embeddings
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Transformer forward pass
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        
        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = self.model.embed_tokens.weight @ hidden_states.transpose(-2, -1)
            logits = logits.transpose(-2, -1)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        # If inputs_embeds are passed, only use them in the first generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
