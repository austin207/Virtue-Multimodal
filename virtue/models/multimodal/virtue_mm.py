"""
Multimodal Virtue model combining text and vision
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..virtue_model import VirtueForCausalLM, VirtueConfig
from .vision_encoder import SigLIPVisionEncoder
from .mm_projector import MultimodalProjector

class VirtueMultimodalForCausalLM(nn.Module):
    """
    Multimodal Virtue model with vision and text capabilities
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text model (core Virtue)
        self.language_model = VirtueForCausalLM(config)
        
        # Vision encoder
        if config.use_multimodal:
            self.vision_tower = SigLIPVisionEncoder(config, freeze_encoder=True)
            
            # Multimodal projector
            self.mm_projector = MultimodalProjector(
                vision_hidden_size=config.vision_hidden_size,
                text_hidden_size=config.hidden_size,
                projector_type=config.mm_projector_type
            )
        else:
            self.vision_tower = None
            self.mm_projector = None
        
        # Special tokens for multimodal
        self.image_token_id = config.vocab_size - 4  # Reserve special tokens
        self.image_start_id = config.vocab_size - 3
        self.image_end_id = config.vocab_size - 2
        self.image_pad_id = config.vocab_size - 1
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # If images are provided, process them
        if images is not None and self.vision_tower is not None:
            inputs_embeds = self._merge_multimodal_embeddings(
                input_ids, images, attention_mask
            )
            input_ids = None  # Use embeddings instead
        
        # Forward through language model
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Merge text embeddings with image embeddings
        """
        
        # Get text embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Process images through vision encoder
        batch_size = images.shape[0]
        vision_features = []
        
        for i in range(batch_size):
            # Extract vision features
            img_features = self.vision_tower(images[i].unsqueeze(0))  # [1, num_patches, vision_dim]
            
            # Project to text space
            projected_features = self.mm_projector(img_features)  # [1, num_patches, text_dim]
            vision_features.append(projected_features.squeeze(0))
        
        # Merge text and vision embeddings
        merged_embeddings = self._insert_vision_embeddings(
            text_embeddings, vision_features, input_ids
        )
        
        return merged_embeddings
    
    def _insert_vision_embeddings(
        self,
        text_embeddings: torch.FloatTensor,
        vision_features: List[torch.FloatTensor],
        input_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Insert vision embeddings at image token positions
        """
        
        batch_size, seq_len, hidden_size = text_embeddings.shape
        new_embeddings = []
        
        for batch_idx in range(batch_size):
            # Find image token positions
            image_token_positions = (input_ids[batch_idx] == self.image_token_id).nonzero(as_tuple=True)[0]
            
            if len(image_token_positions) == 0:
                # No images in this sequence
                new_embeddings.append(text_embeddings[batch_idx])
            else:
                # Replace image tokens with vision features
                sequence_embeddings = []
                last_pos = 0
                
                for img_idx, img_pos in enumerate(image_token_positions):
                    # Add text before image
                    sequence_embeddings.append(text_embeddings[batch_idx, last_pos:img_pos])
                    
                    # Add vision features
                    if img_idx < len(vision_features):
                        sequence_embeddings.append(vision_features[img_idx])
                    
                    last_pos = img_pos + 1
                
                # Add remaining text
                if last_pos < seq_len:
                    sequence_embeddings.append(text_embeddings[batch_idx, last_pos:])
                
                # Concatenate all embeddings
                merged_sequence = torch.cat(sequence_embeddings, dim=0)
                new_embeddings.append(merged_sequence)
        
        # Pad to same length (simplified - production would handle this better)
        max_len = max(emb.shape[0] for emb in new_embeddings)
        padded_embeddings = []
        
        for emb in new_embeddings:
            if emb.shape[0] < max_len:
                padding = torch.zeros(max_len - emb.shape[0], hidden_size, device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=0)
            padded_embeddings.append(emb)
        
        return torch.stack(padded_embeddings, dim=0)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Delegate to language model
        return self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
    
    def generate(self, *args, **kwargs):
        # Delegate to language model's generation
        return self.language_model.generate(*args, **kwargs)
