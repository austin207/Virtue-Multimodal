# configs/model_config.py

from dataclasses import dataclass, field


@dataclass
class VirtueConfig:
    """
    Architecture configuration for Virtue (270M parameters, 32K context, multimodal).
    Distilled from Gemma 3 4B-IT with SigLIP vision encoder.
    """
    # Text Transformer parameters
    vocab_size: int = 256000
    max_position_embeddings: int = 32768        # 32K token context
    hidden_size: int = 1024
    intermediate_size: int = 2736               # 2.67× expansion
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    head_dim: int = 64                           # hidden_size // num_attention_heads

    # Activation & normalization
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0                   # RoPE base frequency
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    # Vision encoder configuration (SigLIP)
    use_multimodal: bool = True
    vision_image_size: int = 896
    vision_patch_size: int = 14
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_hidden_layers: int = 12
    vision_num_attention_heads: int = 12

    # Multimodal fusion
    mm_projector_type: str = "linear"            # linear projection from vision → text
    mm_projector_dim: int = 1024                 # project to text hidden_size

    # Memory & performance
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Parameter counts (for sanity)
    @property
    def total_params(self) -> int:
        # approximate: embeddings + transformer + vision + projector
        embed_params = self.vocab_size * self.hidden_size
        trans_params = self.num_hidden_layers * (
            4 * self.hidden_size * self.hidden_size +
            2 * self.hidden_size * self.intermediate_size +
            2 * self.hidden_size
        )
        vision_params = (
            (self.vision_hidden_size * self.vision_patch_size**2 * 3) +  # patch embed
            self.vision_num_hidden_layers * (
                4 * self.vision_hidden_size * self.vision_hidden_size +
                2 * self.vision_hidden_size * self.vision_intermediate_size +
                2 * self.vision_hidden_size
            )
        )
        proj_params = self.vision_hidden_size * self.mm_projector_dim
        return embed_params + trans_params + vision_params + proj_params











