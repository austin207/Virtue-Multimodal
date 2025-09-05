# Architecture of Virtue Multimodal Model

## Overview
Virtue is a lightweight multimodal language model with 270 million parameters, distilled from Google’s Gemma 3 4B-IT. It supports a 32K token context window and integrates a pre-trained SigLIP vision encoder. Designed to run on a single 8 GB GPU, Virtue combines text and image understanding in a modular codebase.

## Key Design Goals

- **Efficiency:** 270M parameters, optimized for 8 GB VRAM.
- **Multimodal:** Native support for text and vision through SigLIP encoder.
- **Long Context:** 32K token context via Rotary Positional Embeddings (RoPE).
- **Distilled Quality:** Knowledge distilled from a 4 B parameter teacher model.
- **Modularity:** Clear separation of architecture components, training, inference, and utilities.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Text Transformer Backbone](#text-transformer-backbone)
3. [Vision Encoder Integration](#vision-encoder-integration)
4. [Multimodal Fusion](#multimodal-fusion)
5. [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
6. [Flash Attention & Efficiency](#flash-attention--efficiency)
7. [Knowledge Distillation Setup](#knowledge-distillation-setup)
8. [Model Configurations](#model-configurations)

---

## High-Level Architecture

Virtue comprises three primary modules:

1. **Text Transformer**: A 16-layer decoder-only transformer with 1,024 hidden size and 16 attention heads.
2. **Vision Encoder**: A frozen SigLIP vision tower with 12 layers, 768 hidden size, processing 896×896 images.
3. **Multimodal Projector**: A projection layer mapping vision features into text embedding space (1,024 dims).

The overall data flow:

```text
[Image] --SigLIP--> [Vision Features] --Projector--> [Text Embeddings]
                \                               /
                 \--Text Transformer Backbone--/

 Text tokens -> Embeddings -> Transformer -> LM Head -> Output logits
``` 

---

## Text Transformer Backbone

- **Embedding Layer**: Token embeddings of size 256 K×1,024; supports tied embeddings.
- **RoPE**: Rotary positional encoding for long context up to 32 K tokens.
- **Decoder Layers (16)**:
  - Pre-LN: RMSNorm before attention and MLP.
  - Attention: Multi-head scaled dot-product with optional Flash Attention.
  - MLP: SwiGLU feed-forward with intermediate size 2,736.
- **Output**: LM head projecting hidden states to vocabulary logits.

### Layer Structure

```text
Input Embeddings + RoPE → Norm → Self-Attn → Residual → Norm → MLP → Residual
``` 

---

## Vision Encoder Integration

- **SigLIP Model**: Pre-trained on large image corpus; frozen during distillation.
- **Configuration**:
  - Hidden size: 768
  - Patch size: 14 → 64×64 patches for 896×896 images
  - 12 transformer layers, 12 heads
- **Output**: Sequence of patch embeddings (num_patches×768).

---

## Multimodal Fusion

- **Vision → Text Projection**:
  - Linear projector: 768 → 1,024 dims.
  - Projects each patch embedding into text embedding space.
- **Embedding Merge**:
  - Special `<img>` tokens inserted in text sequence.
  - Replace these token positions with projected vision embeddings.
- **Result**: Combined sequence of text and image embeddings fed into text transformer.

---

## Rotary Positional Embeddings (RoPE)

- Provides continuous positional encoding without absolute embedding tables.
- Scales to 32 K tokens by computing sin/cos frequencies:
  
  \[
        ext{inv_freq}_k = 1 / (10000^{2k/	ext{dim}})
  \]

- Applies at attention time for queries and keys.

---

## Flash Attention & Efficiency

- **Flash Attention 2**: GPU kernel optimizing memory and compute in attention.
- **Fallback**: Standard PyTorch SDPA if kernel unavailable.
- **Quantization**: Teacher model loaded in 4-bit (Q4_0) to fit 8 GB VRAM.
- **Gradient Checkpointing**: Activates for transformer layers to save memory.

---

## Knowledge Distillation Setup

- **Teacher**: Gemma 3 4B-IT multimodal, frozen, quantized.
- **Student**: Virtue 270M multimodal.
- **Losses**:
  - `KL` on soft logits (temperature 4.0).
  - Cross-entropy on hard labels.
  - Vision feature MSE between projector outputs and teacher’s vision features.
  - Vision-text alignment loss (cosine or InfoNCE).
- **Training**:
  - Mix 70% multimodal, 30% text-only batches.
  - Warmup: 2 K steps; total: 30 K steps.

---

## Model Configurations

All hyperparameters and architecture settings reside in `configs/model_config.py`:

```python
@dataclass
class VirtueConfig:
    vocab_size: int = 256000
    hidden_size: int = 1024
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    intermediate_size: int = 2736
    max_position_embeddings: int = 32768
    use_multimodal: bool = True
    vision_hidden_size: int = 768
    vision_num_hidden_layers: int = 12
    mm_projector_dim: int = 1024
    # …
```

This configuration drives both model instantiation and training behavior, ensuring reproducible results.

---

## Conclusion

Virtue’s architecture balances **efficiency**, **long-context capability**, and **multimodal integration**. By distilling a large multimodal teacher into a compact student, Virtue achieves near-teacher performance on text and vision tasks, while being suitable for 8 GB GPU setups.
