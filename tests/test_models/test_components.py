# tests/test_models/test_components.py

import torch
import pytest
from virtue.models.components.attention import VirtueAttention
from virtue.models.components.mlp import VirtueMLP
from virtue.models.components.normalization import VirtueRMSNorm
from virtue.models.components.embeddings import VirtueEmbeddings
from virtue.models.components.transformer import VirtueDecoderLayer, VirtueTransformer
from virtue.models.virtue_model import VirtueConfig

@pytest.fixture
def config():
    return VirtueConfig()

def test_attention_forward(config):
    attn = VirtueAttention(config)
    x = torch.randn(2, 8, config.hidden_size)
    out, attw, _ = attn(x)
    assert out.shape == x.shape

def test_mlp_forward(config):
    mlp = VirtueMLP(config)
    x = torch.randn(2, 8, config.hidden_size)
    out = mlp(x)
    assert out.shape == x.shape

def test_rmsnorm_forward(config):
    norm = VirtueRMSNorm(config.hidden_size, config.rms_norm_eps)
    x = torch.randn(2, 8, config.hidden_size)
    out = norm(x)
    assert out.shape == x.shape

def test_embeddings_forward(config):
    emb = VirtueEmbeddings(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    out = emb(x)
    assert out.shape == (2, 8, config.hidden_size)

def test_transformer_forward(config):
    transformer = VirtueTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    out = transformer(input_ids=input_ids)
    assert out[0].shape == (2, 8, config.hidden_size)
