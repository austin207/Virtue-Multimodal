# tests/test_models/test_virtue_model.py

import torch
import pytest
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM

@pytest.fixture
def config():
    return VirtueConfig()

@pytest.fixture
def model(config):
    return VirtueForCausalLM(config)

def test_forward_shape(model):
    # batch_size=2, seq_len=8
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    assert logits.shape == (2, 8, model.config.vocab_size)
