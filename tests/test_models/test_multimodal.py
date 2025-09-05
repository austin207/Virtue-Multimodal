# tests/test_models/test_multimodal.py

import torch
import pytest
from virtue.models.multimodal.virtue_mm import VirtueConfig, VirtueMultimodalForCausalLM
from virtue.models.virtue_model import VirtueForCausalLM

@pytest.fixture
def config():
    return VirtueConfig()

@pytest.fixture
def model(config):
    return VirtueMultimodalForCausalLM(config)

def test_multimodal_forward_shape(model):
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    images = torch.randn(1, 3, model.config.vision_image_size, model.config.vision_image_size)
    outputs = model(input_ids=input_ids, images=images)
    logits = outputs.logits
    assert logits.ndim == 3
    assert logits.size(0) == 1
