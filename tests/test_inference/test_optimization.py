# tests/test_inference/test_optimization.py

import torch
from virtue.inference.optimization.quantization import quantize_model
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM

def test_quantize_model():
    config = VirtueConfig()
    model = VirtueForCausalLM(config)
    qmodel = quantize_model(model, dtype=torch.qint8, inplace=False)
    assert qmodel is not None
