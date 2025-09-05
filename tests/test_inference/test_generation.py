# tests/test_inference/test_generation.py

import torch
from virtue.inference.inference_engine import InferenceEngine
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM
from transformers import AutoTokenizer

def test_text_generation():
    config = VirtueConfig()
    model = VirtueForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    engine = InferenceEngine(model, tokenizer, device="cpu")
    out = engine.generate("Hello", max_length=5)
    assert isinstance(out, str)
