# tests/test_inference/test_multimodal_inference.py

import torch
from PIL import Image
from virtue.inference.multimodal_inference import multimodal_generate
from virtue.inference.inference_engine import InferenceEngine
from virtue.models.virtue_model import VirtueConfig, VirtueMultimodalForCausalLM
from transformers import AutoTokenizer

def test_multimodal_generate(tmp_path):
    img = tmp_path / "img.jpg"
    Image.new("RGB",(16,16)).save(img)
    config = VirtueConfig()
    model = VirtueMultimodalForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    engine = InferenceEngine(model, tokenizer, device="cpu")
    out = multimodal_generate(engine, "Hi", [str(img)], max_length=5)
    assert isinstance(out, str)
