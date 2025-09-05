# tests/test_data/test_processors.py

import torch
from PIL import Image
from virtue.data.processors.text_processor import TextProcessor
from virtue.data.processors.image_processor import ImageProcessor

def test_text_processor(tmp_path):
    proc = TextProcessor(tokenizer_name="bert-base-uncased")
    out = proc("hello world", max_length=5)
    assert "input_ids" in out

def test_image_processor():
    proc = ImageProcessor(image_size=16)
    img = Image.new("RGB",(32,32))
    tensor = proc(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1:] == (16,16)
