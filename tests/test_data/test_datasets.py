# tests/test_data/test_datasets.py

from virtue.data.datasets.dataset_registry import get_dataset
import pytest

def test_registry_text(tmp_path):
    # create dummy file
    data_file = tmp_path / "text.jsonl"
    data_file.write_text('{"text":"hello"}\n')
    ds = get_dataset("text", data_file=str(data_file), text_processor=lambda x,**k: {"input_ids":[0],"attention_mask":[1]}, max_length=10)
    assert len(ds) == 1

def test_registry_multimodal(tmp_path):
    jsonl = tmp_path / "multi.jsonl"
    jsonl.write_text('{"image_path":"img.jpg","text":"hi"}\n')
    # dummy image
    img = tmp_path / "img.jpg"
    from PIL import Image
    Image.new("RGB",(10,10)).save(img)
    ds = get_dataset("multimodal", data_file=str(jsonl), image_root=str(tmp_path),
                     text_processor=lambda x,**k: {"input_ids":[0],"attention_mask":[1]}, image_processor=lambda x: 0, max_text_length=10)
    assert len(ds) == 1
