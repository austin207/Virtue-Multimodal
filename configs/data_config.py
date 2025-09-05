# configs/data_config.py

from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    """
    Dataset and data loader configurations.
    """
    # Dataset names or paths
    multimodal_datasets: List[str] = (
        ["llava-150k", "coco-captions", "vqa-v2", "sharegpt4v"]
    )
    text_datasets: List[str] = (
        ["alpaca-gpt4", "ultrachat-200k"]
    )

    # Image preprocessing
    image_size: int = 224
    image_mean: List[float] = (0.485, 0.456, 0.406)
    image_std: List[float] = (0.229, 0.224, 0.225)

    # Tokenization
    tokenizer_name: str = "google/gemma-3-4b-it"
    max_token_length: int = 2048

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False