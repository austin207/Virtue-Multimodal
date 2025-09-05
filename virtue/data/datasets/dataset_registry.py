# virtue/data/datasets/dataset_registry.py

"""
Registry for dataset classes.
Allows dynamic instantiation by name.
"""

from typing import Type, Dict
from .multimodal_dataset import MultimodalDataset
from .text_dataset import TextDataset

DATASET_REGISTRY: Dict[str, Type] = {
    "multimodal": MultimodalDataset,
    "text": TextDataset,
}

def get_dataset(
    name: str,
    **kwargs
):
    """
    Instantiate a dataset by name.
    name: "multimodal" or "text"
    kwargs: constructor arguments
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](**kwargs)
