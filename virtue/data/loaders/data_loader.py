# virtue/data/loaders/data_loader.py

"""
DataLoader creation utility.
"""

from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

def create_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn = None,
    drop_last: bool = False
) -> DataLoader:
    """
    Create PyTorch DataLoader with standard settings.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=True
    )
