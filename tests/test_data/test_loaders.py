# tests/test_data/test_loaders.py

import torch
from virtue.data.loaders.data_loader import create_dataloader
from virtue.data.loaders.collator import DataCollator
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __len__(self): return 2
    def __getitem__(self, idx):
        return {"input_ids":torch.tensor([1,2]), "attention_mask":torch.tensor([1,1]),
                "images":torch.zeros(3,8,8), "labels":torch.tensor([1,2])}

def test_dataloader_and_collator():
    ds = DummyDataset()
    collator = DataCollator(pad_token_id=0)
    loader = create_dataloader(ds, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0]==2
