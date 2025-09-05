# tests/test_training/test_optimizers.py

import torch
from torch import nn
from virtue.training.optimizers.optimizer_utils import create_optimizer

def test_optimizer_creation():
    model = nn.Linear(10,10)
    optimizer = create_optimizer(model, optimizer_type="adamw", learning_rate=1e-3)
    assert isinstance(optimizer, torch.optim.Optimizer)
