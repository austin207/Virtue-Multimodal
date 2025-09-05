# tests/test_training/test_losses.py

import torch
from virtue.training.losses.distillation_loss import KnowledgeDistillationLoss

def test_kd_loss():
    loss_fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.7)
    logits_s = torch.randn(2,4,10)
    logits_t = torch.randn(2,4,10)
    labels = torch.randint(0,10,(2,4))
    res = loss_fn(logits_s, logits_t, labels)
    assert "total_loss" in res

