# virtue/utils/checkpoint_utils.py

"""
Save and load model checkpoints.
"""

import torch
from pathlib import Path

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    step: int,
    best: bool = False
):
    """
    Save model, optimizer, scheduler state to path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        data["scheduler_state"] = scheduler.state_dict()
    if best:
        path = path.with_name("best_" + path.name)
    torch.save(data, path)

def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load checkpoint into model, optimizer, scheduler.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint.get("step", 0)
