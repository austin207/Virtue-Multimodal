"""
Training module for Virtue multimodal model distillation
"""

from .distillation_trainer import VirtueDistillationTrainer
from .losses.distillation_loss import DistillationLoss, KnowledgeDistillationLoss
from .losses.multimodal_loss import MultimodalDistillationLoss
from .optimizers.scheduler import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from .optimizers.optimizer_utils import create_optimizer
from .callbacks.memory_monitor import MemoryMonitor
from .callbacks.model_checkpoint import ModelCheckpoint
from .callbacks.wandb_logger import WandBLogger

__all__ = [
    "VirtueDistillationTrainer",
    "DistillationLoss",
    "KnowledgeDistillationLoss", 
    "MultimodalDistillationLoss",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "create_optimizer",
    "MemoryMonitor",
    "ModelCheckpoint",
    "WandBLogger",
]
