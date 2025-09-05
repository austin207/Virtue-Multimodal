"""
Loss functions for knowledge distillation
"""

from .distillation_loss import DistillationLoss, KnowledgeDistillationLoss
from .multimodal_loss import MultimodalDistillationLoss, VisionTextAlignmentLoss
from .regularization import ActivationRegularization, AttentionRegularization

__all__ = [
    "DistillationLoss",
    "KnowledgeDistillationLoss",
    "MultimodalDistillationLoss", 
    "VisionTextAlignmentLoss",
    "ActivationRegularization",
    "AttentionRegularization",
]
