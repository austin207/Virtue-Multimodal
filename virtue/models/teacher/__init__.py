"""
Teacher model interface for knowledge distillation
"""

from .gemma_teacher import GemmaTeacher
from .teacher_utils import load_teacher_model, prepare_teacher_outputs

__all__ = [
    "GemmaTeacher",
    "load_teacher_model", 
    "prepare_teacher_outputs",
]
