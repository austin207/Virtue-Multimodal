# virtue/evaluation/__init__.py

"""
Evaluation module for Virtue multimodal model.
"""

from .evaluator import Evaluator
from .metrics.text_metrics import compute_text_metrics
from .metrics.vision_metrics import compute_vision_metrics
from .metrics.multimodal_metrics import compute_multimodal_metrics
from .benchmarks.mmlu import run_mmlu_benchmark
from .benchmarks.vqa_benchmark import run_vqa_benchmark
from .benchmarks.custom_benchmarks import run_custom_benchmarks

__all__ = [
    "Evaluator",
    "compute_text_metrics",
    "compute_vision_metrics",
    "compute_multimodal_metrics",
    "run_mmlu_benchmark",
    "run_vqa_benchmark",
    "run_custom_benchmarks",
]
