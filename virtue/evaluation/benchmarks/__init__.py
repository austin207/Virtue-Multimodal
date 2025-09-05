# virtue/evaluation/benchmarks/__init__.py

"""
Benchmark scripts for Virtue evaluation.
"""

from .mmlu import run_mmlu_benchmark
from .vqa_benchmark import run_vqa_benchmark
from .custom_benchmarks import run_custom_benchmarks

__all__ = [
    "run_mmlu_benchmark",
    "run_vqa_benchmark",
    "run_custom_benchmarks",
]
