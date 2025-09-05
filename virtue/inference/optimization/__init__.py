# virtue/inference/optimization/__init__.py

"""
Model optimization utilities for inference.
"""

from .quantization import quantize_model
from .onnx_export import export_to_onnx

__all__ = [
    "quantize_model",
    "export_to_onnx",
]
