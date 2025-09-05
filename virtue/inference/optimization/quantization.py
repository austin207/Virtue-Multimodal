# virtue/inference/optimization/quantization.py

"""
Post-training quantization utilities.
"""

import torch
from torch.quantization import quantize_dynamic
from transformers import PreTrainedModel

def quantize_model(
    model: PreTrainedModel,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False
) -> PreTrainedModel:
    """
    Apply dynamic quantization to reduce model size.
    """
    backend = "qnnpack" if torch.backends.quantized.supported_engines[0] == "qnnpack" else None
    quantized_model = quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=dtype,
        inplace=inplace
    )
    return quantized_model
