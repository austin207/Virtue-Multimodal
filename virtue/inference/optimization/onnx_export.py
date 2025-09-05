# virtue/inference/optimization/onnx_export.py

"""
ONNX export for inference acceleration.
"""

import torch
from pathlib import Path

def export_to_onnx(
    model,
    output_path: str,
    input_shape: tuple = (1, 16),
    opset_version: int = 17,
    dynamic_axes: dict = None
):
    """
    Export model to ONNX format.
    """
    model.eval()
    inputs = torch.randint(0, model.config.vocab_size, input_shape, dtype=torch.long)
    inputs = inputs.to(next(model.parameters()).device)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (inputs,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes or {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        }
    )
    print(f"Model exported to ONNX: {output_path}")
