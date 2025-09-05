# scripts/deployment/create_onnx.py

"""
Export model checkpoint to ONNX.
"""

import argparse
import torch
from virtue.inference.optimization.onnx_export import export_to_onnx
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = VirtueConfig()
    model = VirtueForCausalLM(cfg)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])
    export_to_onnx(model, args.output)
    
if __name__ == "__main__":
    main()
