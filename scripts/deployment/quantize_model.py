# scripts/deployment/quantize_model.py

"""
Quantize a saved checkpoint for efficient inference.
"""

import argparse
import torch
from virtue.inference.optimization.quantization import quantize_model
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
    qmodel = quantize_model(model, dtype=torch.qint8)
    torch.save(qmodel.state_dict(), args.output)
    print(f"Quantized model saved to {args.output}")

if __name__ == "__main__":
    main()
