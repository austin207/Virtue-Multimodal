# scripts/inference/batch_inference.py

"""
Batch inference script.
"""

import argparse
import torch
import pandas as pd
from virtue.inference.inference_engine import InferenceEngine
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VirtueForCausalLM(VirtueConfig()).to(device)
    model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
    engine = InferenceEngine(model, tokenizer, device=device)

    outputs = []
    for prompt in df["prompt"]:
        outputs.append(engine.generate(prompt, max_length=128))
    df["response"] = outputs
    df.to_csv(args.output_csv, index=False)
    print(f"Batch inference complete. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
