# scripts/evaluation/benchmark_performance.py

"""
Benchmark performance (throughput, latency).
"""

import time
import argparse
import torch
from virtue.inference.inference_engine import InferenceEngine
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VirtueForCausalLM(VirtueConfig()).to(device)
    model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
    engine = InferenceEngine(model, tokenizer, device=device)

    prompt = "Hello"
    start = time.time()
    engine.generate(prompt, max_length=args.tokens)
    end = time.time()
    latency = end-start
    throughput = args.tokens / latency
    print(f"Latency: {latency:.2f}s, Throughput: {throughput:.2f} tokens/s")

if __name__ == "__main__":
    main()
