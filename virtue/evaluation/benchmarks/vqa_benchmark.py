# virtue/evaluation/benchmarks/vqa_benchmark.py

"""
Visual Question Answering (VQA) benchmarks.
"""

from datasets import load_dataset
from ..metrics.multimodal_metrics import compute_multimodal_metrics
from typing import Dict
import torch

def run_vqa_benchmark(
    model, tokenizer, image_processor, max_samples: int = 100
) -> Dict[str, float]:
    """
    Run VQA v2 benchmark.
    """
    dataset = load_dataset("vqa_v2", split="validation[:{}]".format(max_samples))
    preds, refs = [], []
    for example in dataset:
        image = image_processor(example["image"])
        question = example["question"]
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            images=image.unsqueeze(0).to(next(model.parameters()).device)
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(example["answers"]["multiple_choice_answer"])
    return compute_multimodal_metrics(preds, refs)
