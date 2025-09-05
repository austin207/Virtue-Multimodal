# virtue/evaluation/benchmarks/mmlu.py

"""
MMLU text benchmarks.
"""

from datasets import load_dataset
from ..metrics.text_metrics import compute_text_metrics
from typing import Dict

def run_mmlu_benchmark(
    model, tokenizer, batch_size: int = 8, max_samples: int = 100
) -> Dict[str, float]:
    """
    Run MMLU (Massive Multitask Language Understanding) benchmark.
    """
    dataset = load_dataset("mmlu", "common_topics", split="validation[:{}]".format(max_samples))
    questions = dataset["question"]
    answers = dataset["answer"]
    
    preds, refs = [], []
    for q, a in zip(questions, answers):
        input_text = f"Question: {q}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(**inputs, max_length=tokenizer.model_max_length)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        preds.append(pred)
        refs.append(a)
    return compute_text_metrics(preds, refs)
