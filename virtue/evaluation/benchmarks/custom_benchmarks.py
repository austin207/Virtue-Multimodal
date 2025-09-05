# virtue/evaluation/benchmarks/custom_benchmarks.py

"""
Custom benchmarks for domain-specific tests.
"""

from typing import Dict, List

def run_custom_benchmarks(
    model, tokenizer, test_cases: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Run custom benchmarks provided as list of dicts:
    [{"prompt": "...", "expected": "..."}, ...]
    """
    preds, refs = [], []
    for case in test_cases:
        prompt, expected = case["prompt"], case["expected"]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(expected)
    # use text metrics for evaluation
    from ..metrics.text_metrics import compute_text_metrics
    return compute_text_metrics(preds, refs)
