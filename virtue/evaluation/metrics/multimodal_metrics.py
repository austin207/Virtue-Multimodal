# virtue/evaluation/metrics/multimodal_metrics.py

"""
Multimodal task metrics (e.g., VQA).
"""

from typing import List, Dict

def compute_multimodal_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute exact-match and token-level accuracy for VQA-like tasks.
    """
    total = len(predictions)
    exact_match = sum([pred.strip().lower() == ref.strip().lower()
                       for pred, ref in zip(predictions, references)]) / total
    
    # Token-level accuracy
    token_accuracies = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        matches = sum([1 for p, r in zip(pred_tokens, ref_tokens) if p == r])
        token_accuracies.append(matches / max(len(ref_tokens), 1))
    
    token_acc = sum(token_accuracies) / total
    
    return {
        "exact_match": exact_match,
        "token_accuracy": token_acc,
    }
