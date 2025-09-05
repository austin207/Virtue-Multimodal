# virtue/evaluation/metrics/vision_metrics.py

"""
Vision task metrics.
"""

from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_vision_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 for vision classification.
    """
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
