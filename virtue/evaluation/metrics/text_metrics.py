# virtue/evaluation/metrics/text_metrics.py

"""
Text generation metrics.
"""

from typing import List, Dict
from datasets import load_metric

_bleu = load_metric("bleu")
_rouge = load_metric("rouge")

def compute_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE metrics for text generation.
    """
    # Prepare for BLEU
    refs_tokenized = [[ref.split()] for ref in references]
    preds_tokenized = [pred.split() for pred in predictions]
    
    bleu_res = _bleu.compute(predictions=preds_tokenized, references=refs_tokenized)
    rouge_res = _rouge.compute(predictions=predictions, references=references)
    
    return {
        "bleu": bleu_res["bleu"],
        "rouge1": rouge_res["rouge1"].mid.fmeasure,
        "rouge2": rouge_res["rouge2"].mid.fmeasure,
        "rougeL": rouge_res["rougeL"].mid.fmeasure,
    }
