# virtue/utils/visualization.py

"""
Visualization utilities for training and evaluation metrics.
"""

import matplotlib.pyplot as plt
import os
from typing import Dict, List

def plot_training_metrics(
    history: List[Dict[str, float]],
    metrics: List[str],
    save_path: str = None
):
    """
    Plot training history for given metrics.
    history: List of dicts with keys metrics.
    """
    epochs = list(range(1, len(history) + 1))
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        values = [h.get(metric, None) for h in history]
        plt.plot(epochs, values, label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
