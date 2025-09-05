# virtue/utils/model_utils.py

"""
Model configuration and loading utilities.
"""

import json
import torch

def load_config(path: str) -> dict:
    """
    Load JSON config file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_config(config: dict, path: str):
    """
    Save config dict to JSON.
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
