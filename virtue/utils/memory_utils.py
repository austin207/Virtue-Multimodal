# virtue/utils/memory_utils.py

"""
GPU/CPU memory utility functions.
"""

import torch
import psutil

def clear_gpu_cache():
    """
    Clear PyTorch CUDA cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_summary() -> dict:
    """
    Return dict with current GPU and CPU memory stats.
    """
    summary = {}
    if torch.cuda.is_available():
        summary["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        summary["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
    
    vm = psutil.virtual_memory()
    summary["cpu_used_gb"] = vm.used / 1024**3
    summary["cpu_total_gb"] = vm.total / 1024**3
    summary["cpu_percent"] = vm.percent
    
    return summary
