# scripts/evaluation/compare_models.py

"""
Compare multiple checkpoints.
"""

import argparse
from scripts.evaluation.evaluate_model import main as eval_main

if __name__ == "__main__":
    import sys
    # Pass multiple --checkpoint args as needed
    eval_main()
