# scripts/training/resume_training.py

"""
Resume training from a checkpoint.
"""

import argparse
import torch
from scripts.training.train_virtue import main as train_main

if __name__ == "__main__":
    # Assumes train_virtue.py handles checkpoint loading if resume flag is set
    import sys
    sys.argv.append("--resume")
    train_main()
