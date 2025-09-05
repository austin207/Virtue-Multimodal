# scripts/training/distributed_train.py

"""
Launch distributed training using Accelerate.
"""

from accelerate import Accelerator
import torch
from scripts.training.train_virtue import main as train_main

def main():
    accelerator = Accelerator()
    train_main()

if __name__ == "__main__":
    main()
