# scripts/setup/prepare_datasets.py

"""
Prepare and preprocess datasets for training.
"""

import argparse
import os
from virtue.data.datasets.dataset_registry import get_dataset
from virtue.data.processors import TextProcessor, ImageProcessor, MultimodalProcessor
from virtue.data.loaders.collator import DataCollator
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--raw_dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Example: convert raw JSONL to tokenized torch files
    text_proc = TextProcessor()
    for split in ["train", "val"]:
        raw_file = os.path.join(args.raw_dir, f"{split}.jsonl")
        proc_file = os.path.join(args.output_dir, f"{split}.pt")
        print(f"Processing {raw_file} -> {proc_file}")
        ds = get_dataset("text", data_file=raw_file, text_processor=text_proc, max_length=1024)
        torch.save(ds, proc_file)
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
