# scripts/evaluation/evaluate_model.py

"""
Comprehensive evaluation script.
"""

import argparse
import torch
from virtue.evaluation.evaluator import Evaluator
from virtue.data.datasets.dataset_registry import get_dataset
from virtue.data.processors import TextProcessor, ImageProcessor
from virtue.data.loaders import create_dataloader
from transformers import AutoTokenizer
from virtue.models.multimodal.virtue_mm import VirtueMultimodalForCausalLM
from virtue.models.virtue_model import VirtueForCausalLM, VirtueConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["text","vision","multimodal"], required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if args.task=="multimodal":
        model = VirtueMultimodalForCausalLM(VirtueConfig())
    else:
        model = VirtueForCausalLM(VirtueConfig())
    model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
    evaluator = Evaluator(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")

    if args.task=="text":
        ds = get_dataset("text", data_file="data/processed/val.pt", text_processor=TextProcessor("gpt2"), max_length=512)
        loader = create_dataloader(ds, batch_size=8)
        results = evaluator.evaluate_text(loader)
    elif args.task=="vision":
        ds = get_dataset("multimodal", data_file="data/processed/val.pt",
                         image_root="data/raw/images", text_processor=TextProcessor("gpt2"),
                         image_processor=ImageProcessor(), max_text_length=512)
        loader = create_dataloader(ds, batch_size=8)
        results = evaluator.evaluate_vision(loader)
    else:
        ds = get_dataset("multimodal", data_file="data/processed/val.pt",
                         image_root="data/raw/images", text_processor=TextProcessor("gpt2"),
                         image_processor=ImageProcessor(), max_text_length=512)
        loader = create_dataloader(ds, batch_size=8)
        results = evaluator.evaluate_multimodal(loader)

    print("Evaluation results:", results)

if __name__ == "__main__":
    main()
