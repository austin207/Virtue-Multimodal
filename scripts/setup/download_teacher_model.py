# scripts/setup/download_teacher_model.py

"""
Download the Gemma 3 4B-IT teacher model to local cache.
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="google/gemma-3-4b-it", help="Teacher model name"
    )
    args = parser.parse_args()

    print(f"Downloading teacher model and tokenizer: {args.model_name}")
    AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto"
    )
    print("Download complete.")

if __name__ == "__main__":
    main()
