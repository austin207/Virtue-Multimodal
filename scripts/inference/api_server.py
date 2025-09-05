# scripts/inference/api_server.py

"""
REST API server for Virtue inference using FastAPI.
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File
import torch
from virtue.inference.inference_engine import InferenceEngine
from virtue.models.multimodal.virtue_mm import VirtueMultimodalForCausalLM
from transformers import AutoTokenizer
from PIL import Image
import io
from virtue.models import VirtueConfig

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
config = VirtueConfig()
model = VirtueMultimodalForCausalLM(config).to(device)
# load checkpoint...
tokenizer = AutoTokenizer.from_pretrained("gpt2")
engine = InferenceEngine(model, tokenizer, device=device)

@app.post("/generate")
async def generate(prompt: str, file: UploadFile = File(None)):
    image = None
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    output = engine.generate(prompt, images=image, max_length=256)
    return {"response": output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
