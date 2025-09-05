# virtue/inference/multimodal_inference.py

"""
Multimodal inference helper functions.
"""

import torch
from PIL import Image
from .inference_engine import InferenceEngine

def multimodal_generate(
    engine: InferenceEngine,
    prompt: str,
    image_paths: list,
    **kwargs
) -> str:
    """
    Load images, run multimodal generation.
    """
    # Load and preprocess images
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(engine.model.vision_tower(image_processor(img).unsqueeze(0)))
    # Stack features
    images_tensor = torch.cat(images, dim=0)
    
    # Generate response
    return engine.generate(prompt, images=images_tensor, **kwargs)
