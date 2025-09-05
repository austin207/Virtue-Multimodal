# virtue/data/processors/image_processor.py

"""
Image preprocessing for multimodal dataset.
"""

from torchvision import transforms
from typing import Any

class ImageProcessor:
    """
    Standard image transformations.
    """
    def __init__(
        self,
        image_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, image: Any):
        return self.transform(image)
