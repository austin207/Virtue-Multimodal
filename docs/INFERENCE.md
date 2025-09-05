# Inference Guide for Virtue Multimodal Model

## Aim
Provide a comprehensive, step-by-step guide to run inference with **Virtue**, covering text-only, vision-only, and multimodal scenarios. This simplified manual ensures you can deploy and test Virtue easily.

---

## Table of Contents
1. [Setup & Requirements](#setup--requirements)
2. [Loading the Model](#loading-the-model)
3. [Text-Only Generation](#text-only-generation)
4. [Vision-Only Inference](#vision-only-inference)
5. [Multimodal Generation](#multimodal-generation)
6. [Batch & Streaming Modes](#batch--streaming-modes)
7. [API & Server Deployment](#api--server-deployment)
8. [Optimizations & Tips](#optimizations--tips)

---

## 1. Setup & Requirements

1. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
2. **Install Inference Dependencies** (if not already):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. **Prepare Model Checkpoint**
   - Place your trained student checkpoint in `checkpoints/` (e.g., `virtue_best.pt`).

---

## 2. Loading the Model

```python
from transformers import AutoTokenizer
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM
from virtue.inference.inference_engine import InferenceEngine

# Initialize config, model, tokenizer
config = VirtueConfig()
model = VirtueForCausalLM(config)
model.load_state_dict(torch.load("checkpoints/virtue_best.pt")['model_state_dict'])

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or Gemma tokenizer

# Create inference engine
engine = InferenceEngine(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
```

---

## 3. Text-Only Generation

Generate continuations given a prompt:

```python
prompt = "Once upon a time"
output = engine.generate(
    prompt,
    max_length=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
)
print(output)
```

**Parameters Explanation**:
- `max_length`: Total tokens including prompt.
- `temperature`: Sampling randomness.
- `top_k`, `top_p`: Sampling strategies.

---

## 4. Vision-Only Inference

For image classification or feature extraction:

```python
from PIL import Image
from virtue.models.multimodal.vision_encoder import SigLIPVisionEncoder

# Instantiate vision encoder separately
vision_encoder = SigLIPVisionEncoder(config)
vision_encoder.eval()

# Load and preprocess image
ing = Image.open("data/raw/images/example.jpg").convert("RGB")
from virtue.data.processors.image_processor import ImageProcessor
img_proc = ImageProcessor(config.vision_image_size)
tensor = img_proc(ing).unsqueeze(0)  # [1,3,H,W]

# Get features
features = vision_encoder(tensor.to(engine.device))
print("Vision features shape:", features.shape)
```

To classify via projector + simple head:

```python
proj = engine.model.mm_projector
logits = proj(features).mean(dim=1)  # Global average
pred = logits.argmax(dim=-1)
print("Predicted class index:", pred.item())
```

---

## 5. Multimodal Generation

Combine text prompt and image input:

```python
from PIL import Image
# Load image
img = Image.open("data/raw/images/example.jpg").convert("RGB")
img_tensor = img_proc(img).unsqueeze(0)

prompt = "Describe this image:"
output = engine.generate(
    prompt,
    images=img_tensor,
    max_length=100,
)
print(output)
```

**Note**: Internally, `<img>` tokens are replaced by projected vision embeddings.

---

## 6. Batch & Streaming Modes

### Batch Inference
```python
prompts = ["Hello","What is AI?"]
results = [engine.generate(p, max_length=20) for p in prompts]
print(results)
```

### Streaming Generation
```python
for token in engine.stream_generate("The meaning of life is", max_length=20):
    print(token, end="")
```
Enables real-time output in CLI or dashboards.

---

## 7. API & Server Deployment

### FastAPI Example
```bash
python scripts/inference/api_server.py
```
- Serves `/generate` endpoint.
- Accepts JSON prompt and optional image file.

### Gradio Demo
```bash
python scripts/inference/chat_demo.py
```
- Launches local web UI for interactive chat.

---

## 8. Optimizations & Tips

- **Quantization**: Use dynamic quantization for CPU inference.
  ```python
  from virtue.inference.optimization.quantization import quantize_model
  qmodel = quantize_model(model, dtype=torch.qint8)
  ```
- **ONNX Export**: Export to ONNX for cross-platform deployment.
  ```bash
  python scripts/deployment/create_onnx.py --checkpoint checkpoints/virtue_best.pt --output outputs/virtue.onnx
  ```
- **Batch Size**: Adjust batch size in `InferenceEngine.generate` loops.
- **Device**: Use `device_map` (e.g., `transformers` pipeline) for multi-GPU.
- **Memory**: Clear cache: `engine.model.to('cpu'); torch.cuda.empty_cache()`

---

With this guide, you can run text, vision, and multimodal inference confidently, deploy REST APIs, and optimize for various environments. Enjoy exploring Virtue!