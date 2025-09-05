# Deployment Guide for Virtue Multimodal Model

## Aim
Provide a comprehensive deployment strategy for Virtue, including local Docker, cloud, model export, and optimizations. This guide is simplified for easy reading and execution.

---

## Table of Contents
1. [Docker Deployment](#docker-deployment)
2. [Docker Compose Setup](#docker-compose-setup)
3. [ONNX Export & Optimization](#onnx-export--optimization)
4. [Quantization for Inference](#quantization-for-inference)
5. [Cloud Deployment Options](#cloud-deployment-options)
6. [API Containerization](#api-containerization)
7. [Monitoring & Scaling](#monitoring--scaling)
8. [Security & Best Practices](#security--best-practices)

---

## 1. Docker Deployment

### 1.1 Training Container
- **Dockerfile**: `docker/Dockerfile.training`
- **Base Image**: `nvidia/cuda:12.8.1-cudnn8-devel-ubuntu22.04`
- **Features**:
  - Python 3.10 venv
  - PyTorch with CUDA 12.8
  - All training dependencies in `requirements-docker.txt`
  - Exposes ports `6006` (TensorBoard) and `8888` (Jupyter)

**Build & Run**:
```bash
cd docker
docker build -f Dockerfile.training -t virtue-training .

docker run --gpus all -v $PWD/..:/workspace/virtue-multimodal -p 6006:6006 -p 8888:8888 virtue-training
```

### 1.2 Inference Container
- **Dockerfile**: `docker/Dockerfile.inference`
- **Base Image**: `nvidia/cuda:12.8.1-runtime-ubuntu22.04`
- **Features**:
  - FastAPI server (`api_server.py`)
  - Exposes port `8000`

**Build & Run**:
```bash
docker build -f Dockerfile.inference -t virtue-inference .

docker run --gpus all -v $PWD/..:/workspace/virtue-multimodal -p 8000:8000 virtue-inference
```

---

## 2. Docker Compose Setup

**File**: `docker/docker-compose.yml`

```yaml
version: "3.8"
services:
  training:
    build:
      context: ../
      dockerfile: docker/Dockerfile.training
    runtime: nvidia
    volumes:
      - ../:/workspace/virtue-multimodal
    ports:
      - "6006:6006"
      - "8888:8888"

  inference:
    build:
      context: ../
      dockerfile: docker/Dockerfile.inference
    runtime: nvidia
    volumes:
      - ../:/workspace/virtue-multimodal
    ports:
      - "8000:8000"
```

**Usage**:
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 3. ONNX Export & Optimization

### 3.1 Export to ONNX
```bash
python scripts/deployment/create_onnx.py \
  --checkpoint checkpoints/virtue_best.pt \
  --output outputs/virtue.onnx
```
- Exports model with dynamic axes for batch and sequence.

### 3.2 Run ONNX Runtime
```bash
pip install onnxruntime
python - <<EOF
import onnxruntime as ort
sess = ort.InferenceSession('outputs/virtue.onnx')
# prepare input_ids... sess.run(['logits'], {'input_ids': arr})
EOF
```

---

## 4. Quantization for Inference

### 4.1 Dynamic Quantization
```bash
python scripts/deployment/quantize_model.py \
  --checkpoint checkpoints/virtue_best.pt \
  --output outputs/virtue_quantized.pt
```
- Reduces model size and CPU inference latency.

### 4.2 Evaluate Quantized Model
```python
from torch import load
state = load('outputs/virtue_quantized.pt')
# load into model and run inference
```

---

## 5. Cloud Deployment Options

- **AWS ECS/EKS**: Container orchestration with GPU instances.
- **GCP AI Platform**: Deploy ONNX or TorchServe model.
- **Azure ML**: Use ONNX for scalable inference endpoints.

### 5.1 Example: AWS SageMaker
1. Push ONNX to S3 bucket.
2. Create SageMaker endpoint with ONNX model.
3. Use Boto3 to invoke endpoint.

---

## 6. API Containerization

Ensure `docker build` includes:
- `scripts/inference/api_server.py`
- `requirements-docker.txt`
- Model checkpoint.

**CI/CD Tips**:
- Automate Docker build on push.
- Use GitHub Actions or GitLab CI.

---

## 7. Monitoring & Scaling

- **Prometheus & Grafana**: Track GPU/CPU, latency metrics.
- **Autoscaling**: Horizontal pod autoscaler for Kubernetes.
- **Logs**: Centralized with ELK stack.

---

## 8. Security & Best Practices

- **Environment Variables**: Store secrets in `.env` and Kubernetes secrets.
- **HTTPS**: Terminate SSL at load balancer or proxy.
- **Rate Limiting**: Protect `/generate` endpoint.
- **Authentication**: API keys or OAuth2 for production.

---

By following this deployment guide, you can run Virtue in containers, on cloud platforms, and optimize for performance and scalability.