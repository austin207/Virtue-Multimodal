# Training Guide for Virtue Multimodal Model

## Aim
Guide you through training **Virtue**, a 270M-parameter multimodal language model distilled from Gemma 3 4B-IT. This guide uses a modular, step-by-step approach suitable for an 8GB GPU.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Configuration Overview](#configuration-overview)
4. [Trainer Logic](#trainer-logic)
5. [Running Training](#running-training)
6. [Monitoring & Logging](#monitoring--logging)
7. [Checkpointing & Resume](#checkpointing--resume)
8. [Advanced Tips](#advanced-tips)

---

## 1. Environment Setup

### 1.1 Create & Activate Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### 1.2 Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 1.3 Download Teacher Model
```bash
python scripts/setup/download_teacher_model.py
```

---

## 2. Data Preparation

### 2.1 Organize Raw Data
- Place images under `data/raw/images/`.
- Place JSONL files:
  - `data/raw/train.jsonl`
  - `data/raw/val.jsonl`
  - `data/raw/test.jsonl` (optional)

### 2.2 Preprocess & Tokenize
```bash
python scripts/setup/prepare_datasets.py \
  --raw_dir data/raw \
  --output_dir data/processed
```
This generates `train.pt` and `val.pt` with tokenized text and image tensors.

---

## 3. Configuration Overview

### 3.1 Training Hyperparameters (`configs/training_config.py`)
```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 32
learning_rate = 2e-5
weight_decay = 0.01
num_warmup_steps = 2000
num_train_steps = 30000
bf16 = True  # Preferred on 8GB GPU
save_steps = 5000
logging_steps = 100
output_dir = 'checkpoints/'
```

### 3.2 Distillation Settings (`configs/distillation_config.py`)
```python
teacher_model_name = 'google/gemma-3-4b-it'
teacher_quantization = '4bit'
temperature = 4.0
alpha_kd = 0.8
alpha_vision = 0.2
text_to_vision_ratio = 0.3
multimodal_ratio = 0.7
```

---

## 4. Trainer Logic

### 4.1 Core Components
- **Student**: `VirtueForCausalLM`
- **Teacher**: `GemmaTeacher` (frozen, quantized)
- **Data Loader**: `create_dataloader` with `DataCollator`
- **Optimizer**: AdamW via `create_optimizer`
- **Scheduler**: Cosine schedule with warmup
- **Losses**: KL + CE + vision feature MSE + alignment

### 4.2 Training Loop (Distillation)
1. Zero gradients
2. Teacher forward (no grads)
3. Student forward
4. Compute KD loss and multimodal loss
5. Backpropagate and optimizer step
6. Scheduler step
7. Log metrics and memory
8. Save checkpoints

---

## 5. Running Training

```bash
python scripts/training/train_virtue.py \
  --config configs/training_config.py
```
- Use `--resume` flag to continue from last checkpoint.
- Modify batch size or learning rate in `configs/` to tune.

---

## 6. Monitoring & Logging

### 6.1 TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```
- Loss curves, learning rate schedule.

### 6.2 Weights & Biases (W&B)
- Auto logs if `use_wandb` is True.
- Visit the provided W&B run URL.

### 6.3 Memory Monitor
- Prints GPU/CPU stats every 100 steps.
- Alerts if usage > 90%.

---

## 7. Checkpointing & Resume

- Checkpoints saved every 5,000 steps.
- Best model tracked by validation loss.
- Resume training:
  ```bash
  python scripts/training/resume_training.py
  ```

---

## 8. Advanced Tips

- **Mixed Precision**: Use `bf16=True` in config for faster training.
- **Gradient Checkpointing**: Saves memory at the cost of compute.
- **Batch Size**: Increase `gradient_accumulation_steps` to simulate larger batch.
- **Context Window**: Start with `max_sequence_length=8192`, then increase to 32K gradually.
- **Profiling**: Use `MemoryProfiler` to diagnose memory spikes.
- **Hyperparameter Search**: Leverage W&B sweeps or custom scripts.

---

By following these steps, you’ll set up and train Virtue efficiently, monitor its progress, and manage checkpoints. Happy distilling!