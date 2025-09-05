# README.md for Virtue Multimodal — Updated with Repo Structure

***

# Virtue Multimodal Language Model

**270M-parameter lightweight multimodal model distilled from Google’s Gemma 3 4B-IT.**

***

## Overview

Virtue is a compact, efficient multimodal language model supporting text and image inputs. It features a 32K-token context window and integrates the SigLIP vision encoder. Designed to train and run effectively on a single 8GB GPU, it supports research and deployment in resource-constrained settings.

***

## Features

- Multimodal comprehension: text + images
- Long context with Rotary Positional Embeddings (RoPE)
- Knowledge distillation from Gemma 3 4B-IT teacher
- Modular & extensible architecture
- Pre-packaged training, evaluation, inference scripts
- Dockerized for easy setup & deployment

***

## Repository Structure

```
virtue-multimodal/
├── README.md                 # This README
├── requirements.txt          # Core dependencies
├── setup.py                 # Setup script
├── .gitignore
├── .env.example             # Env variable template
├── pyproject.toml           # Build configuration
│
├── configs/                 # All configuration files (model, training, data, deployment)
│   ├── __init__.py
│   ├── model_config.py
│   ├── training_config.py
│   ├── distillation_config.py
│   ├── data_config.py
│   └── deployment_config.py
│
├── virtue/                  # Core library code
│   ├── __init__.py
│   ├── models/              # Model architectures & components
│   ├── training/            # Training logic, losses, optimizers, callbacks
│   ├── data/                # Datasets, preprocessing, loaders
│   ├── inference/           # Inference engines, utilities, optimizations
│   ├── evaluation/          # Evaluators, metrics, benchmarks
│   └── utils/               # Logging, memory management, checkpointing, visualization
│
├── scripts/                 # Standalone scripts for setup, training, inference, evaluation, deployment
│   ├── setup/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── deployment/
│
├── tests/                   # Unit and integration tests
│
├── docker/                  # Dockerfiles, compose, dependencies for containerized runs
│
├── docs/                    # User and developer documentation files
│
├── checkpoints/             # Checkpoint storage
├── logs/                    # Log files
├── data/                    # Raw & processed datasets
└── outputs/                 # Model outputs: trained weights, visualizations, evals
```

***

## Quickstart

### 1. Set up environment

```bash
python3.10 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Download teacher model (Gemma 3 4B-IT)

```bash
python scripts/setup/download_teacher_model.py
```

### 3. Prepare data

```bash
python scripts/setup/prepare_datasets.py --raw_dir data/raw --output_dir data/processed
```

### 4. Train the model

```bash
python scripts/training/train_virtue.py --config configs/training_config.py
```

### 5. Run inference (interactive chat demo)

```bash
python scripts/inference/chat_demo.py
```

--- 

## Contribution Guidelines

- Use `black`, `isort` and `flake8` for style.
- Write tests for new features.
- Follow modular design principles.
- Use GitHub issues and pull requests for changes.
- Refer to `docs/` for detailed architecture, training, inference, API, and deployment.

***

## License

MIT License — See LICENSE file for details.

***

## References and Acknowledgments

- Google’s Gemma 3 4B-IT model.
- SigLIP Vision Encoder project.
- Hugging Face Transformers and Accelerate.
- Open-source ecosystem contributors.

***

*For comprehensive documentation, see the `docs/` folder including `ARCHITECTURE.md`, `TRAINING.md`, `INFERENCE.md`, `API.md`, and `DEPLOYMENT.md`.*

Happy exploring Virtue!
