# Virtue Multimodal

## AIM
 
“Build **Virtue**, a 270 M-parameter multimodal language model distilled from Google’s Gemma 3 4B-IT. It supports a 32 K token context window and leverages the pre-trained SigLIP vision encoder. Virtue will run and train on a single 8 GB GPU, combining text and image understanding via a modular, reusable codebase.”


```
virtue-multimodal/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── pyproject.toml
│
├── configs/
│   ├── __init__.py
│   ├── model_config.py          # Virtue architecture configs
│   ├── training_config.py       # Training hyperparameters
│   ├── distillation_config.py   # Distillation settings
│   ├── data_config.py           # Dataset configurations
│   └── deployment_config.py     # Inference/deployment settings
│
├── virtue/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── virtue_model.py      # Main Virtue 270M model
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py   # Transformer layers
│   │   │   ├── attention.py     # Multi-head attention
│   │   │   ├── embeddings.py    # Token/position embeddings
│   │   │   ├── mlp.py          # Feed-forward networks
│   │   │   └── normalization.py # RMSNorm implementation
│   │   ├── multimodal/
│   │   │   ├── __init__.py
│   │   │   ├── virtue_mm.py     # Multimodal wrapper
│   │   │   ├── vision_encoder.py # Vision processing
│   │   │   └── mm_projector.py  # Vision-text projection
│   │   └── teacher/
│   │       ├── __init__.py
│   │       ├── gemma_teacher.py # Gemma 3 4B teacher interface
│   │       └── teacher_utils.py # Teacher model utilities
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── distillation_trainer.py  # Main training logic
│   │   ├── losses/
│   │   │   ├── __init__.py
│   │   │   ├── distillation_loss.py # KD loss functions
│   │   │   ├── multimodal_loss.py   # Vision-text losses
│   │   │   └── regularization.py    # Additional losses
│   │   ├── optimizers/
│   │   │   ├── __init__.py
│   │   │   ├── scheduler.py     # Learning rate scheduling
│   │   │   └── optimizer_utils.py # Optimizer configurations
│   │   └── callbacks/
│   │       ├── __init__.py
│   │       ├── memory_monitor.py # VRAM monitoring
│   │       ├── model_checkpoint.py # Checkpointing
│   │       └── wandb_logger.py  # Experiment tracking
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── multimodal_dataset.py # Vision-text datasets
│   │   │   ├── text_dataset.py      # Text-only datasets
│   │   │   └── dataset_registry.py   # Dataset factory
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── text_processor.py    # Text tokenization
│   │   │   ├── image_processor.py   # Image preprocessing
│   │   │   └── multimodal_processor.py # Combined processing
│   │   └── loaders/
│   │       ├── __init__.py
│   │       ├── data_loader.py       # Efficient data loading
│   │       └── collator.py          # Batch collation
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference_engine.py      # Main inference class
│   │   ├── generation_utils.py      # Text generation utilities
│   │   ├── multimodal_inference.py  # Vision+text inference
│   │   └── optimization/
│   │       ├── __init__.py
│   │       ├── quantization.py     # Model quantization
│   │       └── onnx_export.py      # ONNX conversion
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py            # Main evaluation logic
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── text_metrics.py     # Text generation metrics
│   │   │   ├── vision_metrics.py   # Vision task metrics
│   │   │   └── multimodal_metrics.py # Combined metrics
│   │   └── benchmarks/
│   │       ├── __init__.py
│   │       ├── mmlu.py            # Text benchmarks
│   │       ├── vqa_benchmark.py   # Vision Q&A benchmarks
│   │       └── custom_benchmarks.py # Domain-specific tests
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py        # Structured logging
│       ├── memory_utils.py         # VRAM optimization
│       ├── model_utils.py          # Model utilities
│       ├── checkpoint_utils.py     # Save/load checkpoints
│       └── visualization.py        # Training visualizations
│
├── scripts/
│   ├── setup/
│   │   ├── install_dependencies.sh
│   │   ├── download_teacher_model.py
│   │   └── prepare_datasets.py
│   ├── training/
│   │   ├── train_virtue.py         # Main training script
│   │   ├── resume_training.py      # Resume from checkpoint
│   │   └── distributed_train.py    # Multi-GPU training
│   ├── evaluation/
│   │   ├── evaluate_model.py       # Comprehensive evaluation
│   │   ├── benchmark_performance.py # Performance benchmarks
│   │   └── compare_models.py       # Model comparison
│   ├── inference/
│   │   ├── chat_demo.py           # Interactive chat demo
│   │   ├── batch_inference.py     # Batch processing
│   │   └── api_server.py          # REST API server
│   └── deployment/
│       ├── quantize_model.py      # Model quantization
│       ├── create_onnx.py         # ONNX export
│       └── docker_build.py        # Docker containerization
│
├── notebooks/
│   ├── 01_architecture_design.ipynb    # Model architecture exploration
│   ├── 02_data_exploration.ipynb       # Dataset analysis
│   ├── 03_distillation_analysis.ipynb  # Training analysis
│   ├── 04_model_evaluation.ipynb       # Performance evaluation
│   └── 05_deployment_demo.ipynb        # Deployment examples
│
├── tests/
│   ├── __init__.py
│   ├── test_models/
│   │   ├── test_virtue_model.py
│   │   ├── test_multimodal.py
│   │   └── test_components.py
│   ├── test_training/
│   │   ├── test_distillation.py
│   │   ├── test_losses.py
│   │   └── test_optimizers.py
│   ├── test_data/
│   │   ├── test_datasets.py
│   │   ├── test_processors.py
│   │   └── test_loaders.py
│   └── test_inference/
│       ├── test_generation.py
│       ├── test_multimodal_inference.py
│       └── test_optimization.py
│
├── docker/
│   ├── Dockerfile.training         # Training environment
│   ├── Dockerfile.inference        # Inference environment
│   ├── docker-compose.yml          # Complete setup
│   └── requirements-docker.txt     # Docker-specific deps
│
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md            # Model architecture details
│   ├── TRAINING.md               # Training guide
│   ├── INFERENCE.md              # Inference guide
│   ├── API.md                    # API documentation
│   └── DEPLOYMENT.md             # Deployment guide
│
├── checkpoints/
│   ├── .gitkeep
│   └── README.md                 # Checkpoint organization
│
├── logs/
│   ├── .gitkeep
│   └── README.md                 # Log organization
│
├── data/
│   ├── raw/                      # Raw datasets
│   ├── processed/                # Processed datasets
│   ├── samples/                  # Sample files for testing
│   └── README.md                 # Data organization
│
└── outputs/
    ├── models/                   # Trained models
    ├── evaluations/              # Evaluation results
    ├── visualizations/           # Training plots
    └── README.md                 # Output organization

```