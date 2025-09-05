# configs/training_config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """
    General training hyperparameters.
    """
    # Batch & optimization
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 2000
    num_train_steps: int = 30000

    # Precision & checkpointing
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    save_steps: int = 5000
    logging_steps: int = 100
    output_dir: str = "checkpoints/"

    # Context
    max_sequence_length: int = 8192       # initial context, expandable to 32K
    pad_to_multiple_of: Optional[int] = None