# scripts/training/train_virtue.py

"""
Main training script for Virtue distillation.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from configs.training_config import TrainingConfig
from configs.distillation_config import DistillationConfig
from configs.data_config import DataConfig
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM
from virtue.models.teacher.gemma_teacher import GemmaTeacher
from virtue.training.distillation_trainer import VirtueDistillationTrainer
from virtue.data.datasets.dataset_registry import get_dataset
from virtue.data.processors import TextProcessor, ImageProcessor
from virtue.data.loaders import create_dataloader, DataCollator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.py")
    args = parser.parse_args()

    train_cfg = TrainingConfig()
    distill_cfg = DistillationConfig()
    data_cfg = DataConfig()

    text_proc = TextProcessor(data_cfg.tokenizer_name)
    image_proc = ImageProcessor(data_cfg.image_size, tuple(data_cfg.image_mean), tuple(data_cfg.image_std))
    collator = DataCollator(pad_token_id=0)

    train_ds = get_dataset("multimodal",
        data_file="data/processed/train.pt",
        image_root="data/raw/images",
        text_processor=text_proc,
        image_processor=image_proc,
        max_text_length=data_cfg.max_token_length
    )
    train_loader = create_dataloader(train_ds, batch_size=train_cfg.per_device_train_batch_size,
                                     num_workers=data_cfg.num_workers, collate_fn=collator)

    student = VirtueForCausalLM(VirtueConfig())
    teacher = GemmaTeacher(model_name=distill_cfg.teacher_model_name,
                           quantization=distill_cfg.teacher_quantization,
                           freeze=distill_cfg.freeze_teacher)

    trainer = VirtueDistillationTrainer(
        student, teacher, train_loader, config={**vars(train_cfg), **vars(distill_cfg)}
    )
    result = trainer.train(num_epochs=1)
    print("Training complete:", result)

if __name__ == "__main__":
    main()
