# tests/test_training/test_distillation.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from virtue.training.distillation_trainer import VirtueDistillationTrainer
from virtue.models.virtue_model import VirtueConfig, VirtueForCausalLM
from virtue.models.teacher.gemma_teacher import GemmaTeacher

def test_train_step_smoke():
    # dummy data
    input_ids = torch.randint(0, 1000, (2, 4))
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    loader = DataLoader(dataset, batch_size=2)
    config = {"learning_rate":1e-5, "max_steps":1, "temperature":1.0, 
              "alpha_kd":0.8, "alpha_vision":0.0}
    student = VirtueForCausalLM(VirtueConfig())
    teacher = GemmaTeacher(freeze=True)
    trainer = VirtueDistillationTrainer(student, teacher, loader, config=config)
    batch = {"input_ids":input_ids, "labels":labels}
    loss = trainer.train_step(batch)
    assert "total_loss" in loss
