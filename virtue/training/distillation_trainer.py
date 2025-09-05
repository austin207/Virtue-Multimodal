"""
Main distillation trainer for Virtue model
"""

import os
import time
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.virtue_model import VirtueForCausalLM
from ..models.multimodal.virtue_mm import VirtueMultimodalForCausalLM
from ..models.teacher.gemma_teacher import GemmaTeacher
from ..utils.memory_utils import clear_gpu_cache, get_gpu_memory_info
from ..utils.logging_utils import get_logger
from .losses.distillation_loss import KnowledgeDistillationLoss
from .losses.multimodal_loss import MultimodalDistillationLoss
from .optimizers.optimizer_utils import create_optimizer
from .optimizers.scheduler import get_cosine_schedule_with_warmup
from .callbacks.memory_monitor import MemoryMonitor
from .callbacks.model_checkpoint import ModelCheckpoint
from .callbacks.wandb_logger import WandBLogger

logger = get_logger(__name__)

class VirtueDistillationTrainer:
    """
    Main trainer for knowledge distillation from Gemma 3 4B-IT to Virtue 270M
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: GemmaTeacher,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.config = config or {}
        
        # Models
        self.student = student_model.to(self.device)
        self.teacher = teacher_model  # Already on device and quantized
        
        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_functions()
        self._setup_callbacks()
        
        logger.info(f"Initialized VirtueDistillationTrainer")
        logger.info(f"Student model: {sum(p.numel() for p in self.student.parameters())/1e6:.1f}M parameters")
        logger.info(f"Device: {self.device}")
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components"""
        
        # Separate parameters for different learning rates
        param_groups = []
        
        # Base model parameters
        base_params = []
        vision_params = []
        projector_params = []
        
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'vision_tower' in name:
                vision_params.append(param)
            elif 'mm_projector' in name:
                projector_params.append(param)
            else:
                base_params.append(param)
        
        # Different learning rates for different components
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.config.get('learning_rate', 2e-5),
                'weight_decay': self.config.get('weight_decay', 0.01)
            })
        
        if vision_params:
            param_groups.append({
                'params': vision_params, 
                'lr': self.config.get('vision_lr', 0.0),  # Frozen by default
                'weight_decay': 0.0
            })
            
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': self.config.get('projector_lr', 1e-4),
                'weight_decay': self.config.get('weight_decay', 0.01)
            })
        
        self.optimizer = create_optimizer(
            param_groups,
            optimizer_type=self.config.get('optimizer_type', 'adamw'),
            **self.config.get('optimizer_kwargs', {})
        )
        
        logger.info(f"Setup optimizer with {len(param_groups)} parameter groups")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        
        total_steps = self.config.get('max_steps', 50000)
        warmup_steps = self.config.get('warmup_steps', 2000)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=self.config.get('num_cycles', 0.5),
            last_epoch=-1
        )
        
        logger.info(f"Setup scheduler: {warmup_steps} warmup, {total_steps} total steps")
    
    def _setup_loss_functions(self):
        """Setup loss functions for distillation"""
        
        # Knowledge distillation loss
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=self.config.get('temperature', 4.0),
            alpha=self.config.get('alpha_kd', 0.8)
        )
        
        # Multimodal distillation loss (if using multimodal model)
        if hasattr(self.student, 'vision_tower'):
            self.mm_loss = MultimodalDistillationLoss(
                alpha_vision=self.config.get('alpha_vision', 0.2)
            )
        else:
            self.mm_loss = None
            
        logger.info("Setup loss functions")
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        
        self.callbacks = []
        
        # Memory monitor
        if self.config.get('monitor_memory', True):
            self.callbacks.append(MemoryMonitor(
                log_interval=self.config.get('memory_log_interval', 100)
            ))
        
        # Model checkpointing
        if self.config.get('save_checkpoints', True):
            self.callbacks.append(ModelCheckpoint(
                checkpoint_dir=self.config.get('output_dir', 'checkpoints/'),
                save_interval=self.config.get('save_steps', 5000),
                save_best=True
            ))
        
        # W&B logging
        if self.config.get('use_wandb', True):
            self.callbacks.append(WandBLogger(
                project=self.config.get('wandb_project', 'virtue-distillation'),
                name=self.config.get('wandb_run_name', None),
                config=self.config
            ))
            
        logger.info(f"Setup {len(self.callbacks)} callbacks")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        self.student.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                images=batch.get('images')
            )
        
        # Student forward pass
        if 'images' in batch and self.mm_loss is not None:
            # Multimodal forward
            student_outputs = self.student(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                images=batch['images'],
                labels=batch.get('labels')
            )
            
            # Compute multimodal distillation loss
            mm_loss = self.mm_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                batch=batch
            )
        else:
            # Text-only forward
            student_outputs = self.student(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch.get('labels')
            )
            mm_loss = 0.0
        
        # Compute knowledge distillation loss
        kd_loss_dict = self.kd_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs['logits'],
            labels=batch.get('labels')
        )
        
        # Total loss
        total_loss = kd_loss_dict['total_loss'] + mm_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.get('max_grad_norm', 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), 
                self.config['max_grad_norm']
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Return loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss_dict['kd_loss'].item(),
            'ce_loss': kd_loss_dict['ce_loss'].item(),
            'mm_loss': mm_loss if isinstance(mm_loss, (int, float)) else mm_loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'step': self.current_step
        }
        
        return loss_dict
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step"""
        
        self.student.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            # Teacher forward pass
            teacher_outputs = self.teacher(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                images=batch.get('images')
            )
            
            # Student forward pass
            if 'images' in batch and self.mm_loss is not None:
                student_outputs = self.student(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    images=batch['images'],
                    labels=batch.get('labels')
                )
                
                mm_loss = self.mm_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    batch=batch
                )
            else:
                student_outputs = self.student(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels')
                )
                mm_loss = 0.0
            
            # Compute losses
            kd_loss_dict = self.kd_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs['logits'],
                labels=batch.get('labels')
            )
            
            total_loss = kd_loss_dict['total_loss'] + mm_loss
        
        return {
            'val_loss': total_loss.item(),
            'val_kd_loss': kd_loss_dict['kd_loss'].item(),
            'val_ce_loss': kd_loss_dict['ce_loss'].item(),
            'val_mm_loss': mm_loss if isinstance(mm_loss, (int, float)) else mm_loss.item(),
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        
        epoch_losses = []
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.current_epoch + 1}",
            disable=not self.config.get('show_progress', True)
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss_dict = self.train_step(batch)
            epoch_losses.append(loss_dict)
            
            self.current_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{loss_dict['lr']:.2e}"
            })
            
            # Callback execution
            for callback in self.callbacks:
                callback.on_batch_end(self.current_step, loss_dict)
            
            # Validation
            if (self.val_dataloader is not None and 
                self.current_step % self.config.get('eval_steps', 2500) == 0):
                val_metrics = self.validate()
                
                # Update best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Early stopping check
            if self.current_step >= self.config.get('max_steps', 50000):
                logger.info(f"Reached max steps: {self.current_step}")
                break
            
            # Memory cleanup
            if self.current_step % 100 == 0:
                clear_gpu_cache()
        
        # Compute epoch averages
        avg_losses = {}
        if epoch_losses:
            for key in epoch_losses[0]:
                if key != 'step':
                    avg_losses[f'epoch_{key}'] = sum(d[key] for d in epoch_losses) / len(epoch_losses)
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        
        if self.val_dataloader is None:
            return {}
        
        logger.info("Running validation...")
        
        val_losses = []
        progress_bar = tqdm(
            self.val_dataloader,
            desc="Validation", 
            disable=not self.config.get('show_progress', True)
        )
        
        for batch in progress_bar:
            val_loss_dict = self.validation_step(batch)
            val_losses.append(val_loss_dict)
            
            progress_bar.set_postfix({
                'val_loss': f"{val_loss_dict['val_loss']:.4f}"
            })
        
        # Compute averages
        avg_val_losses = {}
        if val_losses:
            for key in val_losses[0]:
                avg_val_losses[key] = sum(d[key] for d in val_losses) / len(val_losses)
        
        logger.info(f"Validation complete. Loss: {avg_val_losses.get('val_loss', 0):.4f}")
        
        # Callback execution
        for callback in self.callbacks:
            callback.on_validation_end(self.current_step, avg_val_losses)
        
        return avg_val_losses
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Main training loop"""
        
        logger.info("Starting training...")
        
        # Callback initialization
        for callback in self.callbacks:
            callback.on_train_begin()
        
        max_epochs = num_epochs or self.config.get('num_epochs', 10)
        max_steps = self.config.get('max_steps', 50000)
        
        try:
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                
                # Callback
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch)
                
                # Train epoch
                epoch_metrics = self.train_epoch()
                self.training_history.append(epoch_metrics)
                
                # Log epoch results
                logger.info(f"Epoch {epoch + 1}/{max_epochs} complete:")
                for key, value in epoch_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")
                
                # Callback
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, epoch_metrics)
                
                # Check if max steps reached
                if self.current_step >= max_steps:
                    logger.info(f"Reached max steps ({max_steps}), stopping training")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Callback cleanup
            for callback in self.callbacks:
                callback.on_train_end()
            
            # Final cleanup
            clear_gpu_cache()
        
        logger.info("Training completed!")
        
        return {
            'training_history': self.training_history,
            'final_step': self.current_step,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """Save training checkpoint"""
        
        checkpoint = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'model_state_dict': self.student.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history,
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load training checkpoint"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.student.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.current_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        logger.info(f"Resumed at step {self.current_step}, epoch {self.current_epoch}")
