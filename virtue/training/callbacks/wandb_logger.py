"""
Experiment tracking callbacks (W&B, TensorBoard)
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

class WandBLogger:
    """
    Weights & Biases logging callback
    """
    
    def __init__(
        self,
        project: str = "virtue-distillation",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        log_interval: int = 100,
        log_model: bool = True,
        watch_model: bool = True,
        enabled: bool = True
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.log_interval = log_interval
        self.log_model = log_model
        self.watch_model = watch_model
        self.enabled = enabled and WANDB_AVAILABLE
        
        self.run = None
        self.step_count = 0
        
        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available, logging disabled")
            else:
                print("W&B logging disabled")
            return
        
        self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize W&B run"""
        
        if not self.enabled:
            return
        
        try:
            self.run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                reinit=True
            )
            
            print(f"W&B initialized: {self.run.url}")
            
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            self.enabled = False
    
    def watch_model(self, model):
        """Watch model for gradients and parameters"""
        
        if not self.enabled or not self.watch_model:
            return
        
        try:
            wandb.watch(model, log="all", log_freq=self.log_interval)
            print("W&B watching model")
        except Exception as e:
            print(f"Error watching model: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B"""
        
        if not self.enabled:
            return
        
        try:
            # Filter out non-numeric values
            filtered_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    filtered_metrics[key] = value
                elif hasattr(value, 'item'):  # Tensor with single value
                    try:
                        filtered_metrics[key] = value.item()
                    except:
                        pass
            
            wandb.log(filtered_metrics, step=step)
            
        except Exception as e:
            print(f"Error logging metrics to W&B: {e}")
    
    def log_image(self, key: str, image, caption: str = "", step: Optional[int] = None):
        """Log image to W&B"""
        
        if not self.enabled:
            return
        
        try:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
        except Exception as e:
            print(f"Error logging image to W&B: {e}")
    
    def log_text(self, key: str, text: str, step: Optional[int] = None):
        """Log text to W&B"""
        
        if not self.enabled:
            return
        
        try:
            wandb.log({key: wandb.Text(text)}, step=step)
        except Exception as e:
            print(f"Error logging text to W&B: {e}")
    
    def log_model(self, model_path: str, name: str = "model", metadata: Dict = None):
        """Log model artifact to W&B"""
        
        if not self.enabled or not self.log_model:
            return
        
        try:
            artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            print(f"Model logged to W&B: {name}")
        except Exception as e:
            print(f"Error logging model to W&B: {e}")
    
    def on_train_begin(self):
        """Callback for training start"""
        if self.enabled:
            print("Started W&B logging")
    
    def on_batch_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for batch end"""
        
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self.log_metrics(metrics, step=step)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Callback for epoch end"""
        
        epoch_metrics = {f"epoch_{k}": v for k, v in metrics.items()}
        epoch_metrics["epoch"] = epoch
        
        self.log_metrics(epoch_metrics, step=self.step_count)
    
    def on_validation_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for validation end"""
        
        self.log_metrics(metrics, step=step)
    
    def on_train_end(self):
        """Callback for training end"""
        
        if not self.enabled:
            return
        
        try:
            if self.run:
                self.run.finish()
            print("W&B logging finished")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")

class TensorBoardLogger:
    """
    TensorBoard logging callback
    """
    
    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        log_interval: int = 100,
        enabled: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        
        self.writer = None
        self.step_count = 0
        
        if not self.enabled:
            if not TENSORBOARD_AVAILABLE:
                print("Warning: tensorboard not available, logging disabled")
            else:
                print("TensorBoard logging disabled")
            return
        
        self._initialize_tensorboard()
    
    def _initialize_tensorboard(self):
        """Initialize TensorBoard writer"""
        
        if not self.enabled:
            return
        
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            print(f"TensorBoard initialized: {self.log_dir}")
            
        except Exception as e:
            print(f"Failed to initialize TensorBoard: {e}")
            self.enabled = False
    
    def log_scalars(self, metrics: Dict[str, Any], step: int):
        """Log scalar metrics"""
        
        if not self.enabled:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                elif hasattr(value, 'item'):
                    try:
                        self.writer.add_scalar(key, value.item(), step)
                    except:
                        pass
            
            self.writer.flush()
            
        except Exception as e:
            print(f"Error logging scalars to TensorBoard: {e}")
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        
        if not self.enabled:
            return
        
        try:
            self.writer.add_histogram(tag, values, step)
            self.writer.flush()
        except Exception as e:
            print(f"Error logging histogram to TensorBoard: {e}")
    
    def log_image(self, tag: str, image, step: int):
        """Log image"""
        
        if not self.enabled:
            return
        
        try:
            self.writer.add_image(tag, image, step)
            self.writer.flush()
        except Exception as e:
            print(f"Error logging image to TensorBoard: {e}")
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        
        if not self.enabled:
            return
        
        try:
            self.writer.add_text(tag, text, step)
            self.writer.flush()
        except Exception as e:
            print(f"Error logging text to TensorBoard: {e}")
    
    def log_model_graph(self, model, input_tensor):
        """Log model graph"""
        
        if not self.enabled:
            return
        
        try:
            self.writer.add_graph(model, input_tensor)
            self.writer.flush()
            print("Model graph logged to TensorBoard")
        except Exception as e:
            print(f"Error logging model graph: {e}")
    
    def on_batch_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for batch end"""
        
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self.log_scalars(metrics, step)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Callback for epoch end"""
        
        epoch_metrics = {f"epoch_{k}": v for k, v in metrics.items()}
        self.log_scalars(epoch_metrics, epoch)
    
    def on_validation_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for validation end"""
        
        self.log_scalars(metrics, step)
    
    def on_train_end(self):
        """Callback for training end"""
        
        if not self.enabled:
            return
        
        try:
            if self.writer:
                self.writer.close()
            print("TensorBoard logging finished")
        except Exception as e:
            print(f"Error closing TensorBoard writer: {e}")

class MultiLogger:
    """
    Combine multiple loggers
    """
    
    def __init__(self, loggers: List):
        self.loggers = loggers
        self.enabled_loggers = [logger for logger in loggers if getattr(logger, 'enabled', True)]
        
        print(f"MultiLogger initialized with {len(self.enabled_loggers)} active loggers")
    
    def on_train_begin(self):
        """Callback for training start"""
        for logger in self.enabled_loggers:
            if hasattr(logger, 'on_train_begin'):
                logger.on_train_begin()
    
    def on_batch_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for batch end"""
        for logger in self.enabled_loggers:
            if hasattr(logger, 'on_batch_end'):
                logger.on_batch_end(step, metrics)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Callback for epoch end"""
        for logger in self.enabled_loggers:
            if hasattr(logger, 'on_epoch_end'):
                logger.on_epoch_end(epoch, metrics)
    
    def on_validation_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for validation end"""
        for logger in self.enabled_loggers:
            if hasattr(logger, 'on_validation_end'):
                logger.on_validation_end(step, metrics)
    
    def on_train_end(self):
        """Callback for training end"""
        for logger in self.enabled_loggers:
            if hasattr(logger, 'on_train_end'):
                logger.on_train_end()
    
    def log_model(self, model_path: str, **kwargs):
        """Log model to all compatible loggers"""
        for logger in self.enabled_loggers:
            if hasattr(logger, 'log_model'):
                logger.log_model(model_path, **kwargs)

class EarlyStopping:
    """
    Early stopping callback
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.patience_count = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def _is_better(self, score: float) -> bool:
        """Check if score is better"""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)
    
    def on_validation_end(self, step: int, metrics: Dict[str, Any], model=None):
        """Check for early stopping"""
        
        if self.monitor not in metrics:
            return
        
        score = metrics[self.monitor]
        
        if self._is_better(score):
            self.best_score = score
            self.patience_count = 0
            
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
            
            if self.verbose:
                print(f"EarlyStopping: new best {self.monitor}: {score:.4f}")
        
        else:
            self.patience_count += 1
            
            if self.verbose:
                print(f"EarlyStopping: patience {self.patience_count}/{self.patience}")
            
            if self.patience_count >= self.patience:
                self.should_stop = True
                self.stopped_epoch = step
                
                if self.verbose:
                    print(f"EarlyStopping: stopping training at step {step}")
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights and model:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Restored best model weights")

class LearningRateFinder:
    """
    Learning rate finder callback
    """
    
    def __init__(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_steps: int = 100,
        beta: float = 0.98
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps
        self.beta = beta
        
        self.lr_schedule = []
        self.losses = []
        self.avg_losses = []
        self.step_count = 0
        self.best_loss = float('inf')
        
        # Generate learning rate schedule
        self.lr_mult = (end_lr / start_lr) ** (1.0 / num_steps)
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step"""
        return self.start_lr * (self.lr_mult ** step)
    
    def on_batch_end(self, step: int, loss: float, optimizer):
        """Update learning rate and track loss"""
        
        if self.step_count >= self.num_steps:
            return
        
        # Record current state
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_schedule.append(current_lr)
        self.losses.append(loss)
        
        # Compute smoothed loss
        if self.step_count == 0:
            avg_loss = loss
        else:
            avg_loss = self.beta * self.avg_losses[-1] + (1 - self.beta) * loss
        
        self.avg_losses.append(avg_loss)
        
        # Stop if loss explodes
        if loss > 4 * self.best_loss:
            print(f"LR Finder: Loss exploded at lr={current_lr:.2e}, stopping")
            return
        
        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
        
        # Update learning rate for next step
        self.step_count += 1
        if self.step_count < self.num_steps:
            new_lr = self.get_lr(self.step_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    
    def plot(self, save_path: Optional[str] = None):
        """Plot learning rate vs loss"""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.semilogx(self.lr_schedule, self.avg_losses)
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Rate Finder')
            ax.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"LR Finder plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available, cannot plot")
    
    def suggest_lr(self) -> float:
        """Suggest optimal learning rate"""
        
        if len(self.avg_losses) < 2:
            return self.start_lr
        
        # Find steepest gradient
        gradients = []
        for i in range(1, len(self.avg_losses)):
            grad = (self.avg_losses[i] - self.avg_losses[i-1]) / (
                self.lr_schedule[i] - self.lr_schedule[i-1]
            )
            gradients.append(grad)
        
        # Find minimum gradient (steepest descent)
        min_grad_idx = gradients.index(min(gradients))
        suggested_lr = self.lr_schedule[min_grad_idx + 1]
        
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr
