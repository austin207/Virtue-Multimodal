"""
Model checkpointing callbacks
"""

import os
import torch
import json
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path

class ModelCheckpoint:
    """
    Save model checkpoints during training
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 5000,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoints = []  # List of (score, path) tuples
        
        if verbose:
            print(f"ModelCheckpoint: saving to {self.checkpoint_dir}")
            print(f"  Monitoring: {monitor} ({mode})")
            print(f"  Save interval: {save_interval} steps")
            print(f"  Save top {save_top_k} models")
    
    def _is_better_score(self, score: float) -> bool:
        """Check if score is better than current best"""
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def _should_save_checkpoint(self, step: int, metrics: Dict[str, Any]) -> bool:
        """Determine if checkpoint should be saved"""
        # Regular interval saving
        if step % self.save_interval == 0:
            return True
        
        # Best model saving
        if self.save_best and self.monitor in metrics:
            score = metrics[self.monitor]
            if self._is_better_score(score):
                return True
        
        return False
    
    def save_checkpoint(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Dict[str, Any] = None,
        extra_state: Dict[str, Any] = None
    ) -> str:
        """Save model checkpoint"""
        
        metrics = metrics or {}
        extra_state = extra_state or {}
        
        # Create checkpoint data
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'extra_state': extra_state
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Generate checkpoint filename
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'epoch': epoch,
            'metrics': metrics,
            'model_size_mb': os.path.getsize(checkpoint_path) / 1024**2
        }
        
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_name}")
            if metrics:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"  Metrics: {metric_str}")
        
        return str(checkpoint_path)
    
    def on_batch_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for batch end"""
        # This would be called by the trainer
        pass
    
    def on_validation_end(self, step: int, metrics: Dict[str, Any]):
        """Callback for validation end"""
        if not metrics:
            return
        
        # Check if this is a new best model
        if self.save_best and self.monitor in metrics:
            score = metrics[self.monitor]
            
            if self._is_better_score(score):
                self.best_score = score
                
                if self.verbose:
                    print(f"New best {self.monitor}: {score:.4f}")
                
                # Mark for saving (trainer will handle actual saving)
                metrics['_save_best'] = True
        
        # Manage top-k checkpoints
        if self.monitor in metrics:
            self._update_top_k_checkpoints(step, metrics[self.monitor])
    
    def _update_top_k_checkpoints(self, step: int, score: float):
        """Update list of top-k checkpoints"""
        
        checkpoint_info = (score, step, f"checkpoint_step_{step}.pt")
        
        # Add to list
        self.best_checkpoints.append(checkpoint_info)
        
        # Sort by score (best first)
        if self.mode == 'min':
            self.best_checkpoints.sort(key=lambda x: x[0])
        else:
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove excess checkpoints
        if len(self.best_checkpoints) > self.save_top_k:
            to_remove = self.best_checkpoints[self.save_top_k:]
            self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
            
            # Delete old checkpoint files
            for _, _, filename in to_remove:
                checkpoint_path = self.checkpoint_dir / filename
                metadata_path = checkpoint_path.with_suffix('.json')
                
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    if self.verbose:
                        print(f"Removed old checkpoint: {filename}")
                except Exception as e:
                    print(f"Error removing checkpoint {filename}: {e}")
    
    def save_best_model(self, model: torch.nn.Module, filename: str = "best_model.pt"):
        """Save the best model separately"""
        
        best_path = self.checkpoint_dir / filename
        torch.save(model.state_dict(), best_path)
        
        if self.verbose:
            print(f"Saved best model: {filename} (score: {self.best_score:.4f})")
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint"""
        if not self.best_checkpoints:
            return None
        
        _, _, filename = self.best_checkpoints[0]
        return str(self.checkpoint_dir / filename)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            metadata_file = checkpoint_file.with_suffix('.json')
            
            checkpoint_info = {
                'path': str(checkpoint_file),
                'name': checkpoint_file.name,
                'size_mb': checkpoint_file.stat().st_size / 1024**2
            }
            
            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    checkpoint_info.update(metadata)
                except Exception as e:
                    print(f"Error reading metadata for {checkpoint_file.name}: {e}")
            
            checkpoints.append(checkpoint_info)
        
        # Sort by step
        checkpoints.sort(key=lambda x: x.get('step', 0))
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoints, keeping only the last N"""
        
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Keep best checkpoints and last N
        to_keep = set()
        
        # Keep best checkpoints
        for _, _, filename in self.best_checkpoints:
            to_keep.add(filename)
        
        # Keep last N checkpoints
        for checkpoint in checkpoints[-keep_last_n:]:
            to_keep.add(checkpoint['name'])
        
        # Remove others
        removed_count = 0
        for checkpoint in checkpoints[:-keep_last_n]:
            if checkpoint['name'] not in to_keep:
                checkpoint_path = Path(checkpoint['path'])
                metadata_path = checkpoint_path.with_suffix('.json')
                
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {checkpoint['name']}: {e}")
        
        if self.verbose and removed_count > 0:
            print(f"Cleaned up {removed_count} old checkpoints")

class BestModelCheckpoint:
    """
    Simplified checkpoint callback that only saves the best model
    """
    
    def __init__(
        self,
        filepath: str = "best_model.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_optimizer: bool = False,
        verbose: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.verbose = verbose
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"BestModelCheckpoint: {filepath} (monitoring {monitor}, mode={mode})")
    
    def _is_better(self, score: float) -> bool:
        """Check if score is better"""
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def on_validation_end(
        self, 
        step: int, 
        metrics: Dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None
    ):
        """Check and save if best model"""
        
        if self.monitor not in metrics:
            return
        
        score = metrics[self.monitor]
        
        if self._is_better(score):
            self.best_score = score
            
            # Prepare checkpoint data
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'best_score': self.best_score,
                'monitor': self.monitor,
                'metrics': metrics
            }
            
            if self.save_optimizer and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            # Save checkpoint
            torch.save(checkpoint, self.filepath)
            
            if self.verbose:
                print(f"Saved new best model: {self.monitor}={score:.4f}")

class AutoSaveCallback:
    """
    Automatic saving callback with configurable triggers
    """
    
    def __init__(
        self,
        save_dir: str = "auto_saves",
        save_every_n_steps: int = 1000,
        save_every_n_minutes: int = 30,
        max_saves: int = 10
    ):
        self.save_dir = Path(save_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_minutes = save_every_n_minutes
        self.max_saves = max_saves
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.last_save_time = 0
        self.save_count = 0
    
    def should_save(self, step: int) -> bool:
        """Check if model should be auto-saved"""
        
        import time
        current_time = time.time()
        
        # Step-based saving
        if step % self.save_every_n_steps == 0:
            return True
        
        # Time-based saving
        if (current_time - self.last_save_time) / 60 > self.save_every_n_minutes:
            return True
        
        return False
    
    def save(self, model: torch.nn.Module, step: int):
        """Auto-save the model"""
        
        import time
        
        filename = f"auto_save_step_{step}.pt"
        filepath = self.save_dir / filename
        
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'save_time': time.time()
        }, filepath)
        
        self.last_save_time = time.time()
        self.save_count += 1
        
        # Clean up old saves
        if self.save_count > self.max_saves:
            self._cleanup_old_saves()
        
        print(f"Auto-saved: {filename}")
    
    def _cleanup_old_saves(self):
        """Remove oldest auto-saves"""
        
        saves = sorted(self.save_dir.glob("auto_save_step_*.pt"))
        
        while len(saves) > self.max_saves:
            oldest = saves.pop(0)
            try:
                oldest.unlink()
                print(f"Removed old auto-save: {oldest.name}")
            except Exception as e:
                print(f"Error removing {oldest.name}: {e}")
