"""
Learning rate schedulers for training
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Union

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> _LRScheduler:
    """
    Create cosine schedule with warmup
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch number
    
    Returns:
        Cosine scheduler with warmup
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> _LRScheduler:
    """
    Create linear schedule with warmup
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        last_epoch: Last epoch number
    
    Returns:
        Linear scheduler with warmup
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 2.0,
    last_epoch: int = -1
) -> _LRScheduler:
    """
    Create polynomial decay schedule with warmup
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps  
        num_training_steps: Total training steps
        power: Polynomial power
        last_epoch: Last epoch number
    
    Returns:
        Polynomial decay scheduler with warmup
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        return max(0.0, (1.0 - progress) ** power)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class WarmupScheduler(_LRScheduler):
    """
    Generic warmup scheduler that wraps another scheduler
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: _LRScheduler,
        warmup_steps: int,
        warmup_factor: float = 0.1,
        last_epoch: int = -1
    ):
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_steps
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler
            self.base_scheduler.last_epoch = self.last_epoch - self.warmup_steps
            return self.base_scheduler.get_lr()

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rates
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates"""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self) -> List[float]:
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            return [
                base_lr + (self.max_lr - base_lr) * (
                    1 + math.cos(
                        math.pi * (self.step_in_cycle - self.warmup_steps) / 
                        (self.cur_cycle_steps - self.warmup_steps)
                    )
                ) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class ExponentialScheduler(_LRScheduler):
    """
    Exponential learning rate decay
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        last_epoch: int = -1
    ):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.staircase:
            power = self.last_epoch // self.decay_steps
        else:
            power = self.last_epoch / self.decay_steps
        
        return [base_lr * (self.decay_rate ** power) for base_lr in self.base_lrs]

class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler (used in Transformer paper)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = max(1, self.last_epoch)
        scale = self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
        
        return [base_lr * scale for base_lr in self.base_lrs]

class CyclicScheduler(_LRScheduler):
    """
    Cyclic learning rate scheduler
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int,
        step_size_down: int = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        
        self.total_size = self.step_size_up + self.step_size_down
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_fn = lambda x: 1.0
            scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.0 ** (cycle - 1))
            scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            scale_fn = lambda x: self.gamma ** x
            scale_mode = 'iterations'
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        if scale_mode == 'cycle':
            scale_factor = scale_fn(cycle)
        else:
            scale_factor = scale_fn(self.last_epoch)
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        
        return [lr for _ in self.base_lrs]

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create schedulers
    
    Args:
        optimizer: The optimizer
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Configured scheduler
    """
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'cosine_warmup':
        return get_cosine_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == 'linear_warmup':
        return get_linear_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == 'polynomial_warmup':
        return get_polynomial_decay_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == 'cosine_restarts':
        return CosineAnnealingWarmupRestarts(optimizer, **kwargs)
    elif scheduler_type == 'exponential':
        return ExponentialScheduler(optimizer, **kwargs)
    elif scheduler_type == 'noam':
        return NoamScheduler(optimizer, **kwargs)
    elif scheduler_type == 'cyclic':
        return CyclicScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
