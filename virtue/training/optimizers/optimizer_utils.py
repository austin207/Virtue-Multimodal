"""
Optimizer utilities and configurations
"""

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD, RMSprop
from typing import Dict, List, Union, Optional, Any
import re

def get_optimizer_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    learning_rate: float = 2e-5,
    vision_lr_multiplier: float = 0.0,
    projector_lr_multiplier: float = 5.0,
    no_decay_keywords: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates and weight decay
    
    Args:
        model: The model to optimize
        weight_decay: Default weight decay
        learning_rate: Base learning rate
        vision_lr_multiplier: Learning rate multiplier for vision encoder
        projector_lr_multiplier: Learning rate multiplier for projector
        no_decay_keywords: Parameters matching these keywords won't have weight decay
    
    Returns:
        List of parameter groups for optimizer
    """
    
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln", "bn"]
    
    # Parameter groups
    decay_params = []
    no_decay_params = []
    vision_decay_params = []
    vision_no_decay_params = []
    projector_decay_params = []
    projector_no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter should have weight decay
        no_decay = any(keyword in name.lower() for keyword in no_decay_keywords)
        
        # Categorize parameters
        if 'vision_tower' in name or 'vision_encoder' in name:
            if no_decay:
                vision_no_decay_params.append(param)
            else:
                vision_decay_params.append(param)
        elif 'mm_projector' in name or 'projector' in name:
            if no_decay:
                projector_no_decay_params.append(param)
            else:
                projector_decay_params.append(param)
        else:
            if no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    # Base model parameters
    if decay_params:
        param_groups.append({
            'params': decay_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'base_decay'
        })
    
    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': learning_rate,
            'weight_decay': 0.0,
            'name': 'base_no_decay'
        })
    
    # Vision encoder parameters
    vision_lr = learning_rate * vision_lr_multiplier
    if vision_decay_params:
        param_groups.append({
            'params': vision_decay_params,
            'lr': vision_lr,
            'weight_decay': weight_decay if vision_lr_multiplier > 0 else 0.0,
            'name': 'vision_decay'
        })
    
    if vision_no_decay_params:
        param_groups.append({
            'params': vision_no_decay_params,
            'lr': vision_lr,
            'weight_decay': 0.0,
            'name': 'vision_no_decay'
        })
    
    # Projector parameters
    projector_lr = learning_rate * projector_lr_multiplier
    if projector_decay_params:
        param_groups.append({
            'params': projector_decay_params,
            'lr': projector_lr,
            'weight_decay': weight_decay,
            'name': 'projector_decay'
        })
    
    if projector_no_decay_params:
        param_groups.append({
            'params': projector_no_decay_params,
            'lr': projector_lr,
            'weight_decay': 0.0,
            'name': 'projector_no_decay'
        })
    
    # Log parameter group info
    total_params = sum(len(group['params']) for group in param_groups)
    print(f"Created {len(param_groups)} parameter groups with {total_params} total parameters:")
    
    for i, group in enumerate(param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i} ({group['name']}): {len(group['params'])} tensors, "
              f"{num_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    
    return param_groups

def create_optimizer(
    param_groups: Union[List[Dict], nn.Module],
    optimizer_type: str = 'adamw',
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with specified configuration
    
    Args:
        param_groups: Parameter groups or model
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Configured optimizer
    """
    
    # If model is passed instead of param groups, create default groups
    if isinstance(param_groups, nn.Module):
        param_groups = get_optimizer_groups(
            param_groups, 
            weight_decay=weight_decay,
            learning_rate=learning_rate
        )
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.95)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'adam':
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            momentum=kwargs.get('momentum', 0.0)
        )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    print(f"Created {optimizer_type.upper()} optimizer with {len(param_groups)} parameter groups")
    
    return optimizer

def get_layerwise_decay_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    num_layers: int,
    decay_rate: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Create layerwise learning rate decay groups
    
    Args:
        model: The model
        base_lr: Base learning rate  
        weight_decay: Weight decay
        num_layers: Number of transformer layers
        decay_rate: Decay rate per layer
    
    Returns:
        Parameter groups with layerwise decay
    """
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Extract layer number
        layer_num = None
        layer_match = re.search(r'layers?[.\[](\d+)[.\]]', name)
        if layer_match:
            layer_num = int(layer_match.group(1))
        
        # Calculate learning rate
        if layer_num is not None:
            lr = base_lr * (decay_rate ** (num_layers - layer_num - 1))
        else:
            lr = base_lr  # Default for embeddings, output layers, etc.
        
        # Weight decay
        no_decay = any(keyword in name.lower() 
                      for keyword in ["bias", "layernorm", "layer_norm"])
        wd = 0.0 if no_decay else weight_decay
        
        param_groups.append({
            'params': [param],
            'lr': lr,
            'weight_decay': wd,
            'name': f'{name}_layer_{layer_num}'
        })
    
    return param_groups

class OptimizerScheduler:
    """
    Combined optimizer and scheduler wrapper
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_count = 0
    
    def step(self, closure=None):
        """Optimizer step"""
        self.optimizer.step(closure)
        self.step_count += 1
    
    def scheduler_step(self, *args, **kwargs):
        """Scheduler step"""
        if self.scheduler is not None:
            self.scheduler.step(*args, **kwargs)
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dict"""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step_count = state_dict['step_count']
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])
    
    def get_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self):
        """Get last learning rates"""
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            return self.get_lr()

def print_optimizer_info(optimizer: torch.optim.Optimizer, model: nn.Module = None):
    """Print detailed optimizer information"""
    
    print(f"\n{'='*50}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"{'='*50}")
    
    total_params = 0
    
    for i, group in enumerate(optimizer.param_groups):
        group_params = sum(p.numel() for p in group['params'])
        total_params += group_params
        
        print(f"\nParameter Group {i}:")
        print(f"  Parameters: {len(group['params'])} tensors ({group_params:,} params)")
        print(f"  Learning Rate: {group['lr']:.2e}")
        print(f"  Weight Decay: {group['weight_decay']}")
        
        if 'name' in group:
            print(f"  Name: {group['name']}")
        
        # Additional optimizer-specific info
        if hasattr(optimizer, 'state_dict'):
            state = optimizer.state_dict()
            if 'param_groups' in state:
                group_state = state['param_groups'][i]
                for key, value in group_state.items():
                    if key not in ['params', 'lr', 'weight_decay']:
                        print(f"  {key}: {value}")
    
    print(f"\nTotal Parameters: {total_params:,}")
    
    if model is not None:
        model_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Parameters: {model_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Frozen Parameters: {model_params - trainable_params:,}")
    
    print(f"{'='*50}\n")
