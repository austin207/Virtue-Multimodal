"""
Knowledge distillation loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class DistillationLoss(nn.Module):
    """
    Base class for distillation losses
    """
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class KnowledgeDistillationLoss(DistillationLoss):
    """
    Standard knowledge distillation loss combining soft and hard targets
    """
    
    def __init__(
        self, 
        temperature: float = 4.0, 
        alpha: float = 0.8,
        reduction: str = 'batchmean'
    ):
        super().__init__(temperature)
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size] 
            labels: [batch_size, seq_len] - ground truth tokens
        
        Returns:
            Dictionary of losses
        """
        
        # Flatten logits for loss computation
        student_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        
        # Soft target loss (KL divergence)
        student_log_probs = F.log_softmax(student_flat / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_flat / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction=self.reduction
        ) * (self.temperature ** 2)
        
        # Hard target loss (Cross entropy)
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            # Shift for causal LM
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
        
        return {
            'total_loss': total_loss,
            'kd_loss': kl_loss,
            'ce_loss': ce_loss,
        }

class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss for intermediate representations
    """
    
    def __init__(self, layer_mapping: Optional[Dict[int, int]] = None):
        super().__init__()
        self.layer_mapping = layer_mapping or {}
        
        # Projection layers for dimension matching
        self.projections = nn.ModuleDict()
    
    def add_projection(self, name: str, input_dim: int, output_dim: int):
        """Add projection layer for dimension matching"""
        if input_dim != output_dim:
            self.projections[name] = nn.Linear(input_dim, output_dim)
        else:
            self.projections[name] = nn.Identity()
    
    def forward(
        self, 
        student_features: Dict[str, torch.Tensor], 
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            student_features: Dictionary of student hidden states
            teacher_features: Dictionary of teacher hidden states
        
        Returns:
            Feature distillation loss
        """
        
        total_loss = 0.0
        num_losses = 0
        
        for layer_name in student_features:
            if layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # Apply projection if needed
                if layer_name in self.projections:
                    student_feat = self.projections[layer_name](student_feat)
                
                # MSE loss between features
                loss = F.mse_loss(student_feat, teacher_feat.detach())
                total_loss += loss
                num_losses += 1
        
        return total_loss / max(num_losses, 1)

class AttentionDistillationLoss(nn.Module):
    """
    Attention-based distillation loss
    """
    
    def __init__(self, normalize_attention: bool = True):
        super().__init__()
        self.normalize_attention = normalize_attention
    
    def forward(
        self, 
        student_attentions: torch.Tensor, 
        teacher_attentions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_attentions: [batch, heads, seq_len, seq_len]
            teacher_attentions: [batch, heads, seq_len, seq_len]
        
        Returns:
            Attention distillation loss
        """
        
        if self.normalize_attention:
            # Normalize attention matrices
            student_attentions = F.softmax(student_attentions, dim=-1)
            teacher_attentions = F.softmax(teacher_attentions.detach(), dim=-1)
        
        # MSE loss between attention matrices
        loss = F.mse_loss(student_attentions, teacher_attentions)
        
        return loss

class ProgressiveDistillationLoss(nn.Module):
    """
    Progressive distillation with curriculum learning
    """
    
    def __init__(
        self, 
        base_loss: nn.Module,
        warmup_steps: int = 5000,
        difficulty_schedule: str = 'linear'
    ):
        super().__init__()
        self.base_loss = base_loss
        self.warmup_steps = warmup_steps
        self.difficulty_schedule = difficulty_schedule
        self.current_step = 0
    
    def get_difficulty_weight(self) -> float:
        """Get current difficulty weight based on training progress"""
        
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            
            if self.difficulty_schedule == 'linear':
                return progress
            elif self.difficulty_schedule == 'cosine':
                return 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)))
            else:
                return 1.0
        
        return 1.0
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with difficulty weighting"""
        
        base_losses = self.base_loss(*args, **kwargs)
        difficulty_weight = self.get_difficulty_weight()
        
        # Apply difficulty weighting
        weighted_losses = {}
        for key, loss in base_losses.items():
            if 'total' in key or 'kd' in key:
                weighted_losses[key] = loss * difficulty_weight
            else:
                weighted_losses[key] = loss
        
        return weighted_losses
    
    def step(self):
        """Update step counter"""
        self.current_step += 1

class AdaptiveDistillationLoss(nn.Module):
    """
    Adaptive distillation loss that adjusts based on student performance
    """
    
    def __init__(
        self, 
        base_loss: nn.Module,
        adaptation_rate: float = 0.01,
        min_alpha: float = 0.1,
        max_alpha: float = 0.9
    ):
        super().__init__()
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Running average of student-teacher agreement
        self.agreement_ema = 0.5
        self.alpha = 0.5
    
    def compute_agreement(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> float:
        """Compute agreement between student and teacher predictions"""
        
        with torch.no_grad():
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # Jensen-Shannon divergence as agreement metric
            m = 0.5 * (student_probs + teacher_probs)
            kl1 = F.kl_div(F.log_softmax(student_logits, dim=-1), m, reduction='batchmean')
            kl2 = F.kl_div(F.log_softmax(teacher_logits, dim=-1), m, reduction='batchmean')
            
            js_div = 0.5 * (kl1 + kl2)
            agreement = torch.exp(-js_div).item()
            
        return agreement
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive alpha"""
        
        # Compute current agreement
        agreement = self.compute_agreement(student_logits, teacher_logits)
        
        # Update EMA of agreement
        self.agreement_ema = (1 - self.adaptation_rate) * self.agreement_ema + \
                           self.adaptation_rate * agreement
        
        # Adapt alpha based on agreement (higher agreement -> more hard targets)
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (1 - self.agreement_ema)
        
        # Update base loss alpha if it has one
        if hasattr(self.base_loss, 'alpha'):
            self.base_loss.alpha = self.alpha
        
        # Compute loss
        losses = self.base_loss(student_logits, teacher_logits, **kwargs)
        losses['adaptive_alpha'] = torch.tensor(self.alpha)
        losses['agreement'] = torch.tensor(self.agreement_ema)
        
        return losses
