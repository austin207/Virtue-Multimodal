"""
Regularization losses for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class ActivationRegularization(nn.Module):
    """
    Regularization on model activations to prevent overfitting
    """
    
    def __init__(
        self, 
        regularization_type: str = 'l2',
        lambda_reg: float = 1e-4,
        target_layers: Optional[List[str]] = None
    ):
        super().__init__()
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        self.target_layers = target_layers or ['attention', 'mlp']
    
    def forward(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            activations: Dictionary of layer activations
        
        Returns:
            Regularization loss
        """
        
        total_loss = 0.0
        num_layers = 0
        
        for layer_name, activation in activations.items():
            if any(target in layer_name for target in self.target_layers):
                
                if self.regularization_type == 'l1':
                    loss = torch.abs(activation).mean()
                elif self.regularization_type == 'l2':
                    loss = torch.square(activation).mean()
                elif self.regularization_type == 'entropy':
                    # Encourage diversity in activations
                    probs = F.softmax(activation, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    loss = -entropy  # Negative entropy (maximize diversity)
                else:
                    raise ValueError(f"Unknown regularization type: {self.regularization_type}")
                
                total_loss += loss
                num_layers += 1
        
        if num_layers > 0:
            return self.lambda_reg * total_loss / num_layers
        else:
            return torch.tensor(0.0, device=next(iter(activations.values())).device)

class AttentionRegularization(nn.Module):
    """
    Regularization on attention patterns
    """
    
    def __init__(
        self,
        entropy_weight: float = 1e-3,
        sparsity_weight: float = 1e-3,
        diversity_weight: float = 1e-3
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        
        Returns:
            Attention regularization loss
        """
        
        total_loss = 0.0
        
        # Entropy regularization (encourage concentrated attention)
        if self.entropy_weight > 0:
            attention_probs = F.softmax(attention_weights, dim=-1)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            entropy_loss = entropy.mean()
            total_loss += self.entropy_weight * entropy_loss
        
        # Sparsity regularization (encourage sparse attention)
        if self.sparsity_weight > 0:
            attention_probs = F.softmax(attention_weights, dim=-1)
            sparsity_loss = torch.abs(attention_probs).mean()
            total_loss += self.sparsity_weight * sparsity_loss
        
        # Diversity regularization (encourage head diversity)
        if self.diversity_weight > 0:
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Flatten heads for comparison
            attention_flat = attention_weights.view(batch_size, num_heads, -1)
            
            # Compute pairwise cosine similarities between heads
            attention_norm = F.normalize(attention_flat, dim=-1)
            similarities = torch.matmul(attention_norm, attention_norm.transpose(-2, -1))
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(num_heads, device=attention_weights.device).bool()
            off_diagonal = similarities[:, mask].view(batch_size, num_heads, num_heads - 1)
            
            # Penalize high similarities (encourage diversity)
            diversity_loss = off_diagonal.abs().mean()
            total_loss += self.diversity_weight * diversity_loss
        
        return total_loss

class WeightDecayRegularization(nn.Module):
    """
    Custom weight decay with layer-specific penalties
    """
    
    def __init__(
        self,
        layer_penalties: Optional[Dict[str, float]] = None,
        default_penalty: float = 1e-4
    ):
        super().__init__()
        self.layer_penalties = layer_penalties or {}
        self.default_penalty = default_penalty
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Args:
            model: The model to regularize
        
        Returns:
            Weight decay loss
        """
        
        total_loss = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:  # Skip biases and 1D params
                
                # Get penalty for this layer
                penalty = self.default_penalty
                for layer_name, layer_penalty in self.layer_penalties.items():
                    if layer_name in name:
                        penalty = layer_penalty
                        break
                
                # L2 penalty on weights
                weight_loss = penalty * torch.norm(param, p=2)
                total_loss += weight_loss
        
        return total_loss

class GradientPenalty(nn.Module):
    """
    Gradient penalty for training stability
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(
        self, 
        model: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty (used in WGAN-GP style training)
        """
        
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)
        
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Forward pass through model
        d_interpolates = model(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty

class TemperatureRegularization(nn.Module):
    """
    Regularization on temperature parameters in attention
    """
    
    def __init__(
        self, 
        target_temperature: float = 1.0,
        penalty_weight: float = 1e-3
    ):
        super().__init__()
        self.target_temperature = target_temperature
        self.penalty_weight = penalty_weight
    
    def forward(self, temperatures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temperatures: Temperature parameters to regularize
        
        Returns:
            Temperature regularization loss
        """
        
        # L2 penalty around target temperature
        penalty = F.mse_loss(
            temperatures, 
            torch.full_like(temperatures, self.target_temperature)
        )
        
        return self.penalty_weight * penalty

class FeatureMatchingRegularization(nn.Module):
    """
    Feature matching regularization between different layers
    """
    
    def __init__(self, lambda_fm: float = 1e-3):
        super().__init__()
        self.lambda_fm = lambda_fm
    
    def forward(
        self, 
        features_1: torch.Tensor, 
        features_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage similar statistical properties between features
        """
        
        # Compute first and second moments
        mean_1 = features_1.mean(dim=[0, 1])
        mean_2 = features_2.mean(dim=[0, 1])
        
        var_1 = features_1.var(dim=[0, 1])
        var_2 = features_2.var(dim=[0, 1])
        
        # Feature matching loss
        mean_loss = F.mse_loss(mean_1, mean_2)
        var_loss = F.mse_loss(var_1, var_2)
        
        return self.lambda_fm * (mean_loss + var_loss)

class ConsistencyRegularization(nn.Module):
    """
    Consistency regularization for robust training
    """
    
    def __init__(self, consistency_weight: float = 1e-2):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self, 
        predictions_1: torch.Tensor, 
        predictions_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage consistent predictions under different augmentations
        """
        
        # Convert to probabilities
        probs_1 = F.softmax(predictions_1, dim=-1)
        probs_2 = F.softmax(predictions_2, dim=-1)
        
        # KL divergence between predictions
        consistency_loss = F.kl_div(
            probs_1.log(), probs_2.detach(), 
            reduction='batchmean'
        )
        
        return self.consistency_weight * consistency_loss
