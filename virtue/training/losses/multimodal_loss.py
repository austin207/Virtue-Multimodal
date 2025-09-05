"""
Multimodal distillation losses for vision-text alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

class MultimodalDistillationLoss(nn.Module):
    """
    Multimodal knowledge distillation loss combining vision and text
    """
    
    def __init__(
        self,
        alpha_vision: float = 0.2,
        alpha_alignment: float = 0.1,
        temperature: float = 4.0
    ):
        super().__init__()
        self.alpha_vision = alpha_vision
        self.alpha_alignment = alpha_alignment 
        self.temperature = temperature
        
        # Vision-text alignment loss
        self.alignment_loss = VisionTextAlignmentLoss()
        
        # Vision feature distillation
        self.vision_distillation = VisionFeatureDistillationLoss()
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            batch: Input batch containing images and text
        
        Returns:
            Multimodal distillation loss
        """
        
        total_loss = 0.0
        
        # Vision feature distillation (if both have vision features)
        if ('vision_features' in student_outputs and 
            'vision_features' in teacher_outputs):
            vision_loss = self.vision_distillation(
                student_outputs['vision_features'],
                teacher_outputs['vision_features']
            )
            total_loss += self.alpha_vision * vision_loss
        
        # Vision-text alignment loss
        if 'images' in batch and batch['images'] is not None:
            alignment_loss = self.alignment_loss(
                student_outputs, teacher_outputs, batch
            )
            total_loss += self.alpha_alignment * alignment_loss
        
        return total_loss

class VisionTextAlignmentLoss(nn.Module):
    """
    Loss to align vision and text representations
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Contrastive loss for vision-text alignment
        """
        
        # Extract vision and text features
        if 'vision_features' not in student_outputs:
            return torch.tensor(0.0, device=next(iter(student_outputs.values())).device)
        
        student_vision = student_outputs['vision_features']  # [B, seq_len, dim]
        student_text = student_outputs['hidden_states']      # [B, seq_len, dim]
        
        teacher_vision = teacher_outputs.get('vision_features', None)
        teacher_text = teacher_outputs.get('hidden_states', None)
        
        # Pool features (mean pooling)
        student_vision_pooled = student_vision.mean(dim=1)  # [B, dim]
        student_text_pooled = student_text.mean(dim=1)      # [B, dim]
        
        if teacher_vision is not None and teacher_text is not None:
            teacher_vision_pooled = teacher_vision.mean(dim=1)
            teacher_text_pooled = teacher_text.mean(dim=1)
            
            # L2 normalize
            student_vision_pooled = F.normalize(student_vision_pooled, dim=-1)
            student_text_pooled = F.normalize(student_text_pooled, dim=-1)
            teacher_vision_pooled = F.normalize(teacher_vision_pooled, dim=-1)
            teacher_text_pooled = F.normalize(teacher_text_pooled, dim=-1)
            
            # Alignment loss: encourage student to match teacher alignment
            teacher_alignment = torch.cosine_similarity(teacher_vision_pooled, teacher_text_pooled, dim=-1)
            student_alignment = torch.cosine_similarity(student_vision_pooled, student_text_pooled, dim=-1)
            
            alignment_loss = F.mse_loss(student_alignment, teacher_alignment.detach())
        else:
            # Simple contrastive loss for student only
            batch_size = student_vision_pooled.size(0)
            
            # Normalize features
            student_vision_pooled = F.normalize(student_vision_pooled, dim=-1)
            student_text_pooled = F.normalize(student_text_pooled, dim=-1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(student_vision_pooled, student_text_pooled.T)
            
            # Positive pairs (diagonal)
            positive_scores = torch.diag(similarity_matrix)
            
            # Negative pairs (off-diagonal)
            negative_scores = similarity_matrix.masked_fill(
                torch.eye(batch_size, device=similarity_matrix.device).bool(), 
                float('-inf')
            )
            
            # Contrastive loss
            positive_loss = -positive_scores.mean()
            negative_loss = torch.logsumexp(negative_scores, dim=1).mean()
            
            alignment_loss = positive_loss + negative_loss
        
        return alignment_loss

class VisionFeatureDistillationLoss(nn.Module):
    """
    Distillation loss for vision encoder features
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_features: [batch_size, num_patches, dim]
            teacher_features: [batch_size, num_patches, dim]
        
        Returns:
            Vision feature distillation loss
        """
        
        # Handle dimension mismatch
        if student_features.size(-1) != teacher_features.size(-1):
            # Simple average pooling to match dimensions
            if student_features.size(-1) < teacher_features.size(-1):
                teacher_features = F.adaptive_avg_pool1d(
                    teacher_features.transpose(-2, -1), 
                    student_features.size(-1)
                ).transpose(-2, -1)
            else:
                student_features = F.adaptive_avg_pool1d(
                    student_features.transpose(-2, -1), 
                    teacher_features.size(-1)
                ).transpose(-2, -1)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_features, teacher_features.detach())
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            student_norm = F.normalize(student_features, dim=-1)
            teacher_norm = F.normalize(teacher_features.detach(), dim=-1)
            cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
            loss = (1 - cosine_sim).mean()
        elif self.loss_type == 'kl':
            # KL divergence on attention-like features
            student_probs = F.softmax(student_features, dim=-1)
            teacher_probs = F.softmax(teacher_features.detach(), dim=-1)
            loss = F.kl_div(
                student_probs.log(), teacher_probs, 
                reduction='batchmean'
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss

class CrossModalAttentionLoss(nn.Module):
    """
    Loss for cross-modal attention alignment
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, V, D]
        text_features: torch.Tensor,   # [B, T, D]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-modal attention loss using InfoNCE
        """
        
        batch_size, vision_len, dim = vision_features.shape
        text_len = text_features.shape[1]
        
        # L2 normalize
        vision_norm = F.normalize(vision_features, dim=-1)  # [B, V, D]
        text_norm = F.normalize(text_features, dim=-1)      # [B, T, D]
        
        # Compute cross-modal similarities
        similarities = torch.matmul(
            vision_norm.view(-1, dim),                      # [B*V, D]
            text_norm.view(-1, dim).T                       # [D, B*T]
        ) / self.temperature                                # [B*V, B*T]
        
        # Create labels for positive pairs
        # Each vision patch should attend to text from same sample
        vision_indices = torch.arange(batch_size).repeat_interleave(vision_len)
        text_indices = torch.arange(batch_size).repeat_interleave(text_len)
        
        labels = (vision_indices.unsqueeze(1) == text_indices.unsqueeze(0)).float()
        labels = labels.to(similarities.device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(-1).unsqueeze(0).repeat(similarities.size(0), 1)
            similarities = similarities.masked_fill(~mask.bool(), float('-inf'))
        
        # InfoNCE loss
        log_probs = F.log_softmax(similarities, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()
        
        return loss

class ModalityGapLoss(nn.Module):
    """
    Loss to minimize modality gap between vision and text representations
    """
    
    def __init__(self, gap_penalty: float = 1.0):
        super().__init__()
        self.gap_penalty = gap_penalty
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Minimize the gap between vision and text feature distributions
        """
        
        # Pool features
        vision_pooled = vision_features.mean(dim=1)  # [B, D]
        text_pooled = text_features.mean(dim=1)      # [B, D]
        
        # Compute means and standard deviations
        vision_mean = vision_pooled.mean(dim=0)
        text_mean = text_pooled.mean(dim=0)
        
        vision_std = vision_pooled.std(dim=0)
        text_std = text_pooled.std(dim=0)
        
        # Mean alignment loss
        mean_loss = F.mse_loss(vision_mean, text_mean)
        
        # Standard deviation alignment loss  
        std_loss = F.mse_loss(vision_std, text_std)
        
        return self.gap_penalty * (mean_loss + std_loss)
