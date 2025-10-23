from typing import List, Tuple

import torch
import torch.nn.functional as F

from config import CFG


def kd_loss(student_logits: torch.Tensor, teacher_soft_targets: torch.Tensor, T: float) -> torch.Tensor:
    """KL divergence between student and teacher soft targets (temperature-scaled)."""
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_soft_targets / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
    return loss


def importance_weighted_kd_loss(student_logits: torch.Tensor, teacher_soft_targets: torch.Tensor, 
                               importance_weights: torch.Tensor, T: float) -> torch.Tensor:
    """Importance-weighted KL divergence loss for multi-teacher knowledge distillation."""
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_soft_targets / T, dim=1)
    
    # Compute per-sample KL divergence
    kl_per_sample = F.kl_div(p_s, p_t, reduction="none").sum(dim=1) * (T * T)
    
    # Apply importance weights and take mean
    weighted_loss = (kl_per_sample * importance_weights).mean()
    return weighted_loss


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return (-probs * torch.log(probs + 1e-8)).sum(dim=1)


def confidence_weights(teacher_logits_list: List[torch.Tensor]) -> torch.Tensor:
    """Compute per-teacher weights based on average prediction confidence (low entropy => high weight)."""
    device = teacher_logits_list[0].device
    entropies = []
    for logits in teacher_logits_list:
        probs = F.softmax(logits, dim=1)
        entropies.append(entropy(probs).mean())
    entropies = torch.stack(entropies).to(device)
    # Convert entropy to confidence and normalize
    conf = 1.0 / (entropies + 1e-6)
    weights = conf / conf.sum()
    return weights


def aggregate_teacher_logits(teacher_logits_list: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Weighted average of teacher logits across teachers."""
    stacked = torch.stack(teacher_logits_list)  # [T, B, C]
    w = weights.view(-1, 1, 1)
    return (stacked * w).sum(dim=0)


def feature_alignment_loss(student_feats: List[torch.Tensor], teacher_feats_list: List[List[torch.Tensor]],
                           weights: torch.Tensor) -> torch.Tensor:
    """
    Align student features with weighted teacher features.
    """
    s = student_feats[-1] if isinstance(student_feats, list) else student_feats
    loss = 0.0
    for t_idx, t_feats in enumerate(teacher_feats_list):
        t = t_feats[-1] if isinstance(t_feats, list) else t_feats
        
        # Handle spatial dimension mismatch by adaptive pooling
        if s.shape[2:] != t.shape[2:]:
            # Use adaptive pooling to match spatial dimensions
            t = F.adaptive_avg_pool2d(t, s.shape[2:])
        
        # Handle channel dimension mismatch
        if s.size(1) != t.size(1):
            # Project teacher to student channels using 1x1 conv or averaging
            if t.size(1) > s.size(1):
                # Reduce teacher channels by averaging groups
                groups = t.size(1) // s.size(1)
                t = t.view(t.size(0), s.size(1), groups, *t.shape[2:]).mean(dim=2)
            else:
                # Expand teacher channels by repeating
                repeat_factor = s.size(1) // t.size(1)
                t = t.repeat(1, repeat_factor, 1, 1)
                if s.size(1) % t.size(1) != 0:
                    # Handle remainder by padding
                    remaining = s.size(1) - t.size(1)
                    t = torch.cat([t, t[:, :remaining]], dim=1)
        
        loss = loss + weights[t_idx] * F.mse_loss(s, t)
    return loss


def total_loss(student_logits: torch.Tensor,
               aggregated_teacher_logits: torch.Tensor,
               ce_logits: torch.Tensor,
               labels: torch.Tensor,
               feat_loss: torch.Tensor,
               T: float,
               importance_weights: torch.Tensor = None) -> torch.Tensor:
    """Total loss with optional importance weighting for knowledge distillation."""
    if importance_weights is not None:
        l_kd = importance_weighted_kd_loss(student_logits, aggregated_teacher_logits, importance_weights, T)
    else:
        l_kd = kd_loss(student_logits, aggregated_teacher_logits, T)
    
    l_ce = F.cross_entropy(ce_logits, labels)
    return CFG.lambda_kd * l_kd + CFG.lambda_feat * feat_loss + CFG.lambda_ce * l_ce