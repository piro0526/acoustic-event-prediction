"""Focal Loss implementation for binary classification with class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Focal Loss is designed to address class imbalance by down-weighting
    the loss assigned to well-classified examples. This allows the model
    to focus on hard, misclassified examples.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples.
               alpha=1 gives more weight to positive class (default).
               Can also be a Tensor for per-class weights.
        gamma: Focusing parameter for modulating loss. Higher values
               increase focus on hard examples (default: 2.0).
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum' (default: 'mean').

    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Target: (N, *), same shape as input
        - Output: scalar if reduction='mean' or 'sum', (N, *) if reduction='none'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute focal loss.

        Args:
            inputs: Logits from model (before sigmoid)
            targets: Binary labels (0 or 1)

        Returns:
            Focal loss value
        """
        # Compute sigmoid probabilities
        p = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Compute p_t (probability of the true class)
        # For positive examples (targets=1): p_t = p
        # For negative examples (targets=0): p_t = 1-p
        p_t = p * targets + (1 - p) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            # Simple scalar alpha
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        elif isinstance(self.alpha, torch.Tensor):
            # Tensor alpha (e.g., pos_weight from class imbalance)
            alpha_t = self.alpha * targets + (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss that automatically adjusts alpha based on class distribution.

    This variant automatically computes alpha from the positive class weight,
    making it easier to use with imbalanced datasets.

    Args:
        gamma: Focusing parameter (default: 2.0)
        pos_weight: Weight for positive class (typically computed from dataset)
        reduction: Reduction method (default: 'mean')
    """

    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Compute alpha from pos_weight
        # alpha = pos_weight / (1 + pos_weight) to normalize to [0, 1]
        if pos_weight is not None:
            if isinstance(pos_weight, torch.Tensor):
                self.alpha = pos_weight / (1 + pos_weight)
            else:
                self.alpha = pos_weight / (1 + pos_weight)
        else:
            self.alpha = 0.5  # Balanced classes

    def forward(self, inputs, targets):
        """Compute adaptive focal loss."""
        # Compute sigmoid probabilities
        p = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Compute p_t and focal weight
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
