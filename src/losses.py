"""
Loss Functions for Imbalanced Radar Classification and Multi-Task Hydrodynamics
Implements Class-Weighted Focal Loss and Multi-Task Kinematics Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-Class and Binary Focal Loss with alpha class balancing and gamma focusing parameter.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes)
        # targets: (batch_size,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prevents nans
        focal_term = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiTaskKinematicsLoss(nn.Module):
    """
    Joint loss function combining classification Focal Loss with auxiliary Kinematics regression.
    L_total = L_focal + lambda_rot * MSE(ROT_hat, ROT) + lambda_sog * MSE(SOG_hat, SOG)
    """
    def __init__(self, alpha=None, gamma=2.0, lambda_rot=0.5, lambda_sog=0.25):
        super().__init__()
        self.classification_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.mse = nn.MSELoss()
        self.lambda_rot = lambda_rot
        self.lambda_sog = lambda_sog

    def forward(self, pred_logits, pred_kinematics, target_class, target_kinematics):
        l_cls = self.classification_loss(pred_logits, target_class)
        l_rot = self.mse(pred_kinematics[:, 0], target_kinematics[:, 0])
        l_sog = self.mse(pred_kinematics[:, 1], target_kinematics[:, 1])
        total_loss = l_cls + self.lambda_rot * l_rot + self.lambda_sog * l_sog
        return total_loss, l_cls, l_rot, l_sog
