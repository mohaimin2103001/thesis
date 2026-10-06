"""
Deep Learning Architectures for Spaceborne SAR Ship Wake Analysis
Implements ResNet-18, ResNet-34, ConvNeXt-Tiny, and Dual Ensemble with Test-Time Augmentation (TTA).
Includes Monte Carlo Dropout heads for Epistemic Uncertainty Quantification and Multi-Task Kinematics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SARWakeBackbone(nn.Module):
    """
    Modular SAR Wake Classification Backbone with MC-Dropout and optional Multi-Task Kinematics Head.
    """
    def __init__(self, architecture="resnet18", num_classes=2, dropout_p=0.3, multi_task=False):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.multi_task = multi_task

        if architecture == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.feature_extractor = base
        elif architecture == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.feature_extractor = base
        elif architecture == "convnext_tiny":
            base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            feat_dim = base.classifier[2].in_features
            base.classifier[2] = nn.Identity()
            self.backbone = base
            self.feature_extractor = base
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Epistemic Uncertainty MC-Dropout Head
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(feat_dim, self.num_classes)

        # Auxiliary Kinematics Regressor Head (Rate of Turn & Speed Over Ground)
        if self.multi_task:
            self.kinematics_head = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(64, 2)
            )

    def forward(self, x):
        feat = self.backbone(x)
        feat_drop = self.dropout(feat)
        logits = self.classifier(feat_drop)

        if self.multi_task:
            kinematics_pred = self.kinematics_head(feat)
            return logits, kinematics_pred
        return logits

    def forward_mc_dropout(self, x, num_passes=30):
        """
        Executes N stochastic forward passes with active dropout during test time.
        Computes Mean Prediction Probability, Predictive Entropy, and Epistemic Variance.
        """
        self.eval()
        self.dropout.train()  # Force dropout to remain active

        softmax_probs = []
        with torch.no_grad():
            for _ in range(num_passes):
                logits = self.forward(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                prob = F.softmax(logits, dim=1)
                softmax_probs.append(prob.unsqueeze(0))

        # Shape: (num_passes, batch_size, num_classes)
        stacked = torch.cat(softmax_probs, dim=0)
        mean_prob = torch.mean(stacked, dim=0)
        variance = torch.var(stacked, dim=0)

        # Predictive Entropy: H[x] = - sum(p * log(p + eps))
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8), dim=1)

        return {
            "mean_prob": mean_prob,
            "variance": variance,
            "entropy": entropy,
            "prediction": torch.argmax(mean_prob, dim=1)
        }


class DualEnsembleWakeClassifier(nn.Module):
    """
    Dual-Architecture Ensemble combining ResNet-18 (Specificity on straight tracks)
    and ResNet-34 (Sensitivity on subtle turning curvatures) with Test-Time Augmentation (TTA).
    """
    def __init__(self, model_r18, model_r34, weight_r18=0.45, weight_r34=0.55):
        super().__init__()
        self.model_r18 = model_r18
        self.model_r34 = model_r34
        self.weight_r18 = weight_r18
        self.weight_r34 = weight_r34

    def forward_tta(self, x):
        """
        4-fold Test-Time Augmentation:
        1. Original
        2. Horizontal Flip
        3. Vertical Flip
        4. Diagonal Transpose (90-deg rotation)
        """
        views = [
            x,
            torch.flip(x, dims=[-1]),
            torch.flip(x, dims=[-2]),
            torch.rot90(x, k=1, dims=[-2, -1])
        ]

        probs_r18 = []
        probs_r34 = []

        for v in views:
            logits_18 = self.model_r18(v)
            logits_34 = self.model_r34(v)
            if isinstance(logits_18, tuple):
                logits_18 = logits_18[0]
            if isinstance(logits_34, tuple):
                logits_34 = logits_34[0]

            probs_r18.append(F.softmax(logits_18, dim=1))
            probs_r34.append(F.softmax(logits_34, dim=1))

        avg_p18 = torch.stack(probs_r18).mean(dim=0)
        avg_p34 = torch.stack(probs_r34).mean(dim=0)

        blended = self.weight_r18 * avg_p18 + self.weight_r34 * avg_p34
        return blended
