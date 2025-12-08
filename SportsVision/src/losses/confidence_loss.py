"""Confidence loss with focal loss support"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceLoss(nn.Module):
    """Binary cross-entropy loss with optional focal loss for confidence prediction"""

    def __init__(self, positive_weight=0.03, focal_gamma=0.0):
        super().__init__()
        self.positive_weight = positive_weight
        self.focal_gamma = focal_gamma

    def forward(self, predictions, targets, weights):
        """
        Args:
            predictions: (B, T, C) logits
            targets: (B, T, C) targets in [0, 1]
            weights: (B, T, C) weights

        Returns:
            loss: scalar
        """
        predictions = predictions.sigmoid()

        bce = -(targets * torch.log(predictions + 1e-7) +
                (1 - targets) * torch.log(1 - predictions + 1e-7))

        if self.focal_gamma > 0:
            pt = torch.where(targets > 0.5, predictions, 1 - predictions)
            focal_weight = (1 - pt) ** self.focal_gamma
            bce = bce * focal_weight

        positive_mask = (targets > 0.5).float()
        num_positives = positive_mask.sum() + 1e-7
        num_negatives = (1 - positive_mask).sum() + 1e-7

        positive_loss_weight = self.positive_weight / num_positives
        negative_loss_weight = (1 - self.positive_weight) / num_negatives

        sample_weights = positive_mask * positive_loss_weight + (1 - positive_mask) * negative_loss_weight

        loss = bce * weights * sample_weights
        loss = loss.sum()

        return loss
