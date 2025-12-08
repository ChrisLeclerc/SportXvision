"""Delta loss using Huber loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaLoss(nn.Module):
    """Huber loss for delta (displacement) prediction"""

    def __init__(self, huber_delta=0.5, positive_weight=1.0):
        super().__init__()
        self.huber_delta = huber_delta
        self.positive_weight = positive_weight

    def forward(self, predictions, targets, weights):
        """
        Args:
            predictions: (B, T, C) delta predictions in [-1, 1]
            targets: (B, T, C) delta targets in [-1, 1]
            weights: (B, T, C) weights (0 or 1)

        Returns:
            loss: scalar
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)

        huber = torch.where(
            abs_diff <= self.huber_delta,
            0.5 * diff ** 2,
            self.huber_delta * (abs_diff - 0.5 * self.huber_delta)
        )

        huber_normalized = huber / self.huber_delta

        loss = huber_normalized * weights

        num_valid = weights.sum() + 1e-7
        loss = loss.sum() / num_valid

        return loss
