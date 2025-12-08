"""Prediction heads for confidence and delta"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv1dBlock


class ConfidenceHead(nn.Module):
    """Head for predicting action confidence scores"""

    def __init__(self, in_channels, num_classes, width=16, num_layers=1,
                 use_batch_norm=False, dropout=0.0):
        super().__init__()

        layers = []
        head_width = width * 16

        current_ch = in_channels
        for i in range(num_layers):
            kernel_size = 3 if i == 0 else 1
            layers.append(Conv1dBlock(current_ch, head_width, kernel_size=kernel_size,
                                     use_batch_norm=use_batch_norm, dropout=dropout))
            current_ch = head_width

        self.stack = nn.Sequential(*layers) if layers else nn.Identity()

        self.output_conv = nn.Conv1d(current_ch, num_classes, 1)

    def forward(self, x):
        x = self.stack(x)
        x = self.output_conv(x)
        return x


class DeltaHead(nn.Module):
    """Head for predicting temporal displacements"""

    def __init__(self, in_channels, num_classes, width=16, num_layers=1,
                 use_batch_norm=False, dropout=0.0):
        super().__init__()

        layers = []
        head_width = width * 16

        current_ch = in_channels
        for i in range(num_layers):
            kernel_size = 3 if i == 0 else 1
            layers.append(Conv1dBlock(current_ch, head_width, kernel_size=kernel_size,
                                     use_batch_norm=use_batch_norm, dropout=dropout))
            current_ch = head_width

        self.stack = nn.Sequential(*layers) if layers else nn.Identity()

        self.output_conv = nn.Conv1d(current_ch, num_classes, 1)

    def forward(self, x):
        x = self.stack(x)
        x = self.output_conv(x)
        x = torch.tanh(x)
        return x
