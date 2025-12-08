"""ResNet blocks and basic layers for the model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 use_batch_norm=False, dropout=0.0, activation=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=not use_batch_norm)
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation:
            x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResNetBlockV2(nn.Module):
    """ResNet V2 block with pre-activation and optional bottleneck"""

    def __init__(self, in_channels, out_channels, stride=1, use_bottleneck=True,
                 use_batch_norm=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            bottleneck_channels = max(out_channels // 4, 1)

            self.bn1 = nn.BatchNorm1d(in_channels) if use_batch_norm else nn.Identity()
            self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=not use_batch_norm)

            self.bn2 = nn.BatchNorm1d(bottleneck_channels) if use_batch_norm else nn.Identity()
            self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, 3,
                                  stride=stride, padding=1, bias=not use_batch_norm)

            self.bn3 = nn.BatchNorm1d(bottleneck_channels) if use_batch_norm else nn.Identity()
            self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, 1, bias=not use_batch_norm)

        else:
            self.bn1 = nn.BatchNorm1d(in_channels) if use_batch_norm else nn.Identity()
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride,
                                  padding=1, bias=not use_batch_norm)

            self.bn2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
            self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1,
                                  bias=not use_batch_norm)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = x

        if self.use_bottleneck:
            out = F.relu(self.bn1(x))
            out = self.conv1(out)

            out = F.relu(self.bn2(out))
            out = self.conv2(out)

            out = F.relu(self.bn3(out))
            out = self.conv3(out)
        else:
            out = F.relu(self.bn1(x))
            out = self.conv1(out)

            out = F.relu(self.bn2(out))
            out = self.conv2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        identity = self.skip(identity)
        out = out + identity

        return out


class ResNetStack(nn.Module):
    """Stack of ResNet blocks"""

    def __init__(self, in_channels, out_channels, num_blocks=2, stride=1,
                 use_bottleneck=True, use_batch_norm=False, dropout=0.0):
        super().__init__()

        blocks = []
        blocks.append(ResNetBlockV2(in_channels, out_channels, stride=stride,
                                    use_bottleneck=use_bottleneck,
                                    use_batch_norm=use_batch_norm, dropout=dropout))

        for _ in range(num_blocks - 1):
            blocks.append(ResNetBlockV2(out_channels, out_channels, stride=1,
                                       use_bottleneck=use_bottleneck,
                                       use_batch_norm=use_batch_norm, dropout=dropout))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
