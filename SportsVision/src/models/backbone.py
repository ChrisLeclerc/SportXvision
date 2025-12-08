"""U-Net backbone for temporal feature extraction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResNetStack, Conv1dBlock


class UNetBackbone(nn.Module):
    """U-Net architecture for dense temporal prediction"""

    def __init__(self, in_channels, width=16, start_layer=0, end_layer=6,
                 max_filters=2048, blocks_per_stage=None, downsample='pooling_max',
                 combiner='addition', upsampler='transpose', use_bottleneck=True,
                 use_batch_norm=False, dropout=0.0):
        super().__init__()

        self.start_layer = start_layer
        self.end_layer = end_layer
        self.width = width
        self.downsample = downsample
        self.combiner = combiner
        self.upsampler = upsampler

        if blocks_per_stage is None:
            blocks_per_stage = [2] * 10

        base_channels = width * 16

        self.bottom_up_stages = nn.ModuleList()
        self.top_down_stages = nn.ModuleList()

        current_channels = in_channels

        for layer_idx in range(start_layer, end_layer):
            out_channels = min(base_channels * (2 ** layer_idx), max_filters)

            if layer_idx == start_layer:
                stride = 1
            else:
                stride = 2 if downsample == 'stride' else 1

            stage = ResNetStack(
                current_channels, out_channels,
                num_blocks=blocks_per_stage[layer_idx],
                stride=stride,
                use_bottleneck=use_bottleneck,
                use_batch_norm=use_batch_norm,
                dropout=dropout
            )
            self.bottom_up_stages.append(stage)

            if downsample == 'pooling_max' and layer_idx > start_layer:
                self.bottom_up_stages.append(nn.MaxPool1d(2))
            elif downsample == 'pooling_avg' and layer_idx > start_layer:
                self.bottom_up_stages.append(nn.AvgPool1d(2))

            current_channels = out_channels

        for layer_idx in range(end_layer - 1, start_layer - 1, -1):
            in_ch = min(base_channels * (2 ** layer_idx), max_filters)
            out_ch = min(base_channels * (2 ** layer_idx), max_filters) if layer_idx == start_layer \
                     else min(base_channels * (2 ** (layer_idx - 1)), max_filters)

            if layer_idx > start_layer:
                if upsampler == 'transpose':
                    upsample = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
                else:
                    upsample = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv1d(in_ch, out_ch, 3, padding=1)
                    )
            else:
                upsample = nn.Identity()

            if combiner == 'concatenation' and layer_idx > start_layer:
                combine_in_ch = out_ch * 2
            else:
                combine_in_ch = out_ch

            stage = ResNetStack(
                combine_in_ch, out_ch,
                num_blocks=blocks_per_stage[layer_idx],
                stride=1,
                use_bottleneck=use_bottleneck,
                use_batch_norm=use_batch_norm,
                dropout=dropout
            )

            self.top_down_stages.append(nn.ModuleDict({
                'upsample': upsample,
                'refine': stage
            }))

    def forward(self, x):
        skip_connections = []

        stage_idx = 0
        for layer_idx in range(self.start_layer, self.end_layer):
            x = self.bottom_up_stages[stage_idx](x)
            stage_idx += 1
            skip_connections.append(x)

            if layer_idx > self.start_layer and self.downsample in ['pooling_max', 'pooling_avg']:
                x = self.bottom_up_stages[stage_idx](x)
                stage_idx += 1

        for i, layer_idx in enumerate(range(self.end_layer - 1, self.start_layer - 1, -1)):
            stage = self.top_down_stages[i]

            if layer_idx > self.start_layer:
                x = stage['upsample'](x)

                skip = skip_connections[layer_idx - self.start_layer - 1]

                if x.shape[-1] != skip.shape[-1]:
                    if x.shape[-1] < skip.shape[-1]:
                        x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
                    else:
                        x = x[..., :skip.shape[-1]]

                if self.combiner == 'concatenation':
                    x = torch.cat([x, skip], dim=1)
                else:
                    x = x + skip

            x = stage['refine'](x)

        return x
