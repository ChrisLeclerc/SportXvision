"""Complete action spotting model"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from .backbone import UNetBackbone
from .heads import ConfidenceHead, DeltaHead
from .blocks import Conv1dBlock

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.class_info import NUM_CLASSES


class ActionSpottingModel(nn.Module):
    """Complete model for action spotting"""

    def __init__(self, config):
        super().__init__()

        in_features = config.data.feature_dim
        reduced_features = 512

        if in_features != reduced_features:
            self.feature_reduction = nn.Sequential(
                nn.Conv1d(in_features, 4 * config.model.width, 1),
                nn.ReLU(),
                nn.Conv1d(4 * config.model.width, config.model.width, 1),
                nn.ReLU()
            )
            backbone_in = config.model.width
        else:
            self.feature_reduction = nn.Identity()
            backbone_in = reduced_features

        self.backbone = UNetBackbone(
            in_channels=backbone_in,
            width=config.model.width,
            start_layer=config.model.unet_start_layer,
            end_layer=config.model.unet_end_layer,
            max_filters=config.model.unet_max_filters,
            blocks_per_stage=config.model.unet_blocks_per_stage,
            downsample=config.model.unet_downsample,
            combiner=config.model.unet_combiner,
            upsampler=config.model.unet_upsampler,
            use_bottleneck=config.model.use_bottleneck,
            use_batch_norm=config.model.batch_norm,
            dropout=config.model.dropout
        )

        backbone_out = config.model.width * 16

        if config.loss.confidence_weight > 0:
            self.confidence_head = ConfidenceHead(
                backbone_out, NUM_CLASSES,
                width=config.model.width,
                num_layers=config.model.head_layers,
                use_batch_norm=config.model.batch_norm,
                dropout=config.model.dropout
            )
        else:
            self.confidence_head = None

        if config.loss.delta_weight > 0:
            self.delta_head = DeltaHead(
                backbone_out, NUM_CLASSES,
                width=config.model.width,
                num_layers=config.model.head_layers,
                use_batch_norm=config.model.batch_norm,
                dropout=config.model.dropout
            )
        else:
            self.delta_head = None

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.feature_reduction(x)

        x = self.backbone(x)

        outputs = {}

        if self.confidence_head is not None:
            confidence = self.confidence_head(x)
            outputs['confidence'] = confidence.transpose(1, 2)

        if self.delta_head is not None:
            delta = self.delta_head(x)
            outputs['delta'] = delta.transpose(1, 2)

        return outputs
