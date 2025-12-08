"""Configuration management"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class DataConfig:
    data_root: str = "/home/o_a38510/ML/SportsVision/data"
    feature_name: str = "baidu_2.0"
    splits_file: str = "splits_baidu_2.0.csv"
    frame_rate: float = 2.0
    feature_dim: int = 8576
    chunk_duration: float = 112.0
    chunk_stride: float = 56.0
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    width: int = 16
    unet_start_layer: int = 0
    unet_end_layer: int = 6
    unet_max_filters: int = 2048
    unet_blocks_per_stage: List[int] = field(default_factory=lambda: [2] * 10)
    unet_downsample: str = "pooling_max"
    unet_combiner: str = "addition"
    unet_upsampler: str = "transpose"
    use_bottleneck: bool = True
    use_resnet_v2: bool = True
    head_layers: int = 1
    dropout: float = 0.0
    batch_norm: bool = False


@dataclass
class LossConfig:
    dense_detection_radius: float = 3.0
    delta_radius_multiplier: float = 2.0
    positive_weight_confidence: float = 0.03
    positive_weight_delta: float = 1.0
    focal_gamma: float = 0.0
    huber_delta: float = 0.5
    confidence_weight: float = 1.0
    delta_weight: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 2e-4
    lr_scheduler: str = "linear"
    warmup_epochs: int = 0
    gradient_clip: float = 0.0
    mixup_alpha: float = 2.0
    use_amp: bool = True
    save_every: int = 10
    val_every: int = 10


@dataclass
class EvalConfig:
    nms_window: float = 20.0
    nms_type: str = "suppress"
    confidence_threshold: float = 0.5
    chunk_border_discard: float = 20.0


@dataclass
class WandbConfig:
    project: str = "sportsXvision2"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()
        if data:
            for section, values in data.items():
                if hasattr(config, section):
                    if isinstance(values, dict):
                        section_config = getattr(config, section)
                        for key, value in values.items():
                            if hasattr(section_config, key):
                                setattr(section_config, key, value)
                    else:
                        setattr(config, section, values)
        return config

    def to_dict(self):
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'loss': self.loss.__dict__,
            'training': self.training.__dict__,
            'eval': self.eval.__dict__,
            'wandb': self.wandb.__dict__,
            'seed': self.seed,
            'device': self.device
        }
