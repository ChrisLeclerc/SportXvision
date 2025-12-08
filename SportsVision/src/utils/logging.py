"""Logging utilities"""

import wandb
import torch
from datetime import datetime


def init_wandb(config, model, run_name=None):
    """Initialize Weights & Biases"""
    if run_name is None:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        config=config.to_dict(),
        tags=config.wandb.tags,
        notes=config.wandb.notes
    )

    return wandb.run


def log_metrics(metrics, step, prefix=''):
    """Log metrics to wandb"""
    log_dict = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            log_dict[f"{prefix}/{key}" if prefix else key] = value
        elif isinstance(value, torch.Tensor):
            log_dict[f"{prefix}/{key}" if prefix else key] = value.item()

    wandb.log(log_dict, step=step, commit=True)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressLogger:
    """Logger for tracking training progress"""
    def __init__(self):
        self.metrics = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value)

    def get_averages(self):
        return {key: meter.avg for key, meter in self.metrics.items()}

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()
