"""Optimizer and scheduler utilities"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, config):
    """Create optimizer with weight decay"""
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


def get_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler"""
    warmup_steps = int(config.training.warmup_epochs * total_steps / config.training.epochs)

    if config.training.lr_scheduler == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(0.0, 1.0 - progress)

        scheduler = LambdaLR(optimizer, lr_lambda)

    elif config.training.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, total_steps - warmup_steps)

    elif config.training.lr_scheduler == 'none':
        scheduler = LambdaLR(optimizer, lambda step: 1.0)

    else:
        raise ValueError(f"Unknown scheduler: {config.training.lr_scheduler}")

    return scheduler


def mixup_data(features, targets, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_features, mixed_targets, lam


import numpy as np
