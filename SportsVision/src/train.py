"""Training script for action spotting"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, NUM_CLASSES, LABEL_NAMES
from models import ActionSpottingModel
from data import ActionSpottingDataset, collate_fn
from losses import ConfidenceLoss, DeltaLoss
from evaluation import evaluate_spotting, apply_nms
from utils import get_optimizer, get_scheduler, init_wandb, log_metrics, ProgressLogger


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion_conf, criterion_delta, optimizer, scheduler,
                scaler, config, epoch, logger):
    """Train for one epoch"""
    model.train()
    logger.reset()

    pbar = tqdm(dataloader, desc=f'[Epoch {epoch}/{config.training.epochs}] Training')

    for batch_idx, batch in enumerate(pbar):
        features = batch['features'].to(config.device)
        conf_targets = batch['confidence_targets'].to(config.device)
        delta_targets = batch['delta_targets'].to(config.device)
        conf_weights = batch['confidence_weights'].to(config.device)
        delta_weights = batch['delta_weights'].to(config.device)

        if config.training.mixup_alpha > 0 and conf_targets is not None:
            alpha = config.training.mixup_alpha
            lam = np.random.beta(alpha, alpha)
            batch_size = features.size(0)
            index = torch.randperm(batch_size, device=features.device)

            features = lam * features + (1 - lam) * features[index]
            conf_targets = lam * conf_targets + (1 - lam) * conf_targets[index]
            conf_weights = lam * conf_weights + (1 - lam) * conf_weights[index]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.training.use_amp):
            outputs = model(features)

            total_loss = 0
            losses = {}

            if 'confidence' in outputs and criterion_conf is not None:
                conf_loss = criterion_conf(outputs['confidence'], conf_targets, conf_weights)
                total_loss += config.loss.confidence_weight * conf_loss
                losses['conf_loss'] = conf_loss.item()

            if 'delta' in outputs and criterion_delta is not None:
                delta_loss = criterion_delta(outputs['delta'], delta_targets, delta_weights)
                total_loss += config.loss.delta_weight * delta_loss
                losses['delta_loss'] = delta_loss.item()

            losses['total_loss'] = total_loss.item()

        scaler.scale(total_loss).backward()

        if config.training.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.update(**losses, lr=scheduler.get_last_lr()[0])

        pbar.set_postfix({
            'loss': f"{losses['total_loss']:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return logger.get_averages()


@torch.no_grad()
def validate(model, dataloader, config, epoch, criterion_conf=None, criterion_delta=None):
    """Validate the model"""
    model.eval()

    print(f'[Epoch {epoch}/{config.training.epochs}] Running validation...')

    all_predictions = {}
    all_ground_truths = {}

    val_losses = {'total_loss': 0.0, 'conf_loss': 0.0, 'delta_loss': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc='Validation')

    for batch in pbar:
        features = batch['features'].to(config.device)
        conf_targets = batch['confidence_targets'].to(config.device)
        delta_targets = batch['delta_targets'].to(config.device)
        conf_weights = batch['confidence_weights'].to(config.device)
        delta_weights = batch['delta_weights'].to(config.device)

        outputs = model(features)

        batch_loss = 0.0
        if 'confidence' in outputs and criterion_conf is not None:
            conf_loss = criterion_conf(outputs['confidence'], conf_targets, conf_weights)
            batch_loss += config.loss.confidence_weight * conf_loss
            val_losses['conf_loss'] += conf_loss.item()

        if 'delta' in outputs and criterion_delta is not None:
            delta_loss = criterion_delta(outputs['delta'], delta_targets, delta_weights)
            batch_loss += config.loss.delta_weight * delta_loss
            val_losses['delta_loss'] += delta_loss.item()

        val_losses['total_loss'] += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
        num_batches += 1

        if 'confidence' in outputs:
            conf_scores = outputs['confidence'].sigmoid().cpu().numpy()
        else:
            continue

        for i in range(len(batch['video_names'])):
            video_name = batch['video_names'][i]
            half = batch['halves'][i]
            start_frame = batch['start_frames'][i]

            video_key = (video_name, half)

            scores = conf_scores[i]
            num_frames = scores.shape[0]

            for frame_idx in range(num_frames):
                absolute_frame = start_frame + frame_idx

                for class_idx in range(NUM_CLASSES):
                    score = scores[frame_idx, class_idx]

                    if score > config.eval.confidence_threshold:
                        if video_key not in all_predictions:
                            all_predictions[video_key] = []

                        all_predictions[video_key].append({
                            'frame': absolute_frame,
                            'class': class_idx,
                            'score': score
                        })

    train_dataset = dataloader.dataset
    for video_name in train_dataset.video_names:
        for half in [1, 2]:
            labels = train_dataset._load_labels(video_name, half)

            if labels:
                video_key = (video_name, half)
                all_ground_truths[video_key] = labels

    for key in val_losses:
        val_losses[key] /= num_batches

    tolerances_frames = [int(t * config.data.frame_rate) for t in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]]

    results = evaluate_spotting(all_predictions, all_ground_truths, tolerances_frames, NUM_CLASSES)

    results.update(val_losses)

    print(f"\n[Validation Results]")
    print(f"  Val Loss: {val_losses['total_loss']:.4f}")
    if val_losses['conf_loss'] > 0:
        print(f"  Conf Loss: {val_losses['conf_loss']:.4f}")
    if val_losses['delta_loss'] > 0:
        print(f"  Delta Loss: {val_losses['delta_loss']:.4f}")
    print(f"  mAP (tight): {results.get('mAP_tight', 0):.4f}")
    print(f"  mAP (loose): {results.get('mAP_loose', 0):.4f}")

    for tolerance in [1, 2, 3, 5, 10, 20, 30, 60]:
        tolerance_frames = int(tolerance * config.data.frame_rate)
        if f'mAP@{tolerance_frames}' in results:
            print(f"  mAP@{tolerance}s: {results[f'mAP@{tolerance_frames}']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train action spotting model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    set_seed(config.seed)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("=" * 80)
    print("SportsVision - Action Spotting Training")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Weight decay: {config.training.weight_decay}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Mixed precision: {config.training.use_amp}")
    print("=" * 80)

    print("\n[1/6] Loading datasets...")
    train_dataset = ActionSpottingDataset(config, split='train')
    val_dataset = ActionSpottingDataset(config, split='validation')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        persistent_workers=config.data.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        persistent_workers=config.data.num_workers > 0
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\n[2/6] Building model...")
    model = ActionSpottingModel(config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[3/6] Setting up training...")
    criterion_conf = ConfidenceLoss(
        positive_weight=config.loss.positive_weight_confidence,
        focal_gamma=config.loss.focal_gamma
    ) if config.loss.confidence_weight > 0 else None

    criterion_delta = DeltaLoss(
        huber_delta=config.loss.huber_delta,
        positive_weight=config.loss.positive_weight_delta
    ) if config.loss.delta_weight > 0 else None

    optimizer = get_optimizer(model, config)

    total_steps = len(train_loader) * config.training.epochs
    scheduler = get_scheduler(optimizer, config, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=config.training.use_amp)

    start_epoch = 1
    best_map = 0.0

    if args.resume:
        print(f"\n[4/6] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']}, best mAP: {best_map:.4f}")
    else:
        print("\n[4/6] Starting training from scratch")

    print("\n[5/6] Initializing WandB...")
    os.environ['WANDB_API_KEY'] = 'f90bf2c850509dffd89516a841f2fc15587115e2'
    init_wandb(config, model)

    print("\n[6/6] Training started!")
    print("=" * 80)

    logger = ProgressLogger()

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion_conf, criterion_delta,
            optimizer, scheduler, scaler, config, epoch, logger
        )

        log_metrics(train_metrics, epoch, prefix='train')
        print(f"\n[Epoch {epoch}] Train - Loss: {train_metrics['total_loss']:.4f}, LR: {train_metrics['lr']:.2e}")

        if epoch % config.training.val_every == 0:
            val_results = validate(model, val_loader, config, epoch, criterion_conf, criterion_delta)
            log_metrics(val_results, epoch, prefix='val')

            current_map = val_results.get('mAP_loose', 0)
            if current_map > best_map:
                best_map = current_map
                checkpoint_path = f'checkpoints/best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_map': best_map,
                    'config': config.to_dict()
                }, checkpoint_path)
                print(f"  ✓ Saved best model (mAP: {best_map:.4f})")

        if epoch % config.training.save_every == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_map,
                'config': config.to_dict()
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint at epoch {epoch}")

        print("=" * 80)

    print("\nTraining completed!")
    print(f"Best validation mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
