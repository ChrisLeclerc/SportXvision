"""Testing script for action spotting - Evaluate on test set"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, NUM_CLASSES, LABEL_NAMES
from models import ActionSpottingModel
from data import ActionSpottingDataset, collate_fn
from evaluation import evaluate_spotting, apply_nms


def test(config, checkpoint_path, split='test', save_results=True, output_path=None):
    """
    Run inference on test set and compute metrics

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        split: Dataset split to evaluate ('test', 'validation', 'train')
        save_results: Whether to save results to JSON
        output_path: Path to save results (default: ../test_results_{split}.json)

    Returns:
        results: Dictionary containing mAP and per-class metrics
    """
    print("=" * 80)
    print("SportsVision - Action Spotting Evaluation")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Split: {split}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    print("\n[1/4] Loading model...")
    model = ActionSpottingModel(config).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print(f"\n[2/4] Loading {split} dataset...")
    test_dataset = ActionSpottingDataset(config, split=split)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        persistent_workers=config.data.num_workers > 0
    )

    print(f"Number of videos: {len(test_dataset.video_names)}")
    print(f"Number of batches: {len(test_loader)}")

    # Run inference
    print(f"\n[3/4] Running inference...")
    all_predictions = {}
    all_ground_truths = {}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')

        for batch in pbar:
            features = batch['features'].to(config.device)

            with torch.cuda.amp.autocast(enabled=config.training.use_amp):
                outputs = model(features)

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
                                'score': float(score)
                            })

    # Apply NMS
    print("\nApplying NMS...")
    nms_window = int(config.eval.nms_window * config.data.frame_rate)
    for video_key in all_predictions:
        all_predictions[video_key] = apply_nms(
            all_predictions[video_key],
            nms_window,
            nms_type=config.eval.nms_type
        )

    # Load ground truths
    print("Loading ground truths...")
    for video_name in test_dataset.video_names:
        for half in [1, 2]:
            labels = test_dataset._load_labels(video_name, half)
            if labels:
                video_key = (video_name, half)
                all_ground_truths[video_key] = labels

    # Compute metrics
    print(f"\n[4/4] Computing metrics...")
    tolerances_seconds = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50, 60]
    tolerances_frames = [int(t * config.data.frame_rate) for t in tolerances_seconds]

    results = evaluate_spotting(all_predictions, all_ground_truths, tolerances_frames, NUM_CLASSES)

    # Format results
    formatted_results = {
        'split': split,
        'checkpoint_epoch': epoch,
        'checkpoint_path': str(checkpoint_path),
        'mAP_tight': results.get('mAP_tight', 0),
        'mAP_loose': results.get('mAP_loose', 0),
        'per_tolerance': {},
        'per_class_ap': {}
    }

    # Per tolerance
    for t_sec, t_frames in zip(tolerances_seconds, tolerances_frames):
        key = f'mAP@{t_frames}'
        if key in results:
            formatted_results['per_tolerance'][f'{t_sec}s'] = results[key]

    # Per class (using loose tolerance average)
    loose_tolerance = int(10 * config.data.frame_rate)  # 10s as representative
    class_aps_key = f'class_aps@{loose_tolerance}'
    if class_aps_key in results:
        for i, ap in enumerate(results[class_aps_key]):
            formatted_results['per_class_ap'][LABEL_NAMES[i]] = ap

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nmAP (tight, 1-5s):  {formatted_results['mAP_tight']:.4f} ({formatted_results['mAP_tight']*100:.2f}%)")
    print(f"mAP (loose, 5-60s): {formatted_results['mAP_loose']:.4f} ({formatted_results['mAP_loose']*100:.2f}%)")

    print("\nPer-tolerance mAP:")
    print("-" * 40)
    for tol, val in formatted_results['per_tolerance'].items():
        print(f"  {tol:>6}: {val:.4f} ({val*100:.2f}%)")

    print("\nPer-class AP (loose):")
    print("-" * 40)
    for class_name, ap in formatted_results['per_class_ap'].items():
        print(f"  {class_name:>20}: {ap:.4f} ({ap*100:.2f}%)")

    # Save results
    if save_results:
        if output_path is None:
            output_path = Path(__file__).parent.parent / f'test_results_{split}.json'

        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("=" * 80)

    return formatted_results


def main():
    parser = argparse.ArgumentParser(description='Test action spotting model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: ../configs/confidence_baidu.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: ../checkpoints/best_model.pth)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cuda:7, cpu)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')

    args = parser.parse_args()

    # Load config
    config_path = args.config or str(Path(__file__).parent.parent / 'configs' / 'confidence_baidu.yaml')
    config = Config.from_yaml(config_path)

    # Override device if specified
    if args.device:
        config.device = args.device

    # Set checkpoint path
    checkpoint_path = args.checkpoint or str(Path(__file__).parent.parent / 'checkpoints' / 'best_model.pth')

    # Run test
    test(
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        save_results=not args.no_save,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
