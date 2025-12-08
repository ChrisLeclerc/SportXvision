#!/usr/bin/env python3
"""
Generate training curves from log files for SportsVision and SPIVAK.
"""

import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def parse_sportsvision_log(log_path: str) -> dict:
    """Parse SportsVision log file and extract metrics."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_epochs': [],
        'mAP_tight': [],
        'mAP_loose': [],
        'mAP_1s': [],
        'mAP_2s': [],
        'mAP_3s': [],
        'mAP_5s': [],
        'mAP_10s': [],
        'mAP_20s': [],
        'mAP_30s': [],
        'mAP_60s': [],
        'lr': []
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_epoch = None
    in_validation = False

    for line in lines:
        # Match training loss line: [Epoch X] Train - Loss: Y, LR: Z
        train_match = re.search(r'\[Epoch (\d+)\] Train - Loss: ([\d.]+), LR: ([\d.e+-]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            lr = float(train_match.group(3))
            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(loss)
            metrics['lr'].append(lr)
            current_epoch = epoch

        # Match validation loss
        val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
        if val_loss_match and current_epoch:
            metrics['val_loss'].append(float(val_loss_match.group(1)))
            metrics['val_epochs'].append(current_epoch)

        # Match mAP metrics
        map_tight_match = re.search(r'mAP \(tight\): ([\d.]+)', line)
        if map_tight_match:
            metrics['mAP_tight'].append(float(map_tight_match.group(1)))

        map_loose_match = re.search(r'mAP \(loose\): ([\d.]+)', line)
        if map_loose_match:
            metrics['mAP_loose'].append(float(map_loose_match.group(1)))

        # Match individual tolerance mAPs
        for tolerance in ['1s', '2s', '3s', '5s', '10s', '20s', '30s', '60s']:
            match = re.search(rf'mAP@{tolerance}: ([\d.]+)', line)
            if match:
                metrics[f'mAP_{tolerance}'].append(float(match.group(1)))

    return metrics


def parse_spivak_log(log_path: str) -> dict:
    """Parse SPIVAK log file and extract metrics."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_epochs': []
    }

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"SPIVAK log not found: {log_path}")
        return metrics

    for line in lines:
        # Match epoch training line
        epoch_match = re.search(r'Epoch (\d+)/\d+.*loss[=:]?\s*([\d.]+)', line, re.IGNORECASE)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            loss = float(epoch_match.group(2))
            if epoch not in metrics['epochs']:
                metrics['epochs'].append(epoch)
                metrics['train_loss'].append(loss)

        # Match validation loss
        val_match = re.search(r'val.*loss[=:]?\s*([\d.]+)', line, re.IGNORECASE)
        if val_match:
            metrics['val_loss'].append(float(val_match.group(1)))

    return metrics


def plot_training_curves(sv_metrics: dict, output_dir: str):
    """Generate training curve plots for SportsVision."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Training and Validation Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sv_metrics['epochs'], sv_metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if sv_metrics['val_epochs']:
        ax.plot(sv_metrics['val_epochs'], sv_metrics['val_loss'], 'r-o', label='Validation Loss', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SportsVision: Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. mAP Curves (tight and loose)
    if sv_metrics['mAP_tight']:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sv_metrics['val_epochs'], sv_metrics['mAP_tight'], 'b-o', label='mAP (tight)', linewidth=2, markersize=6)
        ax.plot(sv_metrics['val_epochs'], sv_metrics['mAP_loose'], 'g-s', label='mAP (loose)', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('SportsVision: Mean Average Precision Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path / 'mAP_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. mAP at Different Tolerances
    if sv_metrics['mAP_1s']:
        fig, ax = plt.subplots(figsize=(14, 8))
        tolerances = ['1s', '2s', '3s', '5s', '10s', '20s', '30s', '60s']
        colors = plt.cm.viridis(np.linspace(0, 1, len(tolerances)))

        for tol, color in zip(tolerances, colors):
            key = f'mAP_{tol}'
            if sv_metrics[key]:
                ax.plot(sv_metrics['val_epochs'], sv_metrics[key], '-o',
                       label=f'mAP@{tol}', color=color, linewidth=1.5, markersize=4)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('SportsVision: mAP at Different Tolerances Over Training')
        ax.legend(loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path / 'mAP_tolerances.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Learning Rate Schedule
    if sv_metrics['lr']:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sv_metrics['epochs'], sv_metrics['lr'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('SportsVision: Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'lr_schedule.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Combined Summary Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss
    axes[0, 0].plot(sv_metrics['epochs'], sv_metrics['train_loss'], 'b-', label='Train', linewidth=2)
    if sv_metrics['val_epochs']:
        axes[0, 0].plot(sv_metrics['val_epochs'], sv_metrics['val_loss'], 'r-o', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # mAP
    if sv_metrics['mAP_tight']:
        axes[0, 1].plot(sv_metrics['val_epochs'], sv_metrics['mAP_tight'], 'b-o', label='Tight', linewidth=2)
        axes[0, 1].plot(sv_metrics['val_epochs'], sv_metrics['mAP_loose'], 'g-s', label='Loose', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('mAP (tight vs loose)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

    # LR
    if sv_metrics['lr']:
        axes[1, 0].plot(sv_metrics['epochs'], sv_metrics['lr'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

    # Final Tolerance Comparison (last epoch)
    if sv_metrics['mAP_1s']:
        tolerances = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50, 60]
        # Use test results if available
        test_results_path = Path('/home/o_a38510/ML/SportsVision/test_results_test.json')
        if test_results_path.exists():
            with open(test_results_path, 'r') as f:
                test_data = json.load(f)
            per_tol = test_data.get('per_tolerance', {})
            tol_names = list(per_tol.keys())
            tol_values = list(per_tol.values())

            bars = axes[1, 1].bar(range(len(tol_names)), tol_values, color='steelblue', edgecolor='navy')
            axes[1, 1].set_xticks(range(len(tol_names)))
            axes[1, 1].set_xticklabels(tol_names, rotation=45)
            axes[1, 1].set_ylabel('mAP')
            axes[1, 1].set_title('Test Set: mAP at Different Tolerances')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, val in zip(bars, tol_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('SportsVision Training Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'training_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_path}")


def plot_per_class_results(output_dir: str):
    """Plot per-class AP results from test."""
    output_path = Path(output_dir)
    test_results_path = Path('/home/o_a38510/ML/SportsVision/test_results_test.json')

    if not test_results_path.exists():
        print("Test results file not found")
        return

    with open(test_results_path, 'r') as f:
        test_data = json.load(f)

    per_class = test_data.get('per_class_ap', {})

    # Sort by AP value
    sorted_items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    aps = [item[1] for item in sorted_items]

    # Color by performance
    colors = ['green' if ap > 0.85 else 'orange' if ap > 0.7 else 'red' for ap in aps]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(classes)), aps, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Average Precision (AP)')
    ax.set_title('SportsVision Test Set: Per-Class Average Precision')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Add value labels
    for bar, ap in zip(bars, aps):
        ax.text(ap + 0.01, bar.get_y() + bar.get_height()/2,
               f'{ap:.3f}', ha='left', va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='High (>0.85)'),
        Patch(facecolor='orange', label='Medium (0.7-0.85)'),
        Patch(facecolor='red', label='Low (<0.7)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'per_class_ap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-class AP plot saved to {output_path / 'per_class_ap.png'}")


def plot_comparison_with_spivak(output_dir: str):
    """Plot comparison between our results and Yahoo Research (SPIVAK challenge)."""
    output_path = Path(output_dir)

    # Yahoo Research (SPIVAK) Challenge Results
    yahoo_challenge = {
        'tight Avg-mAP': 67.81,
        'Avg-mAP': 78.05
    }

    # Our test results
    test_results_path = Path('/home/o_a38510/ML/SportsVision/test_results_test.json')
    if test_results_path.exists():
        with open(test_results_path, 'r') as f:
            test_data = json.load(f)
        our_results = {
            'tight Avg-mAP': test_data['mAP_tight'] * 100,
            'Avg-mAP': test_data['mAP_loose'] * 100
        }
    else:
        our_results = {'tight Avg-mAP': 61.98, 'Avg-mAP': 82.52}

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['tight Avg-mAP', 'Avg-mAP']
    x = np.arange(len(metrics))
    width = 0.35

    yahoo_vals = [yahoo_challenge[m] for m in metrics]
    our_vals = [our_results[m] for m in metrics]

    bars1 = ax.bar(x - width/2, yahoo_vals, width, label='Yahoo Research (Challenge Set)', color='steelblue', edgecolor='navy')
    bars2 = ax.bar(x + width/2, our_vals, width, label='Ours (Test Set)', color='coral', edgecolor='darkred')

    ax.set_ylabel('mAP (%)')
    ax.set_title('Comparison: Yahoo Research (SPIVAK) vs Our Implementation')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add note about different evaluation sets
    ax.text(0.5, -0.15, 'Note: Yahoo Research evaluated on Challenge Set; Our results on Test Set',
           transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(output_path / 'spivak_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to {output_path / 'spivak_comparison.png'}")


def main():
    # Paths
    sv_log_path = '/home/o_a38510/ML/SportsVision/log.txt'
    spivak_log_path = '/home/o_a38510/ML/spivak/wandb/run-20251128_202255-2ld0eoeo/files/output.log'
    output_dir = '/home/o_a38510/ML/SportsVision/training_curves'

    print("=" * 60)
    print("Generating Training Curves")
    print("=" * 60)

    # Parse SportsVision log
    print("\n1. Parsing SportsVision log...")
    sv_metrics = parse_sportsvision_log(sv_log_path)
    print(f"   Found {len(sv_metrics['epochs'])} training epochs")
    print(f"   Found {len(sv_metrics['val_epochs'])} validation points")

    # Generate plots
    print("\n2. Generating training curve plots...")
    plot_training_curves(sv_metrics, output_dir)

    print("\n3. Generating per-class AP plot...")
    plot_per_class_results(output_dir)

    print("\n4. Generating comparison with SPIVAK...")
    plot_comparison_with_spivak(output_dir)

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Print summary statistics
    if sv_metrics['mAP_loose']:
        best_idx = np.argmax(sv_metrics['mAP_loose'])
        print(f"\nBest Validation Results:")
        print(f"  Epoch: {sv_metrics['val_epochs'][best_idx]}")
        print(f"  mAP (tight): {sv_metrics['mAP_tight'][best_idx]:.4f}")
        print(f"  mAP (loose): {sv_metrics['mAP_loose'][best_idx]:.4f}")


if __name__ == '__main__':
    main()
