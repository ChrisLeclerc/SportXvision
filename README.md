# SportsVision - Action Spotting in Soccer Videos

A PyTorch implementation for temporal action spotting in soccer broadcast videos using the SoccerNet dataset with Baidu 2.0 features.

## Architecture

SportsVision uses a U-Net backbone with ResNet V2 blocks for temporal feature processing:
- **Input**: Pre-extracted Baidu 2.0 features (8576 dimensions @ 2 FPS)
- **Backbone**: U-Net encoder-decoder with skip connections
- **Output**: Per-frame confidence scores for 17 action classes
- **Loss**: Binary cross-entropy with class balancing (positive_weight=0.03)

## Results

| Metric | SportsVision | SPIVAK (Baseline) |
|--------|--------------|-------------------|
| mAP Loose (5-60s) | **82.52%** | 76.73% |
| mAP Tight (1-5s) | **61.98%** | 46.15% |

## Installation

### 1. Create Conda Environment

```bash
# Create new environment
conda create -n sportsvision python=3.10 -y
conda activate sportsvision

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

### 2. Alternative: Use Existing Environment

```bash
conda activate sportsvision
```

## Data Download

### 1. Install SoccerNet Package

```bash
pip install SoccerNet
```

### 2. Download Labels and Videos

```python
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/path/to/SoccerNet")

# Download labels (required)
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])

# Download videos (optional, for demo app)
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"])
```

### 3. Download Pre-extracted Baidu 2.0 Features

The Baidu 2.0 features are pre-extracted features from the SoccerNet Challenge winners. They consist of 5 different models:

| Model | Dimensions | Description |
|-------|------------|-------------|
| TPN | 2048 | Temporal Pyramid Network |
| GTA | 2048 | Global Temporal Attention |
| VTN | 384 | Video Transformer Network |
| irCSN | 2048 | Channel-Separated Convolutional Networks |
| I3D-Slow | 2048 | Inflated 3D ConvNet (SlowFast) |
| **Total** | **8576** | Concatenated features |

```python
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/path/to/SoccerNet")

# Download Baidu features (1 FPS)
mySoccerNetDownloader.downloadGames(
    files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"],
    split=["train", "valid", "test"]
)
```

**Note:** The downloaded features are at 1 FPS. You need to resample them to 2 FPS for training.

## Feature Resampling (1 FPS → 2 FPS)

The Baidu features are provided at 1 FPS, but our model uses 2 FPS. Use the following script to resample:

### Resampling Script

Create `scripts/resample_features.py`:

```python
"""Resample Baidu features from 1 FPS to 2 FPS using linear interpolation"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm
import argparse


def resample_features(input_path, output_path, source_fps=1.0, target_fps=2.0):
    """
    Resample features from source_fps to target_fps using linear interpolation.

    Args:
        input_path: Path to input .npy file (T, D) at source_fps
        output_path: Path to save resampled .npy file
        source_fps: Original frame rate (default: 1.0)
        target_fps: Target frame rate (default: 2.0)
    """
    # Load features: shape (T_source, D) where D=8576
    features = np.load(input_path)
    T_source, D = features.shape

    # Calculate video duration and target length
    duration = T_source / source_fps  # in seconds
    T_target = int(duration * target_fps)

    # Create time axes
    source_times = np.arange(T_source) / source_fps  # [0, 1, 2, ...] seconds
    target_times = np.arange(T_target) / target_fps  # [0, 0.5, 1, 1.5, ...] seconds

    # Interpolate each feature dimension
    interpolator = interp1d(
        source_times,
        features,
        axis=0,
        kind='linear',
        fill_value='extrapolate'
    )
    resampled = interpolator(target_times)

    # Save resampled features
    np.save(output_path, resampled.astype(np.float32))

    return T_source, T_target


def resample_dataset(input_dir, output_dir, source_fps=1.0, target_fps=2.0):
    """
    Resample all features in a dataset directory.

    Expected structure:
        input_dir/
            league/season/match/
                1_baidu_soccer_embeddings.npy
                2_baidu_soccer_embeddings.npy
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all feature files
    feature_files = list(input_dir.rglob("*_baidu_soccer_embeddings.npy"))
    print(f"Found {len(feature_files)} feature files")

    for input_path in tqdm(feature_files, desc="Resampling"):
        # Create corresponding output path
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resample
        T_src, T_tgt = resample_features(
            input_path, output_path, source_fps, target_fps
        )

    print(f"Resampled features saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample Baidu features to 2 FPS")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original features (1 FPS)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save resampled features (2 FPS)")
    parser.add_argument("--source_fps", type=float, default=1.0,
                        help="Source frame rate (default: 1.0)")
    parser.add_argument("--target_fps", type=float, default=2.0,
                        help="Target frame rate (default: 2.0)")

    args = parser.parse_args()
    resample_dataset(args.input_dir, args.output_dir, args.source_fps, args.target_fps)
```

### Run Resampling

```bash
# Resample all features from 1 FPS to 2 FPS
python scripts/resample_features.py \
    --input_dir /path/to/SoccerNet \
    --output_dir /path/to/SportsVision/data/features/baidu_2.0 \
    --source_fps 1.0 \
    --target_fps 2.0
```

### How Resampling Works

The resampling uses **linear interpolation** via `scipy.interpolate.interp1d`:

```
Original (1 FPS):   [f0] -------- [f1] -------- [f2] -------- [f3]
                     0s           1s           2s           3s

Resampled (2 FPS):  [f0] -- [*] -- [f1] -- [*] -- [f2] -- [*] -- [f3]
                     0s    0.5s    1s    1.5s    2s    2.5s    3s

Where [*] = interpolated frame = 0.5 * f[t] + 0.5 * f[t+1]
```

This doubles the temporal resolution while maintaining smooth feature transitions.

## Project Structure

```
SportsVision/
├── configs/
│   └── confidence_baidu.yaml      # Training configuration
├── checkpoints/
│   └── best_model.pth             # Trained model weights
├── data/
│   ├── features/baidu_2.0/        # Pre-extracted features
│   └── splits_baidu_2.0.csv       # Train/val/test splits
├── src/
│   ├── config/                    # Configuration classes
│   ├── data/                      # Dataset and dataloader
│   ├── models/                    # Model architecture
│   ├── losses/                    # Loss functions
│   ├── evaluation/                # Metrics and NMS
│   ├── utils/                     # Utilities
│   ├── train.py                   # Training script
│   └── test.py                    # Testing script
├── streamlit_app/
│   └── app.py                     # Demo application
├── requirements.txt
└── README.md
```

## Usage

### Training

```bash
# Activate environment
conda activate sportsvision

# Train from scratch
cd /home/o_a38510/ML/SportsVision/src
python train.py --config ../configs/confidence_baidu.yaml

# Resume training from checkpoint
python train.py --config ../configs/confidence_baidu.yaml --resume ../checkpoints/best_model.pth
```

**Training Options:**
- `--config`: Path to YAML config file
- `--resume`: Path to checkpoint to resume from

### Testing

```bash
# Activate environment
conda activate sportsvision

# Evaluate on test set
cd /home/o_a38510/ML/SportsVision/src
python test.py --config ../configs/confidence_baidu.yaml --checkpoint ../checkpoints/best_model.pth --split test

# Evaluate on validation set
python test.py --split validation

# Use specific GPU
python test.py --device cuda:7

# Custom output path
python test.py --output ../results/my_test_results.json
```

**Testing Options:**
- `--config`: Path to YAML config file (default: ../configs/confidence_baidu.yaml)
- `--checkpoint`: Path to model checkpoint (default: ../checkpoints/best_model.pth)
- `--split`: Dataset split to evaluate: train, validation, test (default: test)
- `--output`: Path to save results JSON
- `--device`: Device to use (e.g., cuda:0, cuda:7, cpu)
- `--no-save`: Do not save results to file

### Demo Application

```bash
# Activate environment
conda activate sportsvision

# Run Streamlit app
cd /home/o_a38510/ML/SportsVision
streamlit run streamlit_app/app.py --server.port 8501
```

## Configuration

Key parameters in `configs/confidence_baidu.yaml`:

```yaml
data:
  feature_name: baidu_2.0       # Feature type
  frame_rate: 2.0               # FPS of features
  chunk_duration: 112.0         # Temporal chunk size (seconds)

model:
  width: 16                     # Base channel width
  unet_start_layer: 0           # U-Net start layer
  unet_end_layer: 6             # U-Net end layer
  use_resnet_v2: true           # Use ResNet V2 blocks

loss:
  dense_detection_radius: 3.0   # Gaussian label radius (seconds)
  positive_weight_confidence: 0.03  # Class balancing weight
  confidence_weight: 1.0        # Confidence loss weight

training:
  batch_size: 32
  epochs: 1000
  learning_rate: 0.0002
  weight_decay: 0.0002
  mixup_alpha: 2.0              # Mixup augmentation
```

## Quick Start Commands

```bash
# 1. Activate environment
conda activate sportsvision

# 2. Navigate to source directory
cd /home/o_a38510/ML/SportsVision/src

# 3. Train model
python train.py --config ../configs/confidence_baidu.yaml

# 4. Test model
python test.py --split test

# 5. Run demo
cd .. && streamlit run streamlit_app/app.py --server.port 8501
```

## Action Classes (17 total)

1. Penalty
2. Kick-off
3. Goal
4. Substitution
5. Offside
6. Shots on target
7. Shots off target
8. Clearance
9. Ball out of play
10. Throw-in
11. Foul
12. Indirect free-kick
13. Direct free-kick
14. Corner
15. Yellow card
16. Red card
17. Yellow->red card

## Citation

If you use this code, please cite the SoccerNet dataset:

```bibtex
@inproceedings{Deliege2021SoccerNetv2,
  title={SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  author={Deliege, Adrien and Cioppa, Anthony and others},
  booktitle={CVPR Workshops},
  year={2021}
}
```

## License

This project is for research purposes only.
