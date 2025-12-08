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

The Baidu 2.0 features are pre-extracted features from the SoccerNet Challenge winners.

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

Can be found in `SportsVision/scripts/resample_features.py`


### Run Resampling

```bash
# Resample all features from 1 FPS to 2 FPS
python scripts/resample_features.py \
    --input_dir /path/to/SoccerNet \
    --output_dir /path/to/SportsVision/data/features/baidu_2.0 \
    --source_fps 1.0 \
    --target_fps 2.0
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
cd SportXvision/SportsVision/src
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

