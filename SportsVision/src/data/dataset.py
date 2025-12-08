"""Dataset for action spotting"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.class_info import LABEL_NAMES, NUM_CLASSES


class ActionSpottingDataset(Dataset):
    """Dataset for loading video features and labels"""

    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.frame_rate = config.data.frame_rate
        self.chunk_duration = config.data.chunk_duration
        self.chunk_stride = config.data.chunk_stride if split == 'train' else config.data.chunk_duration
        self.chunk_size = int(self.chunk_duration * self.frame_rate)

        splits_path = os.path.join(config.data.data_root, config.data.splits_file)
        self.splits_df = pd.read_csv(splits_path)
        self.video_names = self.splits_df[self.splits_df['split_key'] == split]['video_name'].tolist()

        self.feature_dir = os.path.join(config.data.data_root, 'features', config.data.feature_name)

        self.chunks = self._create_chunks()

        self.label_to_idx = {label: idx for idx, label in enumerate(LABEL_NAMES)}

        print(f"[Dataset] Split: {split}, Videos: {len(self.video_names)}, Chunks: {len(self.chunks)}")

    def _create_chunks(self):
        """Create list of chunks from videos"""
        chunks = []

        for video_name in self.video_names:
            half1_path = os.path.join(self.feature_dir, video_name, f"1_{self.config.data.feature_name}.npy")
            half2_path = os.path.join(self.feature_dir, video_name, f"2_{self.config.data.feature_name}.npy")

            for half_idx, feat_path in enumerate([half1_path, half2_path], 1):
                if not os.path.exists(feat_path):
                    continue

                features = np.load(feat_path, mmap_mode='r')
                num_frames = features.shape[0]

                chunk_stride_frames = int(self.chunk_stride * self.frame_rate)

                for start_frame in range(0, num_frames, chunk_stride_frames):
                    end_frame = min(start_frame + self.chunk_size, num_frames)

                    chunks.append({
                        'video_name': video_name,
                        'half': half_idx,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'feature_path': feat_path
                    })

                    if end_frame >= num_frames:
                        break

        return chunks

    def _load_labels(self, video_name, half):
        """Load labels for a video half"""
        label_path = os.path.join('/home/o_a38510/ML/Dataset', video_name, 'Labels-v2.json')

        if not os.path.exists(label_path):
            return []

        with open(label_path, 'r') as f:
            data = json.load(f)

        labels = []
        for annotation in data.get('annotations', []):
            game_time = annotation.get('gameTime', '')
            label = annotation.get('label', '')

            if label not in self.label_to_idx:
                continue

            parts = game_time.split(' - ')
            if len(parts) != 2:
                continue

            half_str, time_str = parts
            half_num = int(half_str)

            if half_num != half:
                continue

            time_parts = time_str.split(':')
            if len(time_parts) != 2:
                continue

            minutes, seconds = map(int, time_parts)
            time_in_seconds = minutes * 60 + seconds
            frame = int(time_in_seconds * self.frame_rate)

            labels.append({
                'frame': frame,
                'class': self.label_to_idx[label]
            })

        return labels

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]

        features = np.load(chunk_info['feature_path'], mmap_mode='r')
        start = chunk_info['start_frame']
        end = chunk_info['end_frame']

        chunk_features = features[start:end].copy()

        if chunk_features.shape[0] < self.chunk_size:
            pad_size = self.chunk_size - chunk_features.shape[0]
            chunk_features = np.pad(chunk_features, ((0, pad_size), (0, 0)), mode='constant')

        labels = self._load_labels(chunk_info['video_name'], chunk_info['half'])

        confidence_targets, delta_targets, confidence_weights, delta_weights = self._create_targets(
            labels, start, self.chunk_size, NUM_CLASSES
        )

        return {
            'features': torch.from_numpy(chunk_features).float(),
            'confidence_targets': confidence_targets,
            'delta_targets': delta_targets,
            'confidence_weights': confidence_weights,
            'delta_weights': delta_weights,
            'video_name': chunk_info['video_name'],
            'half': chunk_info['half'],
            'start_frame': start
        }

    def _create_targets(self, labels, chunk_start, chunk_size, num_classes):
        """Create targets for confidence and delta"""
        confidence_radius = self.config.loss.dense_detection_radius * self.frame_rate
        delta_radius = confidence_radius * self.config.loss.delta_radius_multiplier

        confidence_targets = np.zeros((chunk_size, num_classes), dtype=np.float32)
        delta_targets = np.zeros((chunk_size, num_classes), dtype=np.float32)
        confidence_weights = np.ones((chunk_size, num_classes), dtype=np.float32)
        delta_weights = np.zeros((chunk_size, num_classes), dtype=np.float32)

        for frame_idx in range(chunk_size):
            absolute_frame = chunk_start + frame_idx

            for class_idx in range(num_classes):
                class_labels = [l for l in labels if l['class'] == class_idx]

                if not class_labels:
                    continue

                distances = [abs(absolute_frame - l['frame']) for l in class_labels]
                min_distance = min(distances)
                nearest_label = class_labels[distances.index(min_distance)]

                if min_distance <= confidence_radius:
                    confidence_targets[frame_idx, class_idx] = np.exp(-(min_distance ** 2) / (2 * (confidence_radius / 3) ** 2))

                if min_distance <= delta_radius:
                    displacement = nearest_label['frame'] - absolute_frame
                    normalized_delta = displacement / delta_radius
                    normalized_delta = np.clip(normalized_delta, -1.0, 1.0)
                    delta_targets[frame_idx, class_idx] = normalized_delta
                    delta_weights[frame_idx, class_idx] = 1.0

        return (torch.from_numpy(confidence_targets),
                torch.from_numpy(delta_targets),
                torch.from_numpy(confidence_weights),
                torch.from_numpy(delta_weights))


def collate_fn(batch):
    """Custom collate function for batching"""
    return {
        'features': torch.stack([item['features'] for item in batch]),
        'confidence_targets': torch.stack([item['confidence_targets'] for item in batch]),
        'delta_targets': torch.stack([item['delta_targets'] for item in batch]),
        'confidence_weights': torch.stack([item['confidence_weights'] for item in batch]),
        'delta_weights': torch.stack([item['delta_weights'] for item in batch]),
        'video_names': [item['video_name'] for item in batch],
        'halves': [item['half'] for item in batch],
        'start_frames': [item['start_frame'] for item in batch]
    }
