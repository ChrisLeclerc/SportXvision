"""Non-maximum suppression for action spotting"""

import numpy as np
import torch


def nms_suppress(scores, window_size):
    """NMS with binary suppression"""
    num_frames, num_classes = scores.shape
    output = scores.copy()

    for class_idx in range(num_classes):
        class_scores = scores[:, class_idx].copy()

        while True:
            max_idx = np.argmax(class_scores)
            max_score = class_scores[max_idx]

            if max_score <= 0:
                break

            start = max(0, max_idx - window_size)
            end = min(num_frames, max_idx + window_size + 1)

            for i in range(start, end):
                if i != max_idx:
                    class_scores[i] = -1
                    output[i, class_idx] = -1

            class_scores[max_idx] = -1

    return output


def nms_linear(scores, window_size):
    """NMS with linear decay"""
    num_frames, num_classes = scores.shape
    output = scores.copy()

    for class_idx in range(num_classes):
        class_scores = scores[:, class_idx].copy()

        while True:
            max_idx = np.argmax(class_scores)
            max_score = class_scores[max_idx]

            if max_score <= 0:
                break

            start = max(0, max_idx - window_size)
            end = min(num_frames, max_idx + window_size + 1)

            for i in range(start, end):
                if i != max_idx:
                    distance = abs(i - max_idx)
                    decay = max(0, 1 - distance / window_size)
                    output[i, class_idx] *= decay

            class_scores[max_idx] = -1

    return output


def nms_gaussian(scores, window_size):
    """NMS with Gaussian decay"""
    num_frames, num_classes = scores.shape
    output = scores.copy()

    sigma = window_size / 3.0

    for class_idx in range(num_classes):
        class_scores = scores[:, class_idx].copy()

        while True:
            max_idx = np.argmax(class_scores)
            max_score = class_scores[max_idx]

            if max_score <= 0:
                break

            start = max(0, max_idx - window_size)
            end = min(num_frames, max_idx + window_size + 1)

            for i in range(start, end):
                if i != max_idx:
                    distance = abs(i - max_idx)
                    decay = np.exp(-(distance ** 2) / (2 * sigma ** 2))
                    output[i, class_idx] *= decay

            class_scores[max_idx] = -1

    return output


def apply_nms(scores, window_size, nms_type='suppress'):
    """Apply NMS to detection scores"""
    if nms_type == 'suppress':
        return nms_suppress(scores, window_size)
    elif nms_type == 'linear':
        return nms_linear(scores, window_size)
    elif nms_type == 'gaussian':
        return nms_gaussian(scores, window_size)
    else:
        raise ValueError(f"Unknown NMS type: {nms_type}")
