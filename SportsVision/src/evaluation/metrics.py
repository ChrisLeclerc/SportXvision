"""Evaluation metrics for action spotting"""

import numpy as np
from collections import defaultdict


def compute_ap(detections, ground_truths, tolerance_frames):
    """Compute Average Precision for a single class and tolerance"""
    if len(ground_truths) == 0:
        return 0.0 if len(detections) > 0 else 1.0

    if len(detections) == 0:
        return 0.0

    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    matched_gt = set()

    for det_idx, det in enumerate(detections):
        best_match = None
        best_distance = float('inf')

        for gt_idx, gt_frame in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue

            distance = abs(det['frame'] - gt_frame)
            if distance <= tolerance_frames and distance < best_distance:
                best_distance = distance
                best_match = gt_idx

        if best_match is not None:
            tp[det_idx] = 1
            matched_gt.add(best_match)
        else:
            fp[det_idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def evaluate_spotting(predictions, ground_truths, tolerances, num_classes):
    """
    Evaluate action spotting performance

    Args:
        predictions: dict mapping (video_name, half) -> list of detections
                    each detection is {'frame': int, 'class': int, 'score': float}
        ground_truths: dict mapping (video_name, half) -> list of labels
                      each label is {'frame': int, 'class': int}
        tolerances: list of tolerance values in frames
        num_classes: number of action classes

    Returns:
        results: dict with mAP for each tolerance and per-class results
    """
    results = {}

    for tolerance in tolerances:
        class_aps = []

        for class_idx in range(num_classes):
            all_detections = []
            all_ground_truths = []

            for video_key in ground_truths:
                gt_frames = [gt['frame'] for gt in ground_truths[video_key]
                           if gt['class'] == class_idx]
                all_ground_truths.extend(gt_frames)

                if video_key in predictions:
                    dets = [{'frame': det['frame'], 'score': det['score']}
                           for det in predictions[video_key]
                           if det['class'] == class_idx]
                    all_detections.extend(dets)

            ap = compute_ap(all_detections, all_ground_truths, tolerance)
            class_aps.append(ap)

        mean_ap = np.mean(class_aps)
        results[f'mAP@{tolerance}'] = mean_ap
        results[f'class_aps@{tolerance}'] = class_aps

    tight_tolerances = [t for t in tolerances if t <= 5]
    loose_tolerances = [t for t in tolerances if t >= 5]

    if tight_tolerances:
        tight_maps = [results[f'mAP@{t}'] for t in tight_tolerances]
        results['mAP_tight'] = np.mean(tight_maps)

    if loose_tolerances:
        loose_maps = [results[f'mAP@{t}'] for t in loose_tolerances]
        results['mAP_loose'] = np.mean(loose_maps)

    return results
