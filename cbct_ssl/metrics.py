from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    pred = prediction == label
    truth = target == label
    total = int(pred.sum() + truth.sum())
    if total == 0:
        return float("nan")
    return float(2 * np.logical_and(pred, truth).sum() / total)


def iou_score(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    union = np.logical_or(prediction == label, target == label).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(prediction == label, target == label).sum() / union)


def normalized_surface_dice(prediction: np.ndarray, target: np.ndarray, label: int, spacing: Iterable[float], tolerance_mm: float = 1.0) -> float:
    pred = prediction == label
    truth = target == label
    if not pred.any() and not truth.any():
        return float("nan")
    if not pred.any() or not truth.any():
        return 0.0
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    truth_surface = np.logical_xor(truth, binary_erosion(truth))
    distance_to_truth = distance_transform_edt(~truth_surface, sampling=tuple(spacing))
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=tuple(spacing))
    accepted = (distance_to_truth[pred_surface] <= tolerance_mm).sum() + (distance_to_pred[truth_surface] <= tolerance_mm).sum()
    denominator = pred_surface.sum() + truth_surface.sum()
    return float(accepted / max(denominator, 1))


def macro_metrics(prediction: np.ndarray, target: np.ndarray, spacing: Iterable[float], labels: Iterable[int] | None = None) -> dict[str, float]:
    if labels is None:
        labels = sorted(set(np.unique(prediction)).union(np.unique(target)) - {0})
    labels = list(labels)
    dice = [dice_score(prediction, target, label) for label in labels]
    iou = [iou_score(prediction, target, label) for label in labels]
    nsd = [normalized_surface_dice(prediction, target, label, spacing) for label in labels]
    return {
        "macro_dice": float(np.nanmean(dice)) if dice else float("nan"),
        "macro_iou": float(np.nanmean(iou)) if iou else float("nan"),
        "macro_nsd_1mm": float(np.nanmean(nsd)) if nsd else float("nan"),
        "labels_evaluated": len(labels),
    }
