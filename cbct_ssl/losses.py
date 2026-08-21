from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    """Return zero with a valid autograd path without reading tensor values.

    ``reference.sum() * 0`` is unsafe under AMP because a large FP16 reduction
    can overflow to Inf and turn the intended zero into NaN.
    """
    return reference.float().reshape(-1)[:0].sum()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, include_background: bool = False, eps: float = 1e-6) -> torch.Tensor:
    # Reductions cover millions of voxels.  FP16 sums can exceed 65504 even
    # when every probability is valid, so all structural losses use FP32.
    probabilities = logits.float().softmax(dim=1)
    valid = target >= 0
    safe_target = target.clamp_min(0)
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).movedim(-1, 1).to(probabilities.dtype)
    valid = valid[:, None].to(probabilities.dtype)
    probabilities = probabilities * valid
    one_hot = one_hot * valid
    dimensions = (0, 2, 3, 4)
    intersection = (probabilities * one_hot).sum(dimensions)
    denominator = probabilities.sum(dimensions) + one_hot.sum(dimensions)
    dice = (2 * intersection + eps) / (denominator + eps)
    if not include_background and dice.numel() > 1:
        dice = dice[1:]
    return 1 - dice.mean()


def foreground_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    foreground_probability = 1.0 - logits.float().softmax(dim=1)[:, 0]
    foreground_target = (target > 0).to(foreground_probability.dtype)
    valid = target >= 0
    foreground_probability = foreground_probability * valid
    foreground_target = foreground_target * valid
    dimensions = (0, 1, 2, 3)
    intersection = (foreground_probability * foreground_target).sum(dimensions)
    denominator = foreground_probability.sum(dimensions) + foreground_target.sum(dimensions)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def boundary_target(target: torch.Tensor) -> torch.Tensor:
    foreground = (target > 0).float()[:, None]
    dilated = F.max_pool3d(foreground, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-foreground, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_loss(boundary_logits: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if boundary_logits is None:
        return _differentiable_zero(target)
    boundary_logits = boundary_logits.float()
    expected = boundary_target(target)
    bce = F.binary_cross_entropy_with_logits(boundary_logits, expected)
    probability = boundary_logits.sigmoid()
    intersection = (probability * expected).sum()
    dice = (2.0 * intersection + 1.0) / (probability.sum() + expected.sum() + 1.0)
    return bce + (1.0 - dice)


def _soft_erode(value: torch.Tensor) -> torch.Tensor:
    return -F.max_pool3d(-value, kernel_size=3, stride=1, padding=1)


def _soft_dilate(value: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(value, kernel_size=3, stride=1, padding=1)


def _soft_skeleton(value: torch.Tensor, iterations: int) -> torch.Tensor:
    opened = _soft_dilate(_soft_erode(value))
    skeleton = F.relu(value - opened)
    current = value
    for _ in range(max(0, int(iterations) - 1)):
        current = _soft_erode(current)
        opened = _soft_dilate(_soft_erode(current))
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_indices: Sequence[int],
    iterations: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    selected = [int(index) for index in class_indices if 0 < int(index) < logits.shape[1]]
    if not selected:
        return _differentiable_zero(logits)
    probability = logits.float().softmax(dim=1)[:, selected].sum(dim=1, keepdim=True)
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    for index in selected:
        target_mask |= target == index
    target_probability = target_mask[:, None].to(probability.dtype)
    if not target_mask.any():
        return _differentiable_zero(logits)
    skeleton_prediction = _soft_skeleton(probability, iterations)
    skeleton_target = _soft_skeleton(target_probability, iterations)
    precision = (skeleton_prediction * target_probability).sum() / skeleton_prediction.sum().clamp_min(eps)
    sensitivity = (skeleton_target * probability).sum() / skeleton_target.sum().clamp_min(eps)
    return 1.0 - (2.0 * precision * sensitivity + eps) / (precision + sensitivity + eps)


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float,
    ce_weight: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits.float(), target, weight=class_weights, ignore_index=-1)
    dice = soft_dice_loss(logits, target)
    return dice_weight * dice + ce_weight * ce


def segmentation_with_auxiliary(
    output: dict,
    target: torch.Tensor,
    dice_weight: float,
    ce_weight: float,
    auxiliary_weights: Sequence[float],
    class_weights: torch.Tensor | None,
    foreground_weight: float = 0.0,
    boundary_weight: float = 0.0,
    topology_weight: float = 0.0,
    topology_class_indices: Sequence[int] = (),
    topology_iterations: int = 3,
) -> torch.Tensor:
    total = segmentation_loss(output["logits"], target, dice_weight, ce_weight, class_weights)
    if foreground_weight > 0:
        total = total + float(foreground_weight) * foreground_dice_loss(output["logits"], target)
    if boundary_weight > 0:
        total = total + float(boundary_weight) * boundary_loss(output.get("boundary"), target)
    if topology_weight > 0:
        total = total + float(topology_weight) * soft_cldice_loss(
            output["logits"], target, topology_class_indices, topology_iterations
        )
    for weight, logits in zip(auxiliary_weights, output["auxiliary"]):
        downsampled = F.interpolate(target[:, None].float(), size=logits.shape[-3:], mode="nearest")[:, 0].long()
        total = total + float(weight) * segmentation_loss(logits, downsampled, dice_weight, ce_weight, class_weights)
    return total


def weighted_restoration_loss(restored: torch.Tensor, target: torch.Tensor, artifact_mask: torch.Tensor, metal_multiplier: float) -> torch.Tensor:
    # The target is valid in synthetically corrupted voxels. A small identity
    # weight outside the mask keeps the residual head stable.
    restored_float = restored.float()
    target_float = target.float()
    weights = 0.05 + artifact_mask.float() * (1.0 + float(metal_multiplier))
    return (F.smooth_l1_loss(restored_float, target_float, reduction="none") * weights).sum() / weights.sum().clamp_min(1)


def masked_consistency_loss(student_logits: torch.Tensor, teacher_probabilities: torch.Tensor, confidence_mask: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted teacher/student KL in numerically safe FP32.

    CUDA autocast can underflow saturated teacher probabilities to exactly zero
    and overflow perturbed student logits.  Computing KL directly in FP16 can
    then evaluate expressions such as ``0 * inf`` and return NaN.  The caller
    rejects non-finite logits; the conversion and normalization here prevent
    ordinary FP16 saturation from destabilising the optimizer.
    """
    student = student_logits.float().clamp(-30.0, 30.0)
    teacher = teacher_probabilities.float().clamp_min(1e-7)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-7)
    student_log_probabilities = F.log_softmax(student, dim=1)
    divergence = F.kl_div(student_log_probabilities, teacher, reduction="none").sum(dim=1)
    weights = confidence_mask.float()
    active = weights > 0
    if not active.any():
        return _differentiable_zero(student)
    return (divergence[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-6)


def masked_pseudo_label_loss(student_logits: torch.Tensor, pseudo_labels: torch.Tensor, confidence_mask: torch.Tensor) -> torch.Tensor:
    active = confidence_mask > 0
    target = pseudo_labels.clone()
    target[~active] = -1
    if not active.any():
        return _differentiable_zero(student_logits)
    # Cross entropy is deliberately evaluated in FP32.  Clipping affects only
    # the unsupervised objective and avoids extreme perturbed logits dominating
    # a reliable supervised update.
    loss = F.cross_entropy(student_logits.float().clamp(-30.0, 30.0), target, ignore_index=-1, reduction="none")
    weights = confidence_mask.float()
    return (loss[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-6)


def artifact_invariance_loss(
    corrupted_logits: torch.Tensor,
    clean_probabilities: torch.Tensor,
    artifact_mask: torch.Tensor,
) -> torch.Tensor:
    corrupted = corrupted_logits.float().clamp(-30.0, 30.0)
    clean = clean_probabilities.float().clamp_min(1e-7)
    clean = clean / clean.sum(dim=1, keepdim=True).clamp_min(1e-7)
    divergence = F.kl_div(F.log_softmax(corrupted, dim=1), clean, reduction="none").sum(dim=1)
    weights = artifact_mask[:, 0].float().clamp(0.0, 1.0)
    active = weights > 0
    if not active.any():
        return _differentiable_zero(corrupted)
    return (divergence[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-6)
