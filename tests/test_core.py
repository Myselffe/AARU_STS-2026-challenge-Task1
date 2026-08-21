from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.augment import artifact_suppressed_view, synthesize_metal_artifact
from cbct_ssl.engine import (
    _automatic_class_weights,
    _nested_tensors_are_finite,
    _network_input,
    _pseudo_confidence_weights,
)
from cbct_ssl.inference import remove_small_components
from cbct_ssl.io import make_split
from cbct_ssl.losses import (
    masked_consistency_loss,
    masked_pseudo_label_loss,
    segmentation_with_auxiliary,
    soft_cldice_loss,
)
from cbct_ssl.model import ArtifactAwareResUNet3D
from cbct_ssl.precision import resolve_autocast


def test_model_and_structure_losses_are_finite() -> None:
    model = ArtifactAwareResUNet3D(
        3, 4, channels=(8, 16, 24, 32), residual_blocks=(1, 1, 1, 1)
    )
    image = torch.randn(1, 1, 16, 32, 32)
    metal = torch.zeros_like(image)
    model_input = torch.cat([image, artifact_suppressed_view(image, metal), metal], dim=1)
    target = torch.zeros(1, 16, 32, 32, dtype=torch.long)
    target[:, 3:13, 6:26, 6:26] = 1
    target[:, 6:11, 14:18, 14:18] = 2
    output = model(model_input)
    loss = segmentation_with_auxiliary(
        output,
        target,
        1.0,
        1.0,
        (0.25, 0.125),
        None,
        foreground_weight=0.25,
        boundary_weight=0.1,
        topology_weight=0.1,
        topology_class_indices=(2,),
        topology_iterations=2,
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_rare_classes_receive_larger_automatic_weight() -> None:
    info = {"class_voxels": {"0": 100000, "1": 10000, "2": 100}}
    weights = _automatic_class_weights(info, 3, [0.5, 5.0])
    assert weights[2] > weights[1] >= weights[0]


def test_uncertain_background_is_not_used_as_pseudo_label() -> None:
    probabilities = torch.tensor([[[[[0.95]]], [[[0.03]]], [[[0.02]]]]])
    labels = probabilities.argmax(dim=1)
    config = {
        "background_threshold": 0.97,
        "foreground_threshold": 0.72,
        "tubular_threshold": 0.62,
        "confidence_power": 2.0,
        "max_normalized_entropy": 0.55,
        "foreground_unsupervised_weight": 2.0,
    }
    weights = _pseudo_confidence_weights(probabilities, labels, config, {2})
    assert weights.item() == 0.0


def test_confident_background_only_patch_is_not_used_as_pseudo_label() -> None:
    probabilities = torch.zeros(1, 3, 2, 2, 2)
    probabilities[:, 0] = 0.999
    probabilities[:, 1:] = 0.0005
    labels = probabilities.argmax(dim=1)
    config = {
        "background_threshold": 0.97,
        "foreground_threshold": 0.72,
        "tubular_threshold": 0.62,
        "confidence_power": 2.0,
        "max_normalized_entropy": 0.55,
        "foreground_unsupervised_weight": 2.0,
        "background_unsupervised_weight": 0.1,
        "minimum_foreground_pseudo_voxels": 1,
        "max_background_to_foreground_ratio": 4.0,
    }
    weights = _pseudo_confidence_weights(probabilities, labels, config, {2})
    assert torch.count_nonzero(weights) == 0


def test_pseudo_background_is_capped_relative_to_foreground() -> None:
    probabilities = torch.zeros(1, 3, 1, 1, 10)
    probabilities[:, 0] = 0.999
    probabilities[:, 1:] = 0.0005
    probabilities[:, 0, 0, 0, :2] = 0.01
    probabilities[:, 1, 0, 0, :2] = 0.98
    probabilities[:, 2, 0, 0, :2] = 0.01
    labels = probabilities.argmax(dim=1)
    config = {
        "background_threshold": 0.97,
        "foreground_threshold": 0.72,
        "tubular_threshold": 0.62,
        "confidence_power": 2.0,
        "max_normalized_entropy": 0.55,
        "foreground_unsupervised_weight": 2.0,
        "background_unsupervised_weight": 0.1,
        "minimum_foreground_pseudo_voxels": 1,
        "max_background_to_foreground_ratio": 2.0,
    }
    weights = _pseudo_confidence_weights(probabilities, labels, config, {2})
    active_background = ((labels == 0) & (weights > 0)).sum()
    active_foreground = ((labels > 0) & (weights > 0)).sum()
    assert active_background <= 2 * active_foreground


def test_saturated_half_precision_unsupervised_losses_are_finite() -> None:
    logits = torch.tensor([[[[[65000.0]]], [[[-65000.0]]], [[[-65000.0]]]]], dtype=torch.float16)
    teacher = torch.tensor([[[[[1.0]]], [[[0.0]]], [[[0.0]]]]], dtype=torch.float16)
    labels = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    weights = torch.ones_like(labels, dtype=torch.float32)
    consistency = masked_consistency_loss(logits, teacher, weights)
    pseudo = masked_pseudo_label_loss(logits, labels, weights)
    assert torch.isfinite(consistency)
    assert torch.isfinite(pseudo)


def test_nonfinite_checkpoint_state_is_detected() -> None:
    assert _nested_tensors_are_finite({"model": {"weight": torch.ones(2)}})
    assert not _nested_tensors_are_finite({"model": {"weight": torch.tensor([1.0, float("nan")])}})


def test_empty_topology_target_returns_safe_zero_under_half_precision() -> None:
    logits = torch.full((1, 4, 2, 4, 4), 65000.0, dtype=torch.float16, requires_grad=True)
    target = torch.zeros((1, 2, 4, 4), dtype=torch.long)
    loss = soft_cldice_loss(logits, target, class_indices=(2, 3), iterations=2)
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_network_input_clips_extreme_normalized_values() -> None:
    image = torch.full((1, 1, 3, 5, 5), -1000.0)
    image[..., 2:] = 1000.0
    metal = torch.full_like(image, -2.0)
    metal[..., 2:] = 3.0
    value = _network_input(image, metal)
    assert torch.isfinite(value).all()
    assert value[:, :2].abs().max() <= 8.0
    assert value[:, 2:].min() >= 0.0
    assert value[:, 2:].max() <= 1.0


def test_cpu_precision_policy_disables_autocast_safely() -> None:
    enabled, dtype = resolve_autocast(torch.device("cpu"), True, "bfloat16")
    assert not enabled
    assert dtype == torch.float32


def test_component_filter_preserves_configured_thin_class() -> None:
    segmentation = np.zeros((8, 8, 8), dtype=np.int16)
    segmentation[1, 1, 1] = 1
    segmentation[2, 2, 2] = 2
    filtered = remove_small_components(segmentation, minimum_voxels=4, preserve_labels=(2,))
    assert filtered[1, 1, 1] == 0
    assert filtered[2, 2, 2] == 2


def test_split_retains_artifact_cases_in_train_and_validation() -> None:
    ids = [f"normal_{index}" for index in range(8)] + [f"case_{index}_with-artifacts" for index in range(4)]
    split = make_split(ids, 0.25, 7)
    assert any("with-artifacts" in value for value in split["train"])
    assert any("with-artifacts" in value for value in split["val"])


def test_synthetic_artifact_marks_corrupted_support() -> None:
    torch.manual_seed(4)
    image = torch.randn(1, 1, 16, 32, 32)
    corrupted, mask = synthesize_metal_artifact(image, 1.0, 1, (2, 3))
    assert mask.max() > 0
    assert not torch.equal(corrupted, image)
