from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def _rand_uniform(low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty((), device=device).uniform_(low, high)


def weak_intensity_augment(
    image: torch.Tensor,
    noise_std: float,
    contrast_range: Sequence[float],
    brightness_range: Sequence[float],
) -> torch.Tensor:
    """Appearance-only transform; voxel coordinates are deliberately unchanged."""
    result = image.clone()
    for batch_index in range(result.shape[0]):
        contrast = _rand_uniform(float(contrast_range[0]), float(contrast_range[1]), result.device)
        brightness = _rand_uniform(float(brightness_range[0]), float(brightness_range[1]), result.device)
        result[batch_index] = result[batch_index] * contrast + brightness
    if noise_std > 0:
        result = result + torch.randn_like(result) * noise_std
    return result


def strong_intensity_augment(
    image: torch.Tensor,
    noise_std: float,
    contrast_range: Sequence[float],
    brightness_range: Sequence[float],
    gamma_range: Sequence[float],
    blur_probability: float,
    low_resolution_probability: float,
) -> torch.Tensor:
    """Coordinate-preserving strong view used by the student network."""
    result = weak_intensity_augment(image, noise_std, contrast_range, brightness_range)
    for batch_index in range(result.shape[0]):
        current = result[batch_index:batch_index + 1]
        if torch.rand((), device=result.device) < blur_probability:
            current = F.avg_pool3d(current, kernel_size=3, stride=1, padding=1)
        if torch.rand((), device=result.device) < low_resolution_probability:
            spatial = current.shape[-3:]
            scale = float(_rand_uniform(0.50, 0.85, result.device))
            low_shape = tuple(max(2, int(round(size * scale))) for size in spatial)
            current = F.interpolate(current, size=low_shape, mode="trilinear", align_corners=False)
            current = F.interpolate(current, size=spatial, mode="trilinear", align_corners=False)
        gamma = _rand_uniform(float(gamma_range[0]), float(gamma_range[1]), result.device)
        bounded = current.clamp(-3.0, 3.0)
        current = ((bounded + 3.0) / 6.0).clamp(0.0, 1.0).pow(gamma) * 6.0 - 3.0
        result[batch_index:batch_index + 1] = current
    return result


def artifact_suppressed_view(image: torch.Tensor, artifact_mask: torch.Tensor) -> torch.Tensor:
    """Build a conservative MAR proxy without inventing anatomy.

    Only voxels indicated by the soft artifact mask are replaced by a local
    anisotropic mean. The raw image remains a separate network input.
    """
    if image.shape != artifact_mask.shape:
        raise ValueError("image and artifact_mask must have identical shapes")
    smooth = F.avg_pool3d(image, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
    alpha = artifact_mask.clamp(0.0, 1.0)
    return image * (1.0 - alpha) + smooth * alpha


def _ellipsoid_mask(shape: tuple[int, int, int], center: tuple[int, int, int], radii: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    depth, height, width = shape
    z0, y0, x0 = center
    rz, ry, rx = radii
    z_start, z_end = max(0, z0 - rz), min(depth, z0 + rz + 1)
    y_start, y_end = max(0, y0 - ry), min(height, y0 + ry + 1)
    x_start, x_end = max(0, x0 - rx), min(width, x0 + rx + 1)
    z = torch.arange(z_start, z_end, device=device, dtype=torch.float32)[:, None, None]
    y = torch.arange(y_start, y_end, device=device, dtype=torch.float32)[None, :, None]
    x = torch.arange(x_start, x_end, device=device, dtype=torch.float32)[None, None, :]
    local = ((z - z0) / max(rz, 1)) ** 2 + ((y - y0) / max(ry, 1)) ** 2 + ((x - x0) / max(rx, 1)) ** 2 <= 1
    mask = torch.zeros(shape, dtype=torch.float32, device=device)
    mask[z_start:z_end, y_start:y_end, x_start:x_end] = local.float()
    return mask


def synthesize_metal_artifact(
    image: torch.Tensor,
    probability: float,
    max_artifacts: int,
    streaks_per_artifact: Sequence[int] = (4, 9),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create label-preserving metal-like saturation and streak corruption.

    The original CBCT is the reconstruction target. This is self-supervised
    corruption, not a claim that the raw scan is artifact-free.
    """
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError("Expected image shape [B, 1, D, H, W].")
    augmented = image.clone()
    masks = torch.zeros_like(image)
    batch, _, depth, height, width = image.shape
    device = image.device
    yy = torch.linspace(-1, 1, height, device=device)[None, :, None]
    xx = torch.linspace(-1, 1, width, device=device)[None, None, :]

    for batch_index in range(batch):
        if torch.rand((), device=device) >= probability:
            continue
        count = int(torch.randint(1, max(2, max_artifacts + 1), (), device=device))
        combined = torch.zeros((depth, height, width), device=device)
        streak = torch.zeros_like(combined)
        high_density = torch.nonzero(image[batch_index, 0] >= torch.quantile(image[batch_index, 0], 0.85))
        for _ in range(count):
            if high_density.numel() and torch.rand((), device=device) < 0.85:
                selected = high_density[torch.randint(0, high_density.shape[0], (), device=device)]
                center = tuple(int(value) for value in selected.tolist())
            else:
                center = (
                    int(torch.randint(max(1, depth // 8), max(2, depth - max(1, depth // 8)), (), device=device)),
                    int(torch.randint(max(1, height // 8), max(2, height - max(1, height // 8)), (), device=device)),
                    int(torch.randint(max(1, width // 8), max(2, width - max(1, width // 8)), (), device=device)),
                )
            radii = (
                int(torch.randint(2, max(3, depth // 12 + 2), (), device=device)),
                int(torch.randint(3, max(4, height // 18 + 3), (), device=device)),
                int(torch.randint(3, max(4, width // 18 + 3), (), device=device)),
            )
            metal = _ellipsoid_mask((depth, height, width), center, radii, device)
            combined = torch.maximum(combined, metal)
            z_weight = torch.exp(-((torch.arange(depth, device=device, dtype=torch.float32) - center[0]) / max(radii[0] * 3, 1)) ** 2)
            number_of_streaks = int(torch.randint(
                int(streaks_per_artifact[0]), max(int(streaks_per_artifact[0]) + 1, int(streaks_per_artifact[1]) + 1),
                (), device=device,
            ))
            for _ in range(number_of_streaks):
                theta = _rand_uniform(0.0, math.pi, device)
                signed_distance = torch.abs(
                    (yy - (2 * center[1] / max(height - 1, 1) - 1)) * torch.cos(theta)
                    + (xx - (2 * center[2] / max(width - 1, 1) - 1)) * torch.sin(theta)
                )
                line = torch.exp(-signed_distance * _rand_uniform(10.0, 28.0, device))
                sign = -1.0 if torch.rand((), device=device) < 0.55 else 1.0
                streak = streak + sign * _rand_uniform(0.05, 0.25, device) * z_weight[:, None, None] * line

        soft = F.avg_pool3d(combined[None, None], kernel_size=5, stride=1, padding=2)[0, 0]
        augmented[batch_index, 0] = augmented[batch_index, 0] + streak
        augmented[batch_index, 0] = torch.where(combined > 0, torch.full_like(combined, 4.0), augmented[batch_index, 0])
        streak_support = (streak.abs() / 0.20).clamp(0.0, 1.0)
        masks[batch_index, 0] = torch.maximum(torch.maximum(combined, soft), streak_support)
    return augmented, masks


def disruptive_autoencoder_corruption(
    image: torch.Tensor,
    metal_probability: float,
    max_artifacts: int,
    cube_mask_probability: float,
    downsample_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt a CBCT patch for self-supervised structure/MAR pretraining.

    It combines artifact simulation with masked-volume and low-resolution
    reconstruction. Every transform preserves spatial coordinates.
    """
    corrupted, mask = synthesize_metal_artifact(image, metal_probability, max_artifacts)
    batch, _, depth, height, width = image.shape
    for batch_index in range(batch):
        if torch.rand((), device=image.device) < cube_mask_probability:
            cube_size = (
                int(torch.randint(max(2, depth // 10), max(3, depth // 4 + 1), (), device=image.device)),
                int(torch.randint(max(3, height // 10), max(4, height // 4 + 1), (), device=image.device)),
                int(torch.randint(max(3, width // 10), max(4, width // 4 + 1), (), device=image.device)),
            )
            starts = tuple(
                int(torch.randint(0, max(1, size - cube + 1), (), device=image.device))
                for size, cube in zip((depth, height, width), cube_size)
            )
            z, y, x = starts
            dz, dy, dx = cube_size
            corrupted[batch_index, :, z:z + dz, y:y + dy, x:x + dx] = 0.0
            mask[batch_index, :, z:z + dz, y:y + dy, x:x + dx] = 1.0
        if torch.rand((), device=image.device) < downsample_probability:
            low_resolution = F.avg_pool3d(corrupted[batch_index:batch_index + 1], kernel_size=2, stride=2, ceil_mode=True)
            corrupted[batch_index:batch_index + 1] = F.interpolate(
                low_resolution, size=(depth, height, width), mode="trilinear", align_corners=False
            )
            mask[batch_index] = torch.maximum(mask[batch_index], torch.full_like(mask[batch_index], 0.25))
    return corrupted, mask
