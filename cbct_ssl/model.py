from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 6, 4, 3, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int, int] = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int, int] = 1) -> None:
        super().__init__()
        self.first = ConvNormAct(in_channels, out_channels, stride)
        self.second = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.second(self.first(value)) + self.skip(value))


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, downsample: bool) -> None:
        super().__init__()
        layers = [ResidualBlock(in_channels, out_channels, stride=2 if downsample else 1)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(max(0, blocks - 1)))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class UpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        layers = [ResidualBlock(out_channels + skip_channels, out_channels)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(max(0, blocks - 1)))
        self.blocks = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-3:] != skip.shape[-3:]:
            value = F.interpolate(value, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.blocks(torch.cat([value, skip], dim=1))


class AxialContextBlock(nn.Module):
    """Low-cost long-range context at the U-Net bottleneck.

    It keeps the reliable convolutional backbone while adding anisotropic
    depthwise kernels along the slice and in-plane axes.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depth = nn.Conv3d(channels, channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), groups=channels, bias=False)
        self.plane = nn.Conv3d(channels, channels, kernel_size=(1, 7, 7), padding=(0, 3, 3), groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mix(self.depth(value) + self.plane(value))


def perturb_features(value: torch.Tensor, probability: float, keep_range: Sequence[float], noise_scale: float) -> torch.Tensor:
    """Channel dropout, salient-activation masking and relative Gaussian noise."""
    if probability <= 0 or not value.requires_grad or torch.rand((), device=value.device) >= probability:
        return value
    result = F.dropout3d(value, p=0.15, training=True)
    attention = value.detach().abs().mean(dim=1, keepdim=True)
    keep = torch.empty((), device=value.device).uniform_(float(keep_range[0]), float(keep_range[1]))
    threshold = attention.flatten(2).quantile(keep, dim=2, keepdim=True).view(value.shape[0], 1, 1, 1, 1)
    # Remove a random subset of the strongest locations, so weak pulp features remain useful.
    result = result * (attention < threshold).to(result.dtype)
    if noise_scale > 0:
        result = result * (1 + torch.empty_like(result).uniform_(-noise_scale, noise_scale))
    return result


class ArtifactAwareResUNet3D(nn.Module):
    """A compact full-resolution 3-D residual U-Net with segmentation and MAR heads."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: Sequence[int] = (24, 48, 96, 160, 224),
        residual_blocks: Sequence[int] = (1, 2, 2, 2, 2),
        deep_supervision: bool = True,
        axial_context: bool = True,
        boundary_head: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != len(residual_blocks) or len(channels) < 4:
            raise ValueError("channels and residual_blocks must have equal length >= 4.")
        self.deep_supervision = deep_supervision
        self.use_boundary_head = bool(boundary_head)
        self.encoder = nn.ModuleList()
        previous = in_channels
        for index, (width, blocks) in enumerate(zip(channels, residual_blocks)):
            self.encoder.append(ResidualStage(previous, int(width), int(blocks), downsample=index > 0))
            previous = int(width)
        self.context = AxialContextBlock(int(channels[-1])) if axial_context else nn.Identity()
        self.decoder = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for index in range(len(channels) - 1):
            self.decoder.append(
                UpStage(
                    int(reversed_channels[index]),
                    int(reversed_channels[index + 1]),
                    int(reversed_channels[index + 1]),
                    int(residual_blocks[len(channels) - index - 2]),
                )
            )
        self.segmentation_head = nn.Conv3d(int(channels[0]), num_classes, kernel_size=1)
        self.boundary_head = nn.Conv3d(int(channels[0]), 1, kernel_size=1) if self.use_boundary_head else None
        self.auxiliary_heads = nn.ModuleList(
            [nn.Conv3d(int(channels[1]), num_classes, kernel_size=1), nn.Conv3d(int(channels[2]), num_classes, kernel_size=1)]
        )
        self.restoration_head = nn.Sequential(
            ConvNormAct(int(channels[0]), int(channels[0])),
            nn.Conv3d(int(channels[0]), 1, kernel_size=1),
        )

    def forward(
        self,
        value: torch.Tensor,
        perturb: bool = False,
        perturb_probability: float = 0.5,
        keep_range: Sequence[float] = (0.7, 0.9),
        noise_scale: float = 0.2,
        return_restoration: bool = True,
        return_auxiliary: bool | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        encoded: list[torch.Tensor] = []
        current = value
        for stage in self.encoder:
            current = stage(current)
            encoded.append(current)
        encoded[-1] = self.context(encoded[-1])
        if perturb:
            encoded[-1] = perturb_features(encoded[-1], perturb_probability, keep_range, noise_scale)
            encoded[-2] = perturb_features(encoded[-2], perturb_probability, keep_range, noise_scale)
        current = encoded[-1]
        decoded: list[torch.Tensor] = []
        for index, stage in enumerate(self.decoder):
            current = stage(current, encoded[-2 - index])
            decoded.append(current)
        logits = self.segmentation_head(current)
        boundary = self.boundary_head(current) if self.boundary_head is not None else None
        restored = (
            value[:, :1] + 0.20 * torch.tanh(self.restoration_head(current))
            if return_restoration
            else None
        )
        auxiliary: list[torch.Tensor] = []
        if return_auxiliary is None:
            return_auxiliary = self.deep_supervision and self.training
        if self.deep_supervision and return_auxiliary:
            # decoded[-2] has the second-highest resolution, decoded[-3] the third-highest.
            auxiliary = [self.auxiliary_heads[0](decoded[-2]), self.auxiliary_heads[1](decoded[-3])]
        return {"logits": logits, "auxiliary": auxiliary, "restored": restored, "boundary": boundary}
