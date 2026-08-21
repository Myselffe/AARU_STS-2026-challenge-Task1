from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.augment import artifact_suppressed_view
from cbct_ssl.losses import segmentation_with_auxiliary, weighted_restoration_loss
from cbct_ssl.model import ArtifactAwareResUNet3D
from cbct_ssl.precision import autocast_context, resolve_autocast


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtifactAwareResUNet3D(
        in_channels=3, num_classes=5, channels=(8, 16, 24, 32, 48), residual_blocks=(1, 1, 1, 1, 1)
    ).to(device)
    image = torch.randn(1, 1, 32, 64, 64, device=device)
    metal = torch.zeros_like(image)
    target = torch.randint(0, 5, (1, 32, 64, 64), device=device)
    amp_enabled, amp_dtype = resolve_autocast(device, True, "bfloat16")
    model_input = torch.cat([image, artifact_suppressed_view(image, metal), metal], dim=1)
    with autocast_context(device, amp_enabled, amp_dtype):
        output = model(model_input, perturb=True)
    loss = segmentation_with_auxiliary(output, target, 1.0, 1.0, (0.25, 0.125), None)
    loss = loss + 0.15 * weighted_restoration_loss(output["restored"], image, metal, 3.0)
    loss.backward()
    assert torch.isfinite(loss), "The forward/backward pass produced a non-finite loss."
    precision = str(amp_dtype).replace("torch.", "") if amp_enabled else "float32"
    print(f"Smoke test passed on {device} with {precision}; loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
