from __future__ import annotations

from contextlib import nullcontext
import warnings

import torch


def resolve_autocast(
    device: torch.device,
    enabled: bool,
    requested_dtype: str = "bfloat16",
) -> tuple[bool, torch.dtype]:
    """Resolve a safe autocast policy for large 3-D segmentation tensors."""
    if not enabled or device.type != "cuda":
        return False, torch.float32
    name = str(requested_dtype).strip().lower().replace("torch.", "")
    if name in {"bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        warnings.warn(
            "BF16 was requested but this CUDA device does not support it; "
            "falling back to FP32 instead of numerically fragile FP16.",
            RuntimeWarning,
        )
        return False, torch.float32
    if name in {"fp16", "float16", "half"}:
        return True, torch.float16
    if name in {"fp32", "float32", "none", "off"}:
        return False, torch.float32
    raise ValueError("amp_dtype must be one of: bfloat16, float16, float32")


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def grad_scaler_enabled(autocast_enabled: bool, dtype: torch.dtype) -> bool:
    # BF16 has the FP32 exponent range and does not need dynamic loss scaling.
    return bool(autocast_enabled and dtype == torch.float16)
