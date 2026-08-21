from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from .augment import artifact_suppressed_view
from .io import (
    case_id_from_path,
    denormalize_cbct,
    nifti_files,
    normalize_cbct,
    load_prepared_case,
    read_nifti,
    read_nifti_metadata,
    resample_spacing,
    resize_to_shape,
)
from .precision import autocast_context, resolve_autocast


def _positions(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    stride = max(1, int(round(patch * (1 - overlap))))
    positions = list(range(0, max(1, length - patch + 1), stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def _center_importance(patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Prefer predictions made near the center of a sliding-window patch."""
    axes = []
    for size in patch_size:
        coordinate = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
        axes.append((1.0 - coordinate.abs()).clamp_min(1e-3))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


@torch.inference_mode()
def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    metal: torch.Tensor,
    patch_size: tuple[int, int, int],
    overlap: float,
    amp: bool,
    write_restored: bool = False,
    tta_axes: tuple[int, ...] = (),
    amp_dtype: str = "bfloat16",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Low-memory sliding-window inference.

    The complete multi-class logit volume is never allocated. Each voxel keeps
    the prediction from the patch where it lies closest to the patch center.
    The GPU therefore holds only one patch and its activations.
    """
    if image.ndim != 4 or metal.shape != image.shape:
        raise ValueError("image and metal must both have shape [1, D, H, W].")
    device = next(model.parameters()).device
    original_shape = tuple(image.shape[-3:])
    pads: list[int] = []
    for current, requested in zip(reversed(original_shape), reversed(patch_size)):
        missing = max(0, requested - current)
        pads.extend([missing // 2, missing - missing // 2])
    # Keep the full CBCT on CPU. Only the current patch is transferred to GPU.
    bounded_image = image.float().clamp(-8.0, 8.0)
    bounded_metal = metal.float().clamp(0.0, 1.0)
    suppressed = artifact_suppressed_view(bounded_image[None], bounded_metal[None])[0]
    combined = F.pad(
        torch.cat([bounded_image.cpu(), suppressed.cpu(), bounded_metal.cpu()], dim=0)[None],
        pads,
        mode="replicate",
    )[0]
    padded_shape = tuple(combined.shape[-3:])
    z_positions = _positions(padded_shape[0], patch_size[0], overlap)
    y_positions = _positions(padded_shape[1], patch_size[1], overlap)
    x_positions = _positions(padded_shape[2], patch_size[2], overlap)
    segmentation = torch.zeros(padded_shape, dtype=torch.int16)
    best_importance = torch.full(padded_shape, -1.0, dtype=torch.float32)
    patch_importance = _center_importance(patch_size)
    restoration_sum = torch.zeros((1, *padded_shape), dtype=torch.float32) if write_restored else None
    restoration_count = torch.zeros((1, *padded_shape), dtype=torch.float32) if write_restored else None
    autocast_enabled, autocast_dtype = resolve_autocast(device, amp, amp_dtype)
    model.eval()
    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                patch = combined[
                    :,
                    z:z + patch_size[0],
                    y:y + patch_size[1],
                    x:x + patch_size[2],
                ][None].to(device, non_blocking=True)
                with autocast_context(device, autocast_enabled, autocast_dtype):
                    output = model(
                        patch,
                        return_restoration=write_restored,
                        return_auxiliary=False,
                    )
                    logits = output["logits"]
                    if tta_axes:
                        accumulated = logits.float()
                        for axis in tta_axes:
                            tensor_axis = 2 + int(axis)
                            flipped_output = model(
                                torch.flip(patch, dims=(tensor_axis,)),
                                return_restoration=False,
                                return_auxiliary=False,
                            )
                            accumulated = accumulated + torch.flip(
                                flipped_output["logits"].float(), dims=(tensor_axis,)
                            )
                        logits = accumulated / float(1 + len(tta_axes))
                if not torch.isfinite(logits).all() and autocast_enabled:
                    output = model(
                        patch.float(),
                        return_restoration=write_restored,
                        return_auxiliary=False,
                    )
                    logits = output["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Inference logits contain NaN/Inf at sliding-window position {(z, y, x)}."
                    )
                probability = logits.float().softmax(dim=1)
                confidence, patch_segmentation = probability.max(dim=1)
                patch_segmentation = patch_segmentation[0].to("cpu", dtype=torch.int16)
                patch_confidence = confidence[0].to("cpu", dtype=torch.float32)
                importance_region = best_importance[
                    z:z + patch_size[0],
                    y:y + patch_size[1],
                    x:x + patch_size[2],
                ]
                segmentation_region = segmentation[
                    z:z + patch_size[0],
                    y:y + patch_size[1],
                    x:x + patch_size[2],
                ]
                candidate_importance = patch_importance * (0.5 + patch_confidence)
                update = candidate_importance > importance_region
                segmentation_region[update] = patch_segmentation[update]
                importance_region[update] = candidate_importance[update]
                if write_restored:
                    restored_patch = output["restored"]
                    if restored_patch is None:
                        raise RuntimeError("The model did not return the requested restoration output.")
                    assert restoration_sum is not None and restoration_count is not None
                    restoration_sum[
                        :,
                        z:z + patch_size[0],
                        y:y + patch_size[1],
                        x:x + patch_size[2],
                    ] += restored_patch[0].float().cpu()
                    restoration_count[
                        :,
                        z:z + patch_size[0],
                        y:y + patch_size[1],
                        x:x + patch_size[2],
                    ] += 1
                del patch, output, patch_segmentation, patch_confidence, probability, logits
    restored = (
        restoration_sum / restoration_count.clamp_min(1)
        if restoration_sum is not None and restoration_count is not None
        else None
    )
    # Reverse F.pad's [Wl, Wr, Hl, Hr, Dl, Dr] ordering.
    w_left, w_right, h_left, h_right, d_left, d_right = pads
    z_slice = slice(d_left, segmentation.shape[0] - d_right if d_right else None)
    y_slice = slice(h_left, segmentation.shape[1] - h_right if h_right else None)
    x_slice = slice(w_left, segmentation.shape[2] - w_right if w_right else None)
    cropped_restored = restored[:, z_slice, y_slice, x_slice] if restored is not None else None
    return segmentation[z_slice, y_slice, x_slice], cropped_restored


def remove_small_components(
    segmentation: np.ndarray,
    minimum_voxels: int,
    preserve_labels: Iterable[int] = (),
) -> np.ndarray:
    if minimum_voxels <= 0:
        return segmentation
    result = segmentation.copy()
    preserved = {int(value) for value in preserve_labels}
    for label_id in np.unique(segmentation):
        if label_id == 0:
            continue
        if int(label_id) in preserved:
            continue
        components, count = connected_components(segmentation == label_id)
        for component_id in range(1, count + 1):
            if (components == component_id).sum() < minimum_voxels:
                result[components == component_id] = 0
    return result


def remap_prediction(segmentation: np.ndarray, train_to_raw: dict[str, int]) -> np.ndarray:
    dtype = np.int16 if max(int(value) for value in train_to_raw.values()) <= np.iinfo(np.int16).max else np.int32
    result = np.zeros(segmentation.shape, dtype=dtype)
    for train_id, raw_id in train_to_raw.items():
        result[segmentation == int(train_id)] = int(raw_id)
    return result


def predict_case(
    model: torch.nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    train_to_raw: dict[str, int],
    target_spacing: Iterable[float] | None,
    patch_size: tuple[int, int, int],
    overlap: float,
    amp: bool,
    minimum_component_voxels: int,
    restored_path: str | Path | None = None,
    prepared_path: str | Path | None = None,
    image_resample_order: int = 1,
    preserve_train_labels: Iterable[int] = (),
    tta_axes: tuple[int, ...] = (),
    amp_dtype: str = "bfloat16",
) -> None:
    if prepared_path is not None:
        original_shape, affine, spacing, header = read_nifti_metadata(input_path)
        cached = load_prepared_case(prepared_path)
        cached_original_shape = tuple(int(value) for value in cached["original_shape"])
        if cached_original_shape != original_shape:
            raise ValueError(
                f"Prepared cache geometry mismatch for {input_path}: "
                f"cache={cached_original_shape}, NIfTI={original_shape}."
            )
        normalized = cached["image"]
        metal = cached["metal"]
        normalization_values = cached["normalization"]
        normalization = {
            key: float(value)
            for key, value in zip(
                ("low", "high", "mean", "std", "metal_cutoff"),
                normalization_values,
            )
        }
    else:
        raw_image, affine, spacing, header = read_nifti(input_path)
        original_shape = raw_image.shape
        if target_spacing is not None:
            source_spacing_array = np.asarray(spacing, dtype=np.float64)
            target_spacing_array = np.asarray(tuple(target_spacing), dtype=np.float64)
            target_shape = tuple(
                int(value)
                for value in np.maximum(
                    1,
                    np.rint(np.asarray(original_shape) * source_spacing_array / target_spacing_array),
                ).astype(int)
            )
            if target_shape != tuple(original_shape):
                print(
                    f"[preprocess] {Path(input_path).name}: resampling "
                    f"{original_shape} @ {tuple(round(v, 4) for v in spacing)} -> "
                    f"{target_shape} @ {tuple(round(float(v), 4) for v in target_spacing_array)} "
                    f"(order={image_resample_order})",
                    flush=True,
                )
        prepared = resample_spacing(
            raw_image,
            spacing,
            target_spacing,
            order=image_resample_order,
        )
        normalized, metal, normalization = normalize_cbct(prepared)
    image_tensor = torch.from_numpy(normalized[None])
    metal_tensor = torch.from_numpy(metal[None])
    segmentation_tensor, restored = sliding_window_inference(
        model,
        image_tensor,
        metal_tensor,
        patch_size,
        overlap,
        amp,
        write_restored=restored_path is not None,
        tta_axes=tta_axes,
        amp_dtype=amp_dtype,
    )
    segmentation = segmentation_tensor.numpy().astype(np.int32)
    segmentation = remove_small_components(segmentation, minimum_component_voxels, preserve_train_labels)
    segmentation = resize_to_shape(segmentation, original_shape, order=0).astype(np.int32)
    segmentation = remap_prediction(segmentation, train_to_raw)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_header = header.copy()
    segmentation_header.set_data_dtype(segmentation.dtype)
    nib.save(nib.Nifti1Image(segmentation, affine, segmentation_header), str(output_path))
    if restored_path is not None:
        if restored is None:
            raise RuntimeError("Restoration output was requested but not produced.")
        restored_image = denormalize_cbct(restored[0].cpu().numpy(), normalization)
        restored_image = resize_to_shape(restored_image, original_shape, order=1)
        restored_header = header.copy()
        restored_header.set_data_dtype(np.float32)
        restored_path = Path(restored_path)
        restored_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(restored_image.astype(np.float32), affine, restored_header), str(restored_path))


def input_files(input_dir: str | Path) -> list[Path]:
    paths = nifti_files(input_dir)
    if not paths:
        raise FileNotFoundError(f"No .nii or .nii.gz inputs found in {input_dir}")
    return paths
