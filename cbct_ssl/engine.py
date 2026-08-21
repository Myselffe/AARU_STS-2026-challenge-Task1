from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augment import (
    artifact_suppressed_view,
    disruptive_autoencoder_corruption,
    strong_intensity_augment,
    synthesize_metal_artifact,
)
from .dataset import RandomPatchDataset
from .losses import (
    artifact_invariance_loss,
    masked_consistency_loss,
    masked_pseudo_label_loss,
    segmentation_with_auxiliary,
    weighted_restoration_loss,
)
from .model import ArtifactAwareResUNet3D
from .precision import autocast_context, grad_scaler_enabled, resolve_autocast


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(config: dict[str, Any], number_of_classes: int) -> ArtifactAwareResUNet3D:
    model_config = config["model"]
    return ArtifactAwareResUNet3D(
        in_channels=int(model_config["in_channels"]),
        num_classes=number_of_classes,
        channels=tuple(int(value) for value in model_config["channels"]),
        residual_blocks=tuple(int(value) for value in model_config["residual_blocks"]),
        deep_supervision=bool(model_config["deep_supervision"]),
        axial_context=bool(model_config.get("axial_context", True)),
        boundary_head=bool(model_config.get("boundary_head", True)),
    )


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_parameter, parameter in zip(self.model.parameters(), model.parameters()):
            ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def _nested_tensors_are_finite(value: Any) -> bool:
    """Return False when any tensor in a nested checkpoint object is NaN/Inf."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_nested_tensors_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_tensors_are_finite(item) for item in value)
    return True


def _module_is_finite(module: torch.nn.Module) -> bool:
    return _nested_tensors_are_finite(module.state_dict())


def _output_is_finite(output: dict[str, Any]) -> bool:
    return _nested_tensors_are_finite(output)


def _stable_forward(
    model: torch.nn.Module,
    value: torch.Tensor,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    retry_fp32: bool,
    **kwargs: Any,
) -> tuple[dict[str, Any], bool]:
    """Run a model forward and retry non-finite mixed-precision output in FP32."""
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("Network input contains NaN/Inf before the model forward pass.")
    with autocast_context(device, autocast_enabled, autocast_dtype):
        output = model(value, **kwargs)
    if _output_is_finite(output) or not autocast_enabled or not retry_fp32:
        return output, False
    # Parameters remain FP32 under autocast, so a retry can distinguish an AMP
    # activation overflow from corrupt weights or corrupt input data.
    output = model(value.float(), **kwargs)
    return output, True


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config["train"].get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU. This will be very slow for 3-D CBCT training.")
        return torch.device("cpu")
    return torch.device(requested)


def _records_for_ids(records: list[dict[str, Any]], wanted: list[str]) -> list[dict[str, Any]]:
    lookup = {record["id"]: record for record in records}
    absent = set(wanted) - set(lookup)
    if absent:
        raise KeyError(f"Prepared split references missing cases: {sorted(absent)}")
    return [lookup[key] for key in wanted]


def _loader(dataset: RandomPatchDataset, config: dict[str, Any], shuffle: bool = False) -> DataLoader:
    workers = int(config["train"]["num_workers"])
    loader_options: dict[str, Any] = {}
    if workers > 0:
        loader_options["prefetch_factor"] = max(1, int(config["train"].get("prefetch_factor", 1)))
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=True,
        **loader_options,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return result


def _network_input(image: torch.Tensor, metal: torch.Tensor) -> torch.Tensor:
    # Robust normalization should already keep nearly all voxels in this range.
    # Clipping extreme cached/outlier values protects early convolutions without
    # removing the dedicated metal mask or synthetic artifact signal.
    bounded_image = image.float().clamp(-8.0, 8.0)
    bounded_metal = metal.float().clamp(0.0, 1.0)
    suppressed = artifact_suppressed_view(bounded_image, bounded_metal)
    return torch.cat([bounded_image, suppressed, bounded_metal], dim=1)


def _automatic_class_weights(
    info: dict[str, Any], number_of_classes: int, clip_range: list[float] | tuple[float, float]
) -> torch.Tensor:
    counts = torch.tensor(
        [max(1, int(info.get("class_voxels", {}).get(str(index), 1))) for index in range(number_of_classes)],
        dtype=torch.float32,
    )
    frequencies = counts / counts.sum()
    weights = frequencies.clamp_min(1e-12).rsqrt()
    # Background should not dominate, but also should not be eliminated.
    weights /= weights[1:].mean().clamp_min(1e-6) if number_of_classes > 1 else weights.mean()
    return weights.clamp(float(clip_range[0]), float(clip_range[1]))


def _pseudo_confidence_weights(
    probabilities: torch.Tensor,
    pseudo_labels: torch.Tensor,
    semi_config: dict[str, Any],
    tubular_indices: set[int],
) -> torch.Tensor:
    confidence = probabilities.amax(dim=1)
    thresholds = torch.full_like(confidence, float(semi_config["foreground_threshold"]))
    thresholds[pseudo_labels == 0] = float(semi_config["background_threshold"])
    for index in tubular_indices:
        thresholds[pseudo_labels == index] = float(semi_config["tubular_threshold"])
    normalized_entropy = -(
        probabilities.clamp_min(1e-7) * probabilities.clamp_min(1e-7).log()
    ).sum(dim=1) / math.log(max(2, probabilities.shape[1]))
    active = (confidence >= thresholds) & (normalized_entropy <= float(semi_config["max_normalized_entropy"]))
    weights = confidence.pow(float(semi_config["confidence_power"])) * active
    weights = torch.where(
        pseudo_labels > 0,
        weights * float(semi_config["foreground_unsupervised_weight"]),
        weights * float(semi_config.get("background_unsupervised_weight", 0.1)),
    )

    # A teacher that predicts an entire patch as high-confidence background is
    # not providing useful anatomical supervision.  The failed run reached
    # pseudo_coverage=1.0 and pseudo_foreground_fraction=0.0 immediately before
    # NaN propagation.  Suppress such background-only batches, and otherwise
    # cap background voxels so that they cannot overwhelm rare pulp/root labels.
    minimum_foreground = max(1, int(semi_config.get("minimum_foreground_pseudo_voxels", 32)))
    maximum_ratio = max(0.0, float(semi_config.get("max_background_to_foreground_ratio", 4.0)))
    for batch_index in range(weights.shape[0]):
        labels = pseudo_labels[batch_index]
        foreground = (labels > 0) & (weights[batch_index] > 0)
        background = (labels == 0) & (weights[batch_index] > 0)
        foreground_count = int(foreground.sum().item())
        background_indices = torch.nonzero(background.flatten(), as_tuple=False).flatten()
        if foreground_count < minimum_foreground:
            weights[batch_index][labels == 0] = 0
            continue
        maximum_background = int(maximum_ratio * foreground_count)
        if background_indices.numel() > maximum_background:
            flat_weights = weights[batch_index].flatten()
            flat_weights[background_indices] = 0
            if maximum_background > 0:
                # Evenly spaced selection is deterministic and avoids allocating
                # a full-volume random tensor or running a very large top-k.
                stride = max(1, math.ceil(background_indices.numel() / maximum_background))
                selected = background_indices[::stride][:maximum_background]
                flat_weights[selected] = confidence[batch_index].flatten()[selected] * float(
                    semi_config.get("background_unsupervised_weight", 0.1)
                )
    return weights


def _ramp(step: int, start: int, duration: int) -> float:
    if step < start:
        return 0.0
    if duration <= 0:
        return 1.0
    progress = min(1.0, (step - start) / duration)
    return float(math.exp(-5 * (1 - progress) ** 2))


@torch.no_grad()
def validation_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> float:
    was_training = model.training
    model.eval()
    scores: list[torch.Tensor] = []
    for batch in loader:
        batch = _to_device(batch, device)
        output, _ = _stable_forward(
            model, _network_input(batch["image"], batch["metal"]), device,
            amp_enabled, amp_dtype, retry_fp32=True,
            return_restoration=False, return_auxiliary=False,
        )
        logits = output["logits"]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Validation logits contain NaN/Inf; refusing to report a misleading Dice score.")
        prediction = logits.argmax(dim=1)
        target = batch["label"]
        for label in range(1, logits.shape[1]):
            pred_mask = prediction == label
            target_mask = target == label
            denominator = pred_mask.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
            valid = denominator > 0
            if valid.any():
                dice = 2 * (pred_mask & target_mask).sum(dim=(1, 2, 3)).float() / denominator.clamp_min(1)
                scores.extend(dice[valid].cpu())
    model.train(was_training)
    return float(torch.stack(scores).mean()) if scores else 0.0


@torch.no_grad()
def full_volume_validation(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    """A memory-bounded proxy for the official instance/image/IA score.

    NSD is intentionally left to the official evaluator; checkpoint selection
    nevertheless uses whole volumes and exact class ids instead of random patches.
    """
    from .inference import sliding_window_inference
    from .io import load_prepared_case

    instance_dice: list[float] = []
    instance_iou: list[float] = []
    instance_weights: list[float] = []
    image_dice: list[float] = []
    image_iou: list[float] = []
    identification: list[float] = []
    image_weights: list[float] = []
    inference_config = config["inference"]
    was_training = model.training
    model.eval()
    for record in records:
        case = load_prepared_case(record["file"])
        prediction, _ = sliding_window_inference(
            model,
            torch.from_numpy(case["image"][None]),
            torch.from_numpy(case["metal"][None]),
            tuple(int(value) for value in inference_config["patch_size"]),
            float(inference_config["overlap"]),
            bool(inference_config["amp"]),
            write_restored=False,
            tta_axes=(),
            amp_dtype=str(inference_config.get("amp_dtype", config["train"].get("amp_dtype", "bfloat16"))),
        )
        pred = prediction.numpy()
        target = case["label"]
        case_name = str(record["id"]).lower().replace("_", "-")
        weight = float(config["data"].get("artifact_case_sampling_weight", 2.0)) if "with-artifacts" in case_name else 1.0
        pred_foreground = pred > 0
        target_foreground = target > 0
        intersection = float(np.logical_and(pred_foreground, target_foreground).sum())
        denominator = float(pred_foreground.sum() + target_foreground.sum())
        union = float(np.logical_or(pred_foreground, target_foreground).sum())
        image_dice.append(2.0 * intersection / max(1.0, denominator))
        image_iou.append(intersection / max(1.0, union))
        image_weights.append(weight)
        target_labels = np.unique(target[target > 0])
        pred_labels = np.unique(pred[pred > 0])
        matched = 0
        for label in target_labels:
            expected = target == label
            observed = pred == label
            overlap = float(np.logical_and(expected, observed).sum())
            class_denominator = float(expected.sum() + observed.sum())
            class_union = float(np.logical_or(expected, observed).sum())
            dice_value = 2.0 * overlap / max(1.0, class_denominator)
            iou_value = overlap / max(1.0, class_union)
            instance_dice.append(dice_value)
            instance_iou.append(iou_value)
            instance_weights.append(weight)
            matched += int(iou_value >= 0.5)
        union_labels = len(set(target_labels.tolist()).union(set(pred_labels.tolist())))
        identification.append(float(matched) / union_labels if union_labels else 1.0)
    model.train(was_training)

    def weighted(values: list[float], weights: list[float]) -> float:
        return float(np.average(np.asarray(values), weights=np.asarray(weights))) if values else 0.0

    metrics = {
        "dice_instance": weighted(instance_dice, instance_weights),
        "miou_instance": weighted(instance_iou, instance_weights),
        "dice_image": weighted(image_dice, image_weights),
        "miou_image": weighted(image_iou, image_weights),
        "ia": weighted(identification, image_weights),
    }
    metrics["selection_score"] = float(np.mean(list(metrics.values())))
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    step: int,
    best_score: float,
    config: dict[str, Any],
    number_of_classes: int,
) -> None:
    if not _module_is_finite(model) or not _module_is_finite(ema.model):
        raise FloatingPointError(f"Refusing to save a non-finite checkpoint: {path}")
    if not _nested_tensors_are_finite(optimizer.state_dict()):
        raise FloatingPointError(f"Refusing to save a checkpoint with a non-finite optimizer: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_score": best_score,
            "config": config,
            "number_of_classes": number_of_classes,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train(
    config: dict[str, Any],
    work_dir: Path,
    run_dir: Path,
    resume: Path | None = None,
    pretrained: Path | None = None,
) -> None:
    from .io import read_json

    seed_everything(int(config["train"]["seed"]))
    device = _device_from_config(config)
    info = read_json(work_dir / "dataset_info.json")
    index = read_json(work_dir / "prepared_index.json")
    split = read_json(work_dir / "split.json")
    number_of_classes = int(info["number_of_classes"])
    train_records = _records_for_ids(index["labeled"], split["train"])
    val_records = _records_for_ids(index["labeled"], split["val"])
    unlabeled_records = index["unlabeled"]
    train_config = config["train"]
    patch_size = tuple(int(value) for value in train_config["patch_size"])
    steps = int(train_config["steps"])
    cache_size = int(config["data"]["cache_size"])
    batch_size = int(train_config["batch_size"])

    labeled_dataset = RandomPatchDataset(
        train_records, patch_size, float(train_config["foreground_probability"]), cache_size,
        length=max(steps * batch_size, 64), with_labels=True, seed=int(train_config["seed"]),
        rare_class_probability=float(train_config.get("rare_class_probability", 0.0)),
        artifact_patch_probability=float(train_config.get("artifact_patch_probability", 0.0)),
        artifact_case_sampling_weight=float(config["data"].get("artifact_case_sampling_weight", 1.0)),
    )
    val_dataset = RandomPatchDataset(
        val_records, patch_size, float(train_config["foreground_probability"]), cache_size,
        length=max(int(train_config["validation_patches"]), batch_size), with_labels=True, seed=int(train_config["seed"]) + 1,
        rare_class_probability=float(train_config.get("rare_class_probability", 0.0)),
        artifact_patch_probability=float(train_config.get("artifact_patch_probability", 0.0)),
    )
    labeled_loader = _loader(labeled_dataset, config)
    val_loader = _loader(val_dataset, config)
    semi_config = config["semi_supervised"]
    semi_enabled = bool(semi_config["enabled"]) and bool(unlabeled_records)
    unlabeled_loader = None
    if semi_enabled:
        unlabeled_dataset = RandomPatchDataset(
            unlabeled_records, patch_size, 0.0, cache_size,
            length=max(steps * batch_size, 64), with_labels=False, seed=int(train_config["seed"]) + 2,
            artifact_patch_probability=float(train_config.get("artifact_patch_probability", 0.0)),
            informative_unlabeled_probability=float(train_config.get("informative_unlabeled_probability", 0.0)),
            artifact_case_sampling_weight=float(config["data"].get("artifact_case_sampling_weight", 1.0)),
        )
        unlabeled_loader = _loader(unlabeled_dataset, config)

    if resume is not None and pretrained is not None:
        raise ValueError("Use either --resume or --pretrained, not both.")
    model = build_model(config, number_of_classes).to(device)
    if pretrained is not None:
        pretrain_checkpoint = torch.load(pretrained, map_location=device)
        if not _nested_tensors_are_finite(pretrain_checkpoint.get("model", {})):
            raise FloatingPointError(f"Pretraining checkpoint contains NaN/Inf: {pretrained}")
        model.load_state_dict(pretrain_checkpoint["model"], strict=True)
        print(f"Initialized segmentation training from self-supervised checkpoint {pretrained}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count / 1e6:.2f} M")
    ema = EMA(model, float(semi_config["ema_decay"]))
    optimizer = AdamW(model.parameters(), lr=float(train_config["learning_rate"]), weight_decay=float(train_config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(train_config["learning_rate"]) * 0.05)
    amp_enabled, amp_dtype = resolve_autocast(
        device, bool(train_config["amp"]), str(train_config.get("amp_dtype", "bfloat16"))
    )
    scaler_enabled = grad_scaler_enabled(amp_enabled, amp_dtype)
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    retry_fp32 = bool(train_config.get("fp32_retry_on_nonfinite", True))
    print(
        f"Training precision: {'autocast ' + str(amp_dtype).replace('torch.', '') if amp_enabled else 'float32'}; "
        f"GradScaler={'on' if scaler_enabled else 'off'}; FP32 retry={'on' if retry_fp32 else 'off'}"
    )
    class_weight_config = config["loss"].get("class_weights")
    class_weights = None
    if class_weight_config == "auto":
        class_weights = _automatic_class_weights(
            info, number_of_classes, config["loss"].get("class_weight_clip", [0.5, 5.0])
        ).to(device)
        print(f"Automatic class weights: {[round(float(value), 3) for value in class_weights.cpu()]}")
    elif class_weight_config is not None:
        if len(class_weight_config) != number_of_classes:
            raise ValueError("loss.class_weights must have one value per training class, including background.")
        class_weights = torch.tensor(class_weight_config, dtype=torch.float32, device=device)

    raw_to_train = {int(raw): int(train_id) for raw, train_id in info["raw_to_train"].items()}
    tubular_indices = {
        raw_to_train[raw]
        for raw in (int(value) for value in config["loss"].get("tubular_raw_labels", []))
        if raw in raw_to_train
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    start_step, best_score, best_step, stale_validations = 0, -1.0, 0, 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        for state_name in ("model", "ema", "optimizer"):
            if state_name in checkpoint and not _nested_tensors_are_finite(checkpoint[state_name]):
                raise FloatingPointError(
                    f"Cannot resume from {resume}: checkpoint state '{state_name}' contains NaN/Inf. "
                    "Use the last finite checkpoint (normally checkpoint_best.pt)."
                )
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler_enabled:
            scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))
        print(f"Resumed at step {start_step} from {resume}")

    start_semi = int(steps * float(semi_config["start_fraction"]))
    ramp_steps = max(1, int(steps * float(semi_config["ramp_fraction"])))
    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader) if unlabeled_loader is not None else None
    log_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    last_time = time.time()
    nonfinite_steps = 0
    consecutive_nonfinite_steps = 0
    consecutive_forward_nonfinite_steps = 0
    consecutive_gradient_overflows = 0
    skipped_semi_batches = 0
    fp32_forward_retries = 0
    nonfinite_patience = max(1, int(train_config.get("nonfinite_abort_patience", 3)))
    gradient_overflow_patience = max(
        nonfinite_patience, int(train_config.get("amp_gradient_overflow_patience", 20))
    )

    try:
        for step in tqdm(range(start_step, steps), initial=start_step, total=steps, desc="Training"):
            try:
                labeled_batch = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(labeled_loader)
                labeled_batch = next(labeled_iterator)
            labeled_batch = _to_device(labeled_batch, device)
            raw_image, raw_metal, target = labeled_batch["image"], labeled_batch["metal"], labeled_batch["label"]
            if not torch.isfinite(raw_image).all() or not torch.isfinite(raw_metal).all():
                raise FloatingPointError(f"Prepared labeled input contains NaN/Inf: {labeled_batch.get('id')}")
            strong_labeled = strong_intensity_augment(
                raw_image,
                float(config["augmentation"]["noise_std"]),
                config["augmentation"]["contrast_range"],
                config["augmentation"]["brightness_range"],
                config["augmentation"]["gamma_range"],
                float(config["augmentation"]["blur_probability"]),
                float(config["augmentation"]["low_resolution_probability"]),
            )
            aug_image, synthetic_mask = synthesize_metal_artifact(
                strong_labeled,
                float(config["augmentation"]["artifact_probability"]), int(config["augmentation"]["max_artifacts"]),
                config["augmentation"].get("streaks_per_artifact", [4, 9]),
            )
            artifact_mask = torch.maximum(raw_metal, synthetic_mask)
            optimizer.zero_grad(set_to_none=True)
            semi_skip_reason: str | None = None
            numerical_failure: str | None = None
            step_fp32_retries: list[str] = []
            with autocast_context(device, False, amp_dtype):
                labeled_output, used_fp32 = _stable_forward(
                    model, _network_input(aug_image, artifact_mask), device,
                    amp_enabled, amp_dtype, retry_fp32,
                )
                if used_fp32:
                    step_fp32_retries.append("labeled_student")
                supervised = segmentation_with_auxiliary(
                    labeled_output, target, float(config["loss"]["dice_weight"]), float(config["loss"]["ce_weight"]),
                    config["loss"]["auxiliary_weights"], class_weights,
                    foreground_weight=float(config["loss"].get("foreground_dice_weight", 0.0)),
                    boundary_weight=float(config["loss"].get("boundary_weight", 0.0)),
                    topology_weight=float(config["loss"].get("topology_weight", 0.0)),
                    topology_class_indices=sorted(tubular_indices),
                    topology_iterations=int(config["loss"].get("topology_iterations", 3)),
                )
                restoration = weighted_restoration_loss(
                    labeled_output["restored"], raw_image, synthetic_mask, float(config["loss"]["metal_restoration_multiplier"]),
                )
                total_loss = supervised + float(config["loss"]["restoration_weight"]) * restoration
                artifact_consistency = supervised.new_zeros(())
                if (
                    synthetic_mask.any()
                    and torch.rand((), device=device) < float(config["augmentation"].get("artifact_consistency_probability", 0.0))
                ):
                    with torch.no_grad():
                        clean_output, used_fp32 = _stable_forward(
                            ema.model, _network_input(raw_image, raw_metal), device,
                            amp_enabled, amp_dtype, retry_fp32,
                            return_restoration=False, return_auxiliary=False,
                        )
                        if used_fp32:
                            step_fp32_retries.append("artifact_teacher")
                        clean_logits = clean_output["logits"]
                        clean_is_finite = bool(torch.isfinite(clean_logits).all().item())
                        clean_probabilities = clean_logits.float().softmax(dim=1) if clean_is_finite else None
                    if clean_probabilities is not None:
                        artifact_consistency = artifact_invariance_loss(
                            labeled_output["logits"], clean_probabilities, synthetic_mask
                        )
                        total_loss = total_loss + float(config["augmentation"].get("artifact_consistency_weight", 0.0)) * artifact_consistency
                    else:
                        semi_skip_reason = "nonfinite_artifact_teacher"
                consistency = supervised.new_zeros(())
                pseudo = supervised.new_zeros(())
                pseudo_coverage = 0.0
                pseudo_foreground_fraction = 0.0
                ramp = _ramp(step, start_semi, ramp_steps) if semi_enabled else 0.0
                if semi_enabled and ramp > 0 and unlabeled_iterator is not None:
                    try:
                        unlabeled_batch = next(unlabeled_iterator)
                    except StopIteration:
                        unlabeled_iterator = iter(unlabeled_loader)
                        unlabeled_batch = next(unlabeled_iterator)
                    unlabeled_batch = _to_device(unlabeled_batch, device)
                    un_image, un_metal = unlabeled_batch["image"], unlabeled_batch["metal"]
                    if not torch.isfinite(un_image).all() or not torch.isfinite(un_metal).all():
                        raise FloatingPointError(f"Prepared unlabeled input contains NaN/Inf: {unlabeled_batch.get('id')}")
                    with torch.no_grad():
                        teacher_output, used_fp32 = _stable_forward(
                            ema.model, _network_input(un_image, un_metal), device,
                            amp_enabled, amp_dtype, retry_fp32,
                            return_restoration=False, return_auxiliary=False,
                        )
                        if used_fp32:
                            step_fp32_retries.append("pseudo_teacher")
                        teacher_logits = teacher_output["logits"]
                        teacher_is_finite = bool(torch.isfinite(teacher_logits).all().item())
                        if teacher_is_finite:
                            # Softmax outside FP16 prevents confident rare classes
                            # from underflowing to exact zero probabilities.
                            teacher_probabilities = teacher_logits.float().softmax(dim=1)
                            pseudo_labels = teacher_probabilities.argmax(dim=1)
                            confidence_mask = _pseudo_confidence_weights(
                                teacher_probabilities, pseudo_labels, semi_config, tubular_indices
                            )
                            active_pseudo = confidence_mask > 0
                            pseudo_coverage = float(active_pseudo.float().mean().cpu())
                            pseudo_foreground_fraction = float(
                                ((pseudo_labels > 0) & active_pseudo).sum().float().div(active_pseudo.sum().clamp_min(1)).cpu()
                            )
                        else:
                            active_pseudo = torch.zeros_like(teacher_logits[:, 0], dtype=torch.bool)
                            semi_skip_reason = "nonfinite_teacher_logits"
                    if active_pseudo.any():
                        strong_unlabeled = strong_intensity_augment(
                            un_image,
                            float(config["augmentation"]["noise_std"]),
                            config["augmentation"]["contrast_range"],
                            config["augmentation"]["brightness_range"],
                            config["augmentation"]["gamma_range"],
                            float(config["augmentation"]["blur_probability"]),
                            float(config["augmentation"]["low_resolution_probability"]),
                        )
                        strong_image, strong_synthetic_mask = synthesize_metal_artifact(
                            strong_unlabeled,
                            float(config["augmentation"]["artifact_probability"]), int(config["augmentation"]["max_artifacts"]),
                            config["augmentation"].get("streaks_per_artifact", [4, 9]),
                        )
                        student_output, used_fp32 = _stable_forward(
                            model, _network_input(strong_image, torch.maximum(un_metal, strong_synthetic_mask)), device,
                            amp_enabled, amp_dtype, retry_fp32,
                            perturb=True,
                            perturb_probability=float(semi_config["feature_perturb_probability"]),
                            keep_range=semi_config["feature_keep_range"],
                            noise_scale=float(semi_config["feature_noise"]),
                            return_restoration=False,
                            return_auxiliary=False,
                        )
                        if used_fp32:
                            step_fp32_retries.append("unlabeled_student")
                        if torch.isfinite(student_output["logits"]).all():
                            consistency = masked_consistency_loss(student_output["logits"], teacher_probabilities, confidence_mask)
                            pseudo = masked_pseudo_label_loss(student_output["logits"], pseudo_labels, confidence_mask)
                            total_loss = total_loss + ramp * (
                                float(semi_config["consistency_weight"]) * consistency
                                + float(semi_config["pseudo_weight"]) * pseudo
                            )
                        else:
                            semi_skip_reason = "nonfinite_student_logits"
                    elif semi_skip_reason is None:
                        semi_skip_reason = "no_reliable_foreground_pseudo_labels"

            optimizer_step_succeeded = False
            grad_norm_value: float | None = None
            if torch.isfinite(total_loss).all():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["grad_clip_norm"]), error_if_nonfinite=False
                )
                grad_norm_value = float(grad_norm.detach().cpu())
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    ema.update(model)
                    optimizer_step_succeeded = True
                else:
                    numerical_failure = "nonfinite_gradient_norm"
                    optimizer.zero_grad(set_to_none=True)
                    # unscale_ has populated GradScaler's Inf/NaN tracker.
                    scaler.update()
            else:
                loss_components = {
                    "supervised": supervised,
                    "restoration": restoration,
                    "artifact_consistency": artifact_consistency,
                    "consistency": consistency,
                    "pseudo": pseudo,
                }
                bad_components = [
                    name for name, value in loss_components.items()
                    if not bool(torch.isfinite(value.detach()).all().item())
                ]
                numerical_failure = "nonfinite_total_loss:" + (",".join(bad_components) or "unknown")
                optimizer.zero_grad(set_to_none=True)

            if optimizer_step_succeeded:
                consecutive_nonfinite_steps = 0
                consecutive_forward_nonfinite_steps = 0
                consecutive_gradient_overflows = 0
            else:
                nonfinite_steps += 1
                consecutive_nonfinite_steps += 1
                if numerical_failure == "nonfinite_gradient_norm":
                    consecutive_gradient_overflows += 1
                    consecutive_forward_nonfinite_steps = 0
                    abort_threshold = gradient_overflow_patience
                else:
                    consecutive_forward_nonfinite_steps += 1
                    consecutive_gradient_overflows = 0
                    abort_threshold = nonfinite_patience
                warning_record = {
                    "step": step + 1,
                    "loss": None,
                    "event": "optimizer_step_skipped",
                    "reason": numerical_failure,
                    "consecutive_nonfinite_steps": consecutive_nonfinite_steps,
                    "consecutive_forward_nonfinite_steps": consecutive_forward_nonfinite_steps,
                    "consecutive_gradient_overflows": consecutive_gradient_overflows,
                    "gradient_norm": grad_norm_value,
                    "amp_scale": float(scaler.get_scale()),
                    "precision": str(amp_dtype).replace("torch.", "") if amp_enabled else "float32",
                    "fp32_retry_branches": step_fp32_retries,
                }
                log_file.write(json.dumps(warning_record) + "\n")
                log_file.flush()
                active_failure_count = (
                    consecutive_gradient_overflows
                    if numerical_failure == "nonfinite_gradient_norm"
                    else consecutive_forward_nonfinite_steps
                )
                if active_failure_count >= abort_threshold:
                    if _module_is_finite(model) and _module_is_finite(ema.model):
                        _save_checkpoint(
                            run_dir / "checkpoint_recovery.pt", model, ema, optimizer, scheduler,
                            scaler, step, best_score, config, number_of_classes,
                        )
                    raise FloatingPointError(
                        f"Aborting after {active_failure_count} consecutive '{numerical_failure}' updates at step {step + 1}. "
                        "The last finite state was written to checkpoint_recovery.pt when possible."
                    )

            if semi_skip_reason is not None:
                skipped_semi_batches += 1
            fp32_forward_retries += len(step_fp32_retries)

            if (step + 1) % 50 == 0:
                elapsed = max(time.time() - last_time, 1e-6)
                last_time = time.time()
                record = {
                    "step": step + 1, "loss": float(total_loss.detach().cpu()), "supervised": float(supervised.detach().cpu()),
                    "restoration": float(restoration.detach().cpu()), "consistency": float(consistency.detach().cpu()),
                    "pseudo": float(pseudo.detach().cpu()), "artifact_consistency": float(artifact_consistency.detach().cpu()),
                    "pseudo_coverage": pseudo_coverage, "pseudo_foreground_fraction": pseudo_foreground_fraction,
                    "semi_ramp": ramp, "semi_skip_reason": semi_skip_reason,
                    "skipped_semi_batches": skipped_semi_batches,
                    "optimizer_step_succeeded": optimizer_step_succeeded,
                    "nonfinite_steps": nonfinite_steps, "gradient_norm": grad_norm_value,
                    "precision": str(amp_dtype).replace("torch.", "") if amp_enabled else "float32",
                    "fp32_retry_branches": step_fp32_retries,
                    "fp32_forward_retries": fp32_forward_retries,
                    "learning_rate": optimizer.param_groups[0]["lr"], "steps_per_second": 50 / elapsed,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

            if (step + 1) % int(train_config["validate_every"]) == 0 or step + 1 == steps:
                if not _module_is_finite(ema.model):
                    raise FloatingPointError("EMA model became non-finite before validation; stopping immediately.")
                patch_score = validation_dice(ema.model, val_loader, device, amp_enabled, amp_dtype)
                full_interval = int(train_config.get("full_volume_validate_every", 0))
                run_full = full_interval > 0 and ((step + 1) % full_interval == 0 or step + 1 == steps)
                full_metrics = full_volume_validation(ema.model, val_records, device, config) if run_full else None
                score = full_metrics["selection_score"] if full_metrics is not None else patch_score
                validation_record = {
                    "step": step + 1,
                    "validation_macro_dice": patch_score,
                    "full_volume": full_metrics,
                }
                log_file.write(json.dumps(validation_record) + "\n")
                log_file.flush()
                print(f"\nStep {step + 1}: patch macro Dice = {patch_score:.4f}")
                if full_metrics is not None:
                    print(f"Full-volume official-score proxy = {score:.4f}: {full_metrics}")
                if full_metrics is not None or full_interval <= 0:
                    if score > best_score:
                        best_score, best_step, stale_validations = score, step + 1, 0
                        _save_checkpoint(run_dir / "checkpoint_best.pt", model, ema, optimizer, scheduler, scaler, step, best_score, config, number_of_classes)
                    else:
                        stale_validations += 1
                    if stale_validations >= int(train_config["early_stop_patience"]):
                        print(f"Early stopping at step {step + 1}; best full-volume score {best_score:.4f} at {best_step}.")
                        break
            if (step + 1) % int(train_config["checkpoint_every"]) == 0:
                _save_checkpoint(run_dir / "checkpoint_last.pt", model, ema, optimizer, scheduler, scaler, step, best_score, config, number_of_classes)
        _save_checkpoint(run_dir / "checkpoint_last.pt", model, ema, optimizer, scheduler, scaler, step, best_score, config, number_of_classes)
    finally:
        log_file.close()

    print(f"Completed. Best full-volume validation proxy: {best_score:.4f}; checkpoint: {run_dir / 'checkpoint_best.pt'}")


def pretrain(config: dict[str, Any], work_dir: Path, run_dir: Path, resume: Path | None = None) -> None:
    """Self-supervised disruptive autoencoder pretraining on labelled + unlabelled CBCT."""
    from .io import read_json

    seed_everything(int(config["train"]["seed"]))
    device = _device_from_config(config)
    pretrain_config = config["pretrain"]
    index = read_json(work_dir / "prepared_index.json")
    info = read_json(work_dir / "dataset_info.json")
    records = list(index["labeled"]) + list(index["unlabeled"])
    if not records:
        raise ValueError("No prepared labelled or unlabelled cases are available for pretraining.")
    steps = int(pretrain_config["steps"])
    batch_size = int(config["train"]["batch_size"])
    dataset = RandomPatchDataset(
        records=records,
        patch_size=tuple(int(value) for value in config["train"]["patch_size"]),
        foreground_probability=0.0,
        cache_size=int(config["data"]["cache_size"]),
        length=max(steps * batch_size, 64),
        with_labels=False,
        seed=int(config["train"]["seed"]) + 101,
        artifact_patch_probability=float(config["train"].get("artifact_patch_probability", 0.0)),
        informative_unlabeled_probability=float(config["train"].get("informative_unlabeled_probability", 0.0)),
        artifact_case_sampling_weight=float(config["data"].get("artifact_case_sampling_weight", 1.0)),
    )
    loader = _loader(dataset, config)
    model = build_model(config, int(info["number_of_classes"])).to(device)
    optimizer = AdamW(model.parameters(), lr=float(pretrain_config["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=float(pretrain_config["learning_rate"]) * 0.05)
    amp_enabled, amp_dtype = resolve_autocast(
        device, bool(config["train"]["amp"]), str(config["train"].get("amp_dtype", "bfloat16"))
    )
    scaler_enabled = grad_scaler_enabled(amp_enabled, amp_dtype)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f} M")
    print(f"Pretraining precision: {str(amp_dtype).replace('torch.', '') if amp_enabled else 'float32'}")
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    start_step = 0
    best_loss = float("inf")
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        for state_name in ("model", "optimizer"):
            if state_name in checkpoint and not _nested_tensors_are_finite(checkpoint[state_name]):
                raise FloatingPointError(f"Cannot resume non-finite pretraining state '{state_name}': {resume}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler_enabled:
            scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
    iterator = iter(loader)
    log_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    try:
        for step in tqdm(range(start_step, steps), initial=start_step, total=steps, desc="DAE pretraining"):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _to_device(batch, device)
            target = batch["image"]
            corrupted, corruption_mask = disruptive_autoencoder_corruption(
                target,
                float(config["augmentation"]["artifact_probability"]),
                int(config["augmentation"]["max_artifacts"]),
                float(pretrain_config["cube_mask_probability"]),
                float(pretrain_config["downsample_probability"]),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output, used_fp32 = _stable_forward(
                model,
                _network_input(corrupted, torch.maximum(batch["metal"], corruption_mask)),
                device,
                amp_enabled,
                amp_dtype,
                retry_fp32=True,
            )
            with autocast_context(device, False, amp_dtype):
                loss = weighted_restoration_loss(
                    output["restored"], target, corruption_mask,
                    float(config["loss"]["metal_restoration_multiplier"]),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite pretraining restoration loss at step {step + 1}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                raise FloatingPointError(f"Non-finite pretraining gradient norm at step {step + 1}.")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            value = float(loss.detach().cpu())
            log_file.write(json.dumps({
                "step": step + 1,
                "restoration_loss": value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "precision": str(amp_dtype).replace("torch.", "") if amp_enabled else "float32",
                "fp32_retry": used_fp32,
            }) + "\n")
            if value < best_loss:
                best_loss = value
                torch.save({"model": model.state_dict(), "step": step, "best_loss": best_loss, "config": config, "number_of_classes": int(info["number_of_classes"])}, run_dir / "pretrain_best.pt")
            if (step + 1) % int(pretrain_config["checkpoint_every"]) == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "best_loss": best_loss, "config": config, "number_of_classes": int(info["number_of_classes"])}, run_dir / "pretrain_last.pt")
    finally:
        log_file.close()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "best_loss": best_loss, "config": config, "number_of_classes": int(info["number_of_classes"])}, run_dir / "pretrain_last.pt")
    print(f"Completed self-supervised pretraining. Best restoration loss: {best_loss:.6f}; checkpoint: {run_dir / 'pretrain_best.pt'}")
