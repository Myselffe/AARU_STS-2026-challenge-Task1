from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


NIFTI_SUFFIXES = (".nii.gz", ".nii")


def case_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Expected a NIfTI file, got '{path}'.")


def nifti_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and any(path.name.endswith(suffix) for suffix in NIFTI_SUFFIXES)
    )


def read_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], nib.Nifti1Header]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Only 3-D CBCT NIfTI files are supported: {path} has shape {data.shape}.")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data, image.affine, spacing, image.header.copy()


def read_nifti_metadata(
    path: str | Path,
) -> tuple[tuple[int, int, int], np.ndarray, tuple[float, float, float], nib.Nifti1Header]:
    """Read geometry without materializing the complete voxel array."""
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Only 3-D CBCT NIfTI files are supported: {path} has shape {image.shape}.")
    shape = tuple(int(value) for value in image.shape)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return shape, image.affine, spacing, image.header.copy()


def resize_to_shape(array: np.ndarray, target_shape: Iterable[int], order: int) -> np.ndarray:
    target = tuple(int(value) for value in target_shape)
    factors = [target[i] / array.shape[i] for i in range(3)]
    result = zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)
    # scipy may differ by one voxel due to floating point rounding.
    slices = tuple(slice(0, min(result.shape[i], target[i])) for i in range(3))
    result = result[slices]
    pad = [(0, max(0, target[i] - result.shape[i])) for i in range(3)]
    if any(after for _, after in pad):
        result = np.pad(result, pad, mode="edge")
    return result


def resample_spacing(
    array: np.ndarray,
    source_spacing: Iterable[float],
    target_spacing: Iterable[float] | None,
    order: int,
) -> np.ndarray:
    if target_spacing is None:
        return array
    source = np.asarray(tuple(source_spacing), dtype=np.float64)
    target = np.asarray(tuple(target_spacing), dtype=np.float64)
    if np.any(target <= 0):
        raise ValueError(f"Invalid target spacing: {target_spacing}")
    target_shape = np.maximum(1, np.rint(np.asarray(array.shape) * source / target)).astype(int)
    if tuple(target_shape) == tuple(array.shape):
        return array
    return resize_to_shape(array, target_shape, order)


def normalize_cbct(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Robust foreground Z-score normalization with a conservative metal seed."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("The CBCT contains no finite voxel values.")
    low, high = np.percentile(finite, [0.5, 99.5])
    metal_cutoff = np.percentile(finite, 99.9)
    clipped = np.clip(np.nan_to_num(image, nan=low, posinf=high, neginf=low), low, high)
    foreground_cutoff = np.percentile(finite, 20.0)
    foreground = clipped[clipped > foreground_cutoff]
    statistics = foreground if foreground.size >= 1024 else clipped.reshape(-1)
    mean = float(statistics.mean())
    std = max(float(statistics.std()), 1e-6)
    normalized = (clipped - mean) / std
    metal = (image >= metal_cutoff).astype(np.float32)
    return normalized.astype(np.float32), metal, {
        "low": float(low), "high": float(high), "mean": mean, "std": std, "metal_cutoff": float(metal_cutoff)
    }


def denormalize_cbct(normalized: np.ndarray, normalization: dict[str, float]) -> np.ndarray:
    restored = normalized * normalization["std"] + normalization["mean"]
    return np.clip(restored, normalization["low"], normalization["high"]).astype(np.float32)


def discover_dataset(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    labeled_images = {case_id_from_path(path): path for path in nifti_files(root / "Train-Labeled" / "images")}
    labeled_labels = {case_id_from_path(path): path for path in nifti_files(root / "Train-Labeled" / "labels")}
    missing_labels = sorted(set(labeled_images) - set(labeled_labels))
    orphan_labels = sorted(set(labeled_labels) - set(labeled_images))
    if missing_labels or orphan_labels:
        raise ValueError(
            "Train-Labeled images/labels do not match. "
            f"Missing labels: {missing_labels[:5]}; orphan labels: {orphan_labels[:5]}"
        )
    if not labeled_images:
        raise FileNotFoundError(f"No labelled NIfTI files found in {root / 'Train-Labeled' / 'images'}")
    unlabeled = {case_id_from_path(path): path for path in nifti_files(root / "Train-Unlabeled")}
    validation = {case_id_from_path(path): path for path in nifti_files(root / "Validation" / "images")}
    return {
        "root": str(root.resolve()),
        "labeled": [{"id": key, "image": str(labeled_images[key]), "label": str(labeled_labels[key])} for key in sorted(labeled_images)],
        "unlabeled": [{"id": key, "image": str(unlabeled[key])} for key in sorted(unlabeled)],
        "validation": [{"id": key, "image": str(validation[key])} for key in sorted(validation)],
    }


def collect_label_mapping(records: list[dict[str, str]]) -> tuple[dict[int, int], dict[int, int]]:
    labels: set[int] = {0}
    for record in records:
        array, _, _, _ = read_nifti(record["label"])
        if not np.allclose(array, np.rint(array)):
            raise ValueError(f"Label map is not integer-valued: {record['label']}")
        labels.update(int(value) for value in np.unique(array))
    ordered = [0] + sorted(value for value in labels if value != 0)
    raw_to_train = {raw: index for index, raw in enumerate(ordered)}
    train_to_raw = {index: raw for raw, index in raw_to_train.items()}
    return raw_to_train, train_to_raw


def remap_labels(labels: np.ndarray, raw_to_train: dict[int, int]) -> np.ndarray:
    source = np.rint(labels).astype(np.int64)
    result = np.zeros(source.shape, dtype=np.int16 if len(raw_to_train) < np.iinfo(np.int16).max else np.int32)
    for raw, train in raw_to_train.items():
        result[source == raw] = train
    unseen = set(np.unique(source).tolist()) - set(raw_to_train)
    if unseen:
        raise ValueError(f"Encountered labels not present in the mapping: {sorted(unseen)}")
    return result


def median_spacing(records: list[dict[str, str]]) -> tuple[float, float, float]:
    spacings = []
    for record in records:
        # Spacing lives in the NIfTI header. Materializing every compressed CBCT
        # here made an already prepared run appear frozen before tqdm started.
        _, _, spacing, _ = read_nifti_metadata(record["image"])
        spacings.append(spacing)
    return tuple(float(value) for value in np.median(np.asarray(spacings), axis=0))


def write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_prepared_case(
    record: dict[str, str],
    output_file: str | Path,
    target_spacing: tuple[float, float, float] | None,
    raw_to_train: dict[int, int] | None,
) -> dict[str, Any]:
    image, affine, spacing, _ = read_nifti(record["image"])
    original_shape = tuple(int(value) for value in image.shape)
    image = resample_spacing(image, spacing, target_spacing, order=3)
    normalized, metal, normalization = normalize_cbct(image)
    payload: dict[str, Any] = {
        "image": normalized.astype(np.float16),
        "metal": metal.astype(np.uint8),
        "original_shape": np.asarray(original_shape, dtype=np.int32),
        "original_spacing": np.asarray(spacing, dtype=np.float32),
        "prepared_spacing": np.asarray(target_spacing or spacing, dtype=np.float32),
        "affine": affine.astype(np.float64),
        "normalization": np.asarray([normalization[key] for key in ("low", "high", "mean", "std", "metal_cutoff")], dtype=np.float32),
    }
    if "label" in record:
        label, _, label_spacing, _ = read_nifti(record["label"])
        if tuple(label.shape) != original_shape or not np.allclose(label_spacing, spacing):
            raise ValueError(f"Image and label geometry differs for case '{record['id']}'.")
        label = resample_spacing(label, spacing, target_spacing, order=0)
        payload["label"] = remap_labels(label, raw_to_train or {0: 0})
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **payload)
    return {"id": record["id"], "file": str(output_file.resolve()), "has_label": "label" in record}


def load_prepared_case(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            result = {key: data[key] for key in data.files}
    except Exception as error:
        raise ValueError(f"Cannot read prepared case '{path}': {error}") from error
    required = {"image", "metal"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Prepared case '{path}' is missing arrays: {sorted(missing)}")
    result["image"] = result["image"].astype(np.float32)
    result["metal"] = result["metal"].astype(np.float32)
    if result["image"].shape != result["metal"].shape or result["image"].ndim != 3:
        raise ValueError(
            f"Prepared image/metal geometry mismatch in '{path}': "
            f"image={result['image'].shape}, metal={result['metal'].shape}"
        )
    if not np.isfinite(result["image"]).all() or not np.isfinite(result["metal"]).all():
        raise ValueError(f"Prepared case '{path}' contains NaN/Inf image or metal values.")
    if "label" in result:
        result["label"] = result["label"].astype(np.int64)
        if result["label"].shape != result["image"].shape:
            raise ValueError(
                f"Prepared image/label geometry mismatch in '{path}': "
                f"image={result['image'].shape}, label={result['label'].shape}"
            )
    return result


def make_split(ids: list[str], validation_fraction: float, seed: int) -> dict[str, list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    if len(ids) < 3:
        raise ValueError("At least three labelled cases are required for an internal train/validation split.")
    rng = np.random.default_rng(seed)
    val_count = min(len(ids) - 1, max(1, int(round(len(ids) * validation_fraction))))
    artifact = [value for value in sorted(ids) if "with-artifacts" in value.lower().replace("_", "-")]
    normal = [value for value in sorted(ids) if value not in artifact]
    selected: list[str] = []
    remaining: list[str] = []
    for group in (artifact, normal):
        shuffled = np.asarray(group, dtype=object)
        rng.shuffle(shuffled)
        if len(group) >= 2:
            count = min(len(group) - 1, max(1, int(round(len(group) * validation_fraction))))
        else:
            count = 0
        selected.extend(shuffled[:count].tolist())
        remaining.extend(shuffled[count:].tolist())
    rng.shuffle(remaining)
    while len(selected) < val_count and remaining:
        selected.append(remaining.pop())
    while len(selected) > val_count:
        remaining.append(selected.pop())
    return {"train": sorted(remaining), "val": sorted(selected)}


def class_histogram(prepared_records: list[dict[str, Any]], number_of_classes: int) -> Counter[int]:
    histogram: Counter[int] = Counter()
    for record in prepared_records:
        if not record["has_label"]:
            continue
        case = load_prepared_case(record["file"])
        values, counts = np.unique(case["label"], return_counts=True)
        histogram.update({int(value): int(count) for value, count in zip(values, counts)})
    for index in range(number_of_classes):
        histogram.setdefault(index, 0)
    return histogram
