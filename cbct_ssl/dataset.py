from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_prepared_case


class VolumeCache:
    def __init__(self, size: int) -> None:
        self.size = max(0, int(size))
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: str) -> dict[str, Any]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        item = load_prepared_case(path)
        if self.size > 0:
            self.cache[path] = item
            self.cache.move_to_end(path)
            while len(self.cache) > self.size:
                self.cache.popitem(last=False)
        return item


def _crop_with_padding(array: np.ndarray, start: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    slices = []
    pads = []
    for axis, size in enumerate(patch_size):
        lower = start[axis]
        upper = lower + size
        slices.append(slice(max(0, lower), min(array.shape[axis], upper)))
        pads.append((max(0, -lower), max(0, upper - array.shape[axis])))
    cropped = array[tuple(slices)]
    if any(sum(pair) for pair in pads):
        mode = "constant" if np.issubdtype(array.dtype, np.integer) else "edge"
        cropped = np.pad(cropped, pads, mode=mode)
    return cropped


class RandomPatchDataset(Dataset):
    """Artifact-aware random patch sampler with rare-class oversampling."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        patch_size: tuple[int, int, int],
        foreground_probability: float,
        cache_size: int,
        length: int,
        with_labels: bool,
        seed: int,
        rare_class_probability: float = 0.0,
        artifact_patch_probability: float = 0.0,
        informative_unlabeled_probability: float = 0.0,
        artifact_case_sampling_weight: float = 1.0,
    ) -> None:
        if not records:
            raise ValueError("The dataset received no records.")
        self.records = records
        self.patch_size = tuple(int(value) for value in patch_size)
        self.foreground_probability = foreground_probability
        self.cache = VolumeCache(cache_size)
        self.length = int(length)
        self.with_labels = with_labels
        self.seed = int(seed)
        self.rare_class_probability = float(rare_class_probability)
        self.artifact_patch_probability = float(artifact_patch_probability)
        self.informative_unlabeled_probability = float(informative_unlabeled_probability)
        self.record_probabilities = np.asarray([
            float(artifact_case_sampling_weight)
            if "with-artifacts" in str(record["id"]).lower().replace("_", "-")
            else 1.0
            for record in records
        ], dtype=np.float64)
        self.record_probabilities /= self.record_probabilities.sum()

    def __len__(self) -> int:
        return self.length

    def _start_from_center(self, center: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
        jitter = np.asarray([rng.integers(-size // 4, size // 4 + 1) for size in self.patch_size])
        return tuple(int(center[axis] + jitter[axis] - self.patch_size[axis] // 2) for axis in range(3))

    def _random_start(self, rng: np.random.Generator, case: dict[str, Any]) -> tuple[int, int, int]:
        image = case["image"]
        if rng.random() < self.artifact_patch_probability:
            if "_metal_points" not in case:
                case["_metal_points"] = np.argwhere(case["metal"] > 0)
            points = case["_metal_points"]
            center = points[int(rng.integers(0, len(points)))] if len(points) else None
            if center is not None:
                return self._start_from_center(center, rng)
        if self.with_labels and rng.random() < self.foreground_probability:
            label = case["label"]
            if "_label_summary" not in case:
                case["_label_summary"] = np.unique(label[label > 0], return_counts=True)
            present, counts = case["_label_summary"]
            if present.size:
                if rng.random() < self.rare_class_probability:
                    probabilities = 1.0 / np.sqrt(counts.astype(np.float64).clip(min=1))
                    probabilities /= probabilities.sum()
                    selected = int(rng.choice(present, p=probabilities))
                else:
                    selected = int(rng.choice(present))
                points_key = f"_label_points_{selected}"
                if points_key not in case:
                    case[points_key] = np.argwhere(label == selected)
                candidates = case[points_key]
                center = candidates[int(rng.integers(0, len(candidates)))]
                return self._start_from_center(center, rng)
        if not self.with_labels and rng.random() < self.informative_unlabeled_probability:
            if "_informative_threshold" not in case:
                case["_informative_threshold"] = float(np.percentile(image, 70.0))
            threshold = case["_informative_threshold"]
            for _ in range(64):
                center = np.asarray([rng.integers(0, size) for size in image.shape])
                if image[tuple(center)] >= threshold:
                    return self._start_from_center(center, rng)
        return tuple(int(rng.integers(-size // 4, max(1, image.shape[axis] - size + size // 4 + 1))) for axis, size in enumerate(self.patch_size))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + index * 7919 + worker_id * 104729)
        record = self.records[int(rng.choice(len(self.records), p=self.record_probabilities))]
        case = self.cache.get(record["file"])
        start = self._random_start(rng, case)
        image = _crop_with_padding(case["image"], start, self.patch_size)
        metal = _crop_with_padding(case["metal"], start, self.patch_size)
        output: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(image[None].copy()),
            "metal": torch.from_numpy(metal[None].copy()),
            "id": record["id"],
        }
        if self.with_labels:
            output["label"] = torch.from_numpy(_crop_with_padding(case["label"], start, self.patch_size).copy()).long()
        return output
