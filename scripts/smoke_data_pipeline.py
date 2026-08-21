"""Create temporary NIfTI data and exercise preparation, sampling and inference.

This is an implementation check only; it does not alter the user's Dataset folder.
"""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.dataset import RandomPatchDataset
from cbct_ssl.engine import pretrain, train
from cbct_ssl.inference import predict_case
from cbct_ssl.io import collect_label_mapping, discover_dataset, load_prepared_case, save_prepared_case, write_json
from cbct_ssl.model import ArtifactAwareResUNet3D


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "Dataset"
        rng = np.random.default_rng(7)
        for case_index in range(3):
            image = rng.normal(size=(16, 32, 32)).astype(np.float32)
            image[5:9, 12:18, 12:18] = 8.0
            target = np.zeros_like(image, dtype=np.int16)
            target[4:12, 8:24, 8:24] = 11
            target[7:10, 13:18, 13:18] = 101
            case_id = f"case_{case_index:03d}"
            _write(root / "Train-Labeled" / "images" / f"{case_id}.nii.gz", image)
            _write(root / "Train-Labeled" / "labels" / f"{case_id}.nii.gz", target)
        _write(root / "Train-Unlabeled" / "case_100.nii.gz", rng.normal(size=(16, 32, 32)).astype(np.float32))
        _write(root / "Validation" / "images" / "case_200.nii.gz", rng.normal(size=(16, 32, 32)).astype(np.float32))

        dataset = discover_dataset(root)
        mapping, inverse = collect_label_mapping(dataset["labeled"])
        prepared_records = []
        for record in dataset["labeled"]:
            prepared_records.append(save_prepared_case(record, root / "prepared" / f"{record['id']}.npz", (1.0, 1.0, 1.0), mapping))
        prepared_unlabeled = [
            save_prepared_case(record, root / "prepared" / f"{record['id']}.npz", (1.0, 1.0, 1.0), None)
            for record in dataset["unlabeled"]
        ]
        prepared_validation = [
            save_prepared_case(record, root / "prepared" / f"{record['id']}.npz", (1.0, 1.0, 1.0), None)
            for record in dataset["validation"]
        ]
        loaded = load_prepared_case(prepared_records[0]["file"])
        assert set(np.unique(loaded["label"])) <= set(mapping.values())
        patches = RandomPatchDataset(prepared_records, (16, 32, 32), 1.0, 1, 2, True, 1)
        patch = patches[0]
        assert patch["image"].shape == (1, 16, 32, 32)
        assert patch["label"].shape == (16, 32, 32)

        model = ArtifactAwareResUNet3D(3, len(mapping), channels=(8, 16, 24, 32, 48), residual_blocks=(1, 1, 1, 1, 1)).eval()
        output_file = root / "prediction" / "case_200.nii.gz"
        predict_case(
            model, root / "Validation" / "images" / "case_200.nii.gz", output_file,
            {str(key): value for key, value in inverse.items()}, (1.0, 1.0, 1.0), (16, 32, 32), 0.5, False, 0,
            prepared_path=prepared_validation[0]["file"],
        )
        assert output_file.exists()

        work_dir = root / "work"
        write_json(
            {"labeled": prepared_records, "unlabeled": prepared_unlabeled, "validation": prepared_validation},
            work_dir / "prepared_index.json",
        )
        write_json({"train": [record["id"] for record in prepared_records[:2]], "val": [prepared_records[2]["id"]]}, work_dir / "split.json")
        write_json(
            {
                "number_of_classes": len(mapping),
                "raw_to_train": {str(key): value for key, value in mapping.items()},
                "train_to_raw": {str(key): value for key, value in inverse.items()},
                "class_voxels": {str(index): 1 for index in range(len(mapping))},
            },
            work_dir / "dataset_info.json",
        )
        config = {
            "data": {"work_dir": str(work_dir), "cache_size": 1, "artifact_case_sampling_weight": 2.0},
            "model": {"in_channels": 3, "channels": [8, 16, 24, 32, 48], "residual_blocks": [1, 1, 1, 1, 1], "deep_supervision": True, "axial_context": True, "boundary_head": True},
            "pretrain": {"steps": 1, "learning_rate": 0.0002, "checkpoint_every": 1, "cube_mask_probability": 0.5, "downsample_probability": 0.5},
            "train": {
                "device": "cpu", "seed": 7, "patch_size": [16, 32, 32], "batch_size": 1, "num_workers": 0,
                "steps": 1, "validate_every": 1, "full_volume_validate_every": 1, "checkpoint_every": 1, "learning_rate": 0.0002,
                "weight_decay": 0.0, "grad_clip_norm": 12.0, "amp": False, "foreground_probability": 1.0,
                "rare_class_probability": 1.0, "artifact_patch_probability": 0.0, "informative_unlabeled_probability": 1.0,
                "validation_patches": 1, "early_stop_patience": 2,
            },
            "loss": {"dice_weight": 1.0, "ce_weight": 1.0, "foreground_dice_weight": 0.1, "boundary_weight": 0.1, "topology_weight": 0.1, "topology_iterations": 2, "tubular_raw_labels": [2], "auxiliary_weights": [0.25, 0.125], "restoration_weight": 0.15, "metal_restoration_multiplier": 3.0, "class_weights": None},
            "semi_supervised": {"enabled": True, "start_fraction": 0.0, "ramp_fraction": 0.0, "consistency_weight": 1.0, "pseudo_weight": 0.15, "background_threshold": 0.0, "foreground_threshold": 0.0, "tubular_threshold": 0.0, "confidence_power": 1.0, "max_normalized_entropy": 1.0, "foreground_unsupervised_weight": 1.0, "ema_decay": 0.99, "feature_perturb_probability": 0.5, "feature_keep_range": [0.7, 0.9], "feature_noise": 0.2},
            "augmentation": {"artifact_probability": 0.5, "max_artifacts": 1, "streaks_per_artifact": [1, 2], "artifact_consistency_weight": 0.1, "artifact_consistency_probability": 1.0, "noise_std": 0.01, "contrast_range": [0.9, 1.1], "brightness_range": [-0.05, 0.05], "gamma_range": [0.9, 1.1], "blur_probability": 0.0, "low_resolution_probability": 0.0},
            "inference": {"patch_size": [16, 32, 32], "overlap": 0.5, "amp": False, "min_component_voxels": 0, "tta_axes": []},
        }
        pretrain(config, work_dir, work_dir / "pretrain" / "smoke")
        train(config, work_dir, work_dir / "runs" / "smoke", pretrained=work_dir / "pretrain" / "smoke" / "pretrain_best.pt")
        assert (work_dir / "runs" / "smoke" / "checkpoint_best.pt").exists()
        print("NIfTI preparation, patch sampling, one semi-supervised training step and sliding-window inference smoke test passed.")


if __name__ == "__main__":
    main()
