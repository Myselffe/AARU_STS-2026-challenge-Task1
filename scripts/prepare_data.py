from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.config import load_config, with_overrides
from cbct_ssl.io import (
    class_histogram,
    collect_label_mapping,
    discover_dataset,
    make_split,
    median_spacing,
    save_prepared_case,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Task 1 directory into self-contained training archives.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing prepared .npz files.")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. train.patch_size='[80,160,160]'.")
    return parser.parse_args()


def resolve_target_spacing(value, labeled_records):
    if value is None:
        return None
    if value == "auto":
        return median_spacing(labeled_records)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(item) for item in value)
    raise ValueError("data.target_spacing must be auto, null, or [sx, sy, sz].")


def main() -> None:
    args = parse_args()
    config = with_overrides(load_config(args.config), args.set)
    work_dir = Path(config["data"]["work_dir"]).resolve()
    prepared_dir = work_dir / "prepared"
    dataset = discover_dataset(config["data"]["dataset_root"])
    print(f"Discovered {len(dataset['labeled'])} labeled, {len(dataset['unlabeled'])} unlabeled and {len(dataset['validation'])} validation cases.", flush=True)
    print("Scanning labeled masks to build the class mapping...", flush=True)
    raw_to_train, train_to_raw = collect_label_mapping(dataset["labeled"])
    print("Reading NIfTI headers to determine target spacing...", flush=True)
    spacing = resolve_target_spacing(config["data"]["target_spacing"], dataset["labeled"])

    output_records = {"labeled": [], "unlabeled": [], "validation": []}
    for group in ("labeled", "unlabeled", "validation"):
        for record in tqdm(dataset[group], desc=f"Preparing {group}"):
            output_file = prepared_dir / group / f"{record['id']}.npz"
            if args.overwrite or not output_file.exists():
                prepared_record = save_prepared_case(record, output_file, spacing, raw_to_train if group == "labeled" else None)
            else:
                prepared_record = {"id": record["id"], "file": str(output_file.resolve()), "has_label": group == "labeled"}
            output_records[group].append(prepared_record)

    split = make_split(
        [record["id"] for record in output_records["labeled"]],
        float(config["data"]["validation_fraction"]),
        int(config["data"]["split_seed"]),
    )
    number_of_classes = len(raw_to_train)
    histogram = class_histogram(output_records["labeled"], number_of_classes)
    write_json(output_records, work_dir / "prepared_index.json")
    write_json(split, work_dir / "split.json")
    write_json(
        {
            "dataset_root": dataset["root"],
            "target_spacing": list(spacing) if spacing is not None else None,
            "number_of_classes": number_of_classes,
            "raw_to_train": {str(key): value for key, value in raw_to_train.items()},
            "train_to_raw": {str(key): value for key, value in train_to_raw.items()},
            "class_voxels": {str(key): value for key, value in sorted(histogram.items())},
            "notes": "Prepared files contain normalized images, a metal proxy mask and labels remapped to contiguous train IDs.",
        },
        work_dir / "dataset_info.json",
    )
    print(f"Prepared data: {prepared_dir}")
    print(f"Classes (including background): {number_of_classes}")
    print(f"Internal split: {len(split['train'])} train / {len(split['val'])} validation")
    print(f"Target spacing: {spacing or 'original spacing'}")


if __name__ == "__main__":
    main()
