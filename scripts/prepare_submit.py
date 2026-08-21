#!/usr/bin/env python3
"""Validate and package Task 1 NIfTI predictions."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


def nifti_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    )


def validate_mask(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            errors.append(f"expected a 3-D image, got {image.shape}")
            return errors
        data = np.asarray(image.dataobj)
        if not np.isfinite(data).all():
            errors.append("contains NaN or infinity")
        if np.any(data < 0):
            errors.append("contains negative labels")
        if not np.allclose(data, np.rint(data)):
            errors.append("contains non-integer labels")
    except Exception as exc:
        errors.append(str(exc))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and zip STS Task 1 NIfTI masks.")
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("task1_submission.zip"))
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument(
        "--archive-root",
        default="",
        help="Optional folder inside the zip (for example 'res'). The default puts masks at zip root.",
    )
    args = parser.parse_args()
    if not args.masks.is_dir():
        raise FileNotFoundError(args.masks)
    files = nifti_files(args.masks)
    if len(files) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} masks, found {len(files)} in {args.masks}")
    failures = {path.name: validate_mask(path) for path in files}
    failures = {name: values for name, values in failures.items() if values}
    if failures:
        details = "; ".join(f"{name}: {', '.join(values)}" for name, values in failures.items())
        raise ValueError(f"Invalid predictions: {details}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.archive_root.strip("/\\")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive_name = f"{prefix}/{path.name}" if prefix else path.name
            archive.write(path, archive_name)
    print(f"Validated and packaged {len(files)} masks: {args.output.resolve()}")


if __name__ == "__main__":
    main()
