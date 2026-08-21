from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.io import case_id_from_path, nifti_files, read_nifti
from cbct_ssl.metrics import macro_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute local Dice, mIoU and 1-mm NSD for labelled development data.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--nsd-tolerance-mm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = {case_id_from_path(path): path for path in nifti_files(args.prediction_dir)}
    labels = {case_id_from_path(path): path for path in nifti_files(args.label_dir)}
    case_ids = sorted(set(predictions) & set(labels))
    if not case_ids:
        raise FileNotFoundError("No matching prediction/label NIfTI names were found.")
    all_metrics = []
    for case_id in case_ids:
        prediction, _, _, _ = read_nifti(predictions[case_id])
        target, _, spacing, _ = read_nifti(labels[case_id])
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch for {case_id}: {prediction.shape} vs {target.shape}")
        metrics = macro_metrics(np.rint(prediction).astype(np.int64), np.rint(target).astype(np.int64), spacing)
        metrics["case_id"] = case_id
        all_metrics.append(metrics)
    summary = {
        "cases": len(all_metrics),
        "macro_dice": float(np.nanmean([entry["macro_dice"] for entry in all_metrics])),
        "macro_iou": float(np.nanmean([entry["macro_iou"] for entry in all_metrics])),
        "macro_nsd_1mm": float(np.nanmean([entry["macro_nsd_1mm"] for entry in all_metrics])),
        "per_case": all_metrics,
        "note": "IoA/instance-identification follows the official challenge definition and is intentionally not approximated here. Use the official evaluator for final ranking.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
