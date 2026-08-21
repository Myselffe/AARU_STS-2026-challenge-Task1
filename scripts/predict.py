from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.config import load_config, with_overrides
from cbct_ssl.engine import build_model
from cbct_ssl.inference import input_files, predict_case
from cbct_ssl.io import case_id_from_path, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone CBCT tooth/pulp segmentation and optional MAR output.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint_best.pt or checkpoint_last.pt")
    parser.add_argument("--input-dir", default=None, help="Defaults to Dataset/Validation/images from the config.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <work_dir>/predictions.")
    parser.add_argument("--write-restored", action="store_true", help="Write self-supervised artifact-suppressed CBCT volumes too.")
    parser.add_argument(
        "--submission-fast",
        action="store_true",
        help="Only write submission masks and use 25%% overlap for faster inference.",
    )
    parser.add_argument("--overlap", type=float, default=None, help="Override sliding-window overlap, e.g. 0.25 or 0.5.")
    parser.add_argument(
        "--tta-axes",
        type=int,
        nargs="*",
        choices=(0, 1, 2),
        default=None,
        help="Optional spatial flip TTA axes. Use only axes whose flip does not change label semantics.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=None,
        help="Override inference patch size. Reduce this only if one patch still causes CUDA OOM.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip masks that already exist in the output directory.")
    parser.add_argument(
        "--ignore-prepared-cache",
        action="store_true",
        help="Ignore prepared/validation/*.npz and preprocess the original NIfTI again.",
    )
    parser.add_argument(
        "--resample-order",
        type=int,
        choices=(1, 3),
        default=1,
        help="Fallback image interpolation when no prepared validation cache exists. Default 1 is much faster than cubic 3.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Timing/peak-VRAM JSON. Defaults beside the output directory, not inside the submission folder.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Only used if checkpoint lacks its saved config.")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.submission_fast and args.write_restored:
        raise ValueError("--submission-fast only creates submission masks; do not combine it with --write-restored.")
    if args.overlap is not None and not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1).")

    fallback_config = load_config(args.config)
    # Loading the complete training checkpoint directly on CUDA can copy EMA,
    # optimizer and scheduler states to the GPU even though inference needs only
    # one state dict. Keep it on CPU until the selected model has been restored.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = with_overrides(checkpoint.get("config", fallback_config), args.set)
    requested_device = config["train"].get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() or not str(requested_device).startswith("cuda") else "cpu")
    work_dir = Path(config["data"]["work_dir"]).resolve()
    info = read_json(work_dir / "dataset_info.json")
    number_of_classes = int(checkpoint.get("number_of_classes", info["number_of_classes"]))
    model = build_model(config, number_of_classes)
    # EMA is the preferred inference model; old checkpoints may only contain model.
    state_dict = checkpoint.get("ema", checkpoint["model"])
    model.load_state_dict(state_dict)
    del state_dict, checkpoint
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    input_dir = Path(args.input_dir or Path(config["data"]["dataset_root"]) / "Validation" / "images")
    output_dir = Path(args.output_dir or work_dir / "predictions")
    restored_dir = output_dir.parent / f"{output_dir.name}_mar" if args.write_restored else None
    inference_config = config["inference"]
    patch_size = tuple(args.patch_size or (int(value) for value in inference_config["patch_size"]))
    overlap = args.overlap if args.overlap is not None else (0.25 if args.submission_fast else float(inference_config["overlap"]))
    tta_axes = () if args.submission_fast else tuple(args.tta_axes if args.tta_axes is not None else inference_config.get("tta_axes", []))
    amp = bool(inference_config["amp"])
    amp_dtype = str(inference_config.get("amp_dtype", config["train"].get("amp_dtype", "bfloat16")))
    raw_to_train = {int(raw): int(train_id) for raw, train_id in info["raw_to_train"].items()}
    preserve_train_labels = [
        raw_to_train[int(raw)]
        for raw in inference_config.get("preserve_raw_labels", [])
        if int(raw) in raw_to_train
    ]
    cases = input_files(input_dir)
    case_metrics: list[dict[str, float | str]] = []
    prepared_validation: dict[str, Path] = {}
    if not args.ignore_prepared_cache:
        prepared_index_path = work_dir / "prepared_index.json"
        if prepared_index_path.exists():
            prepared_index = read_json(prepared_index_path)
            for record in prepared_index.get("validation", []):
                record_id = str(record["id"])
                recorded_path = Path(record["file"])
                fallback_path = work_dir / "prepared" / "validation" / f"{record_id}.npz"
                if recorded_path.exists():
                    prepared_validation[record_id] = recorded_path
                elif fallback_path.exists():
                    prepared_validation[record_id] = fallback_path
        # Also support moved projects or an index created by an older version.
        for path in cases:
            case_id = case_id_from_path(path)
            direct_path = work_dir / "prepared" / "validation" / f"{case_id}.npz"
            if case_id not in prepared_validation and direct_path.exists():
                prepared_validation[case_id] = direct_path
    print(
        f"Prepared validation cache: {len(prepared_validation)}/{len(cases)} cases. "
        + ("Raw NIfTI preprocessing forced." if args.ignore_prepared_cache else "Cached cases skip SciPy resampling."),
        flush=True,
    )

    for path in tqdm(cases, desc="Predicting"):
        case_id = case_id_from_path(path)
        output_path = output_dir / f"{case_id}.nii.gz"
        if args.skip_existing and output_path.exists():
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predict_case(
            model=model,
            input_path=path,
            output_path=output_path,
            train_to_raw=info["train_to_raw"],
            target_spacing=info["target_spacing"],
            patch_size=patch_size,
            overlap=overlap,
            amp=amp,
            minimum_component_voxels=int(inference_config["min_component_voxels"]),
            restored_path=(restored_dir / f"{case_id}_mar.nii.gz") if restored_dir is not None else None,
            prepared_path=prepared_validation.get(case_id),
            image_resample_order=args.resample_order,
            preserve_train_labels=preserve_train_labels,
            tta_axes=tta_axes,
            amp_dtype=amp_dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        else:
            peak_allocated_mb = 0.0
            peak_reserved_mb = 0.0
        case_metrics.append(
            {
                "case_id": case_id,
                "seconds": round(time.perf_counter() - started, 3),
                "peak_gpu_allocated_mb": round(peak_allocated_mb, 1),
                "peak_gpu_reserved_mb": round(peak_reserved_mb, 1),
            }
        )

    total_seconds = sum(float(item["seconds"]) for item in case_metrics)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output_dir.parent / f"{output_dir.name}_inference_metrics.json"
    write_json(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "device": str(device),
            "patch_size": list(patch_size),
            "overlap": overlap,
            "amp": amp,
            "amp_dtype": amp_dtype,
            "tta_axes": list(tta_axes),
            "submission_fast": args.submission_fast,
            "prepared_cache_cases": len(prepared_validation),
            "fallback_resample_order": args.resample_order,
            "processed_cases": len(case_metrics),
            "total_seconds": round(total_seconds, 3),
            "mean_seconds_per_case": round(total_seconds / len(case_metrics), 3) if case_metrics else 0.0,
            "cases": case_metrics,
        },
        metrics_path,
    )
    print(f"Segmentation outputs: {output_dir}")
    print(f"Inference efficiency metrics: {metrics_path}")
    if restored_dir is not None:
        print(f"Artifact-suppressed CBCT outputs: {restored_dir}")


if __name__ == "__main__":
    main()
