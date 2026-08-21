from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CBCT training metrics.jsonl and give a conservative stop/continue recommendation."
    )
    parser.add_argument(
        "metrics",
        nargs="?",
        default="实验结果/train/metrics.jsonl",
        help="Path to metrics.jsonl (default: 实验结果/train/metrics.jsonl).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training config.json. By default, use config.json beside metrics.jsonl when present.",
    )
    parser.add_argument("--window", type=int, default=10, help="Number of logged training points in each trend window.")
    parser.add_argument("--output-json", default=None, help="Summary JSON path; defaults beside metrics.jsonl.")
    parser.add_argument("--output-csv", default=None, help="Normalized CSV path; defaults beside metrics.jsonl.")
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                invalid.append({"line": line_number, "error": str(error), "text": line[:200]})
                continue
            if not isinstance(record, dict):
                invalid.append({"line": line_number, "error": "JSON value is not an object", "text": line[:200]})
            elif "validation_macro_dice" in record:
                validation.append(record)
            elif "loss" in record:
                training.append(record)
            else:
                invalid.append({"line": line_number, "error": "Unknown record type", "text": line[:200]})
    training.sort(key=lambda item: int(item["step"]))
    validation.sort(key=lambda item: int(item["step"]))
    return training, validation, invalid


def finite_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        try:
            value = float(record[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def linear_slope(records: list[dict[str, Any]], key: str) -> float | None:
    points: list[tuple[float, float]] = []
    for record in records:
        try:
            x, y = float(record["step"]), float(record[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    if len(points) < 2:
        return None
    x_mean = statistics.fmean(point[0] for point in points)
    y_mean = statistics.fmean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    slope_per_step = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    return slope_per_step * 1000.0


def load_config(metrics_path: Path, explicit_path: str | None) -> tuple[dict[str, Any], Path | None]:
    config_path = Path(explicit_path) if explicit_path else metrics_path.with_name("config.json")
    if not config_path.exists():
        return {}, None
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), config_path.resolve()


def summarize(
    metrics_path: Path,
    training: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    config: dict[str, Any],
    window: int,
) -> dict[str, Any]:
    if not training:
        raise ValueError("No training records containing 'loss' were found.")
    window = max(2, min(window, len(training)))
    early = training[:window]
    recent = training[-window:]
    latest_step = max(int(training[-1]["step"]), int(validation[-1]["step"]) if validation else 0)
    losses = finite_values(training, "loss")
    early_losses = finite_values(early, "loss")
    recent_losses = finite_values(recent, "loss")
    speed_values = [value for value in finite_values(training, "steps_per_second") if value > 0]
    non_finite_count = 0
    for record in training:
        for key in ("loss", "supervised", "restoration", "consistency", "pseudo", "learning_rate"):
            if key in record:
                try:
                    if not math.isfinite(float(record[key])):
                        non_finite_count += 1
                except (TypeError, ValueError):
                    non_finite_count += 1

    train_config = config.get("train", {})
    semi_config = config.get("semi_supervised", {})
    total_steps = int(train_config["steps"]) if "steps" in train_config else None
    validate_every = int(train_config["validate_every"]) if "validate_every" in train_config else None
    checkpoint_every = int(train_config["checkpoint_every"]) if "checkpoint_every" in train_config else None
    patience = int(train_config["early_stop_patience"]) if "early_stop_patience" in train_config else None
    semi_enabled = bool(semi_config.get("enabled", False))
    semi_start = int(total_steps * float(semi_config.get("start_fraction", 0.0))) if total_steps and semi_enabled else None
    semi_ramp_end = (
        semi_start + int(total_steps * float(semi_config.get("ramp_fraction", 0.0)))
        if total_steps and semi_start is not None
        else None
    )

    full_validation = [
        {"step": int(item["step"]), **item["full_volume"]}
        for item in validation
        if isinstance(item.get("full_volume"), dict) and item["full_volume"].get("selection_score") is not None
    ]
    selection_records = full_validation if full_validation else validation
    selection_key = "selection_score" if full_validation else "validation_macro_dice"
    best_validation = None
    best_validation_step = None
    stale_validations = None
    validation_slope = linear_slope(selection_records, selection_key)
    if selection_records:
        best_index = max(range(len(selection_records)), key=lambda index: float(selection_records[index][selection_key]))
        best_validation = float(selection_records[best_index][selection_key])
        best_validation_step = int(selection_records[best_index]["step"])
        stale_validations = len(selection_records) - best_index - 1

    median_speed = statistics.median(speed_values) if speed_values else None
    remaining_steps = max(0, total_steps - latest_step) if total_steps is not None else None
    eta_hours = remaining_steps / median_speed / 3600.0 if remaining_steps is not None and median_speed else None
    expected_checkpoint_step = (
        latest_step - latest_step % checkpoint_every if checkpoint_every and latest_step >= checkpoint_every else None
    )

    reasons: list[str] = []
    decision = "INSUFFICIENT_DATA"
    if non_finite_count:
        decision = "STOP_AND_INVESTIGATE"
        reasons.append(f"Found {non_finite_count} non-finite numeric values.")
    elif semi_enabled and semi_start is not None and latest_step < semi_start:
        decision = "CONTINUE"
        reasons.append(f"Semi-supervised learning has not started; it begins at step {semi_start}.")
    elif len(validation) < 3:
        decision = "CONTINUE"
        reasons.append("Fewer than three validation points are available, so convergence cannot be assessed.")
    elif patience is not None and stale_validations is not None and stale_validations >= patience:
        decision = "STOP_OR_ACCEPT_EARLY_STOP"
        reasons.append(f"Validation has not improved for {stale_validations} checks (patience={patience}).")
    else:
        decision = "CONTINUE"
        reasons.append("The configured early-stopping condition has not been reached.")
    if best_validation is not None and best_validation < 0.01:
        reasons.append("Best internal validation Dice is below 0.01; this is not a usable converged model yet.")
    if len(validation) < 3:
        reasons.append("Only two or fewer validation points are available; a convergence trend is not established.")

    recent_loss_mean = statistics.fmean(recent_losses) if recent_losses else None
    early_loss_mean = statistics.fmean(early_losses) if early_losses else None
    try:
        latest_loss = float(training[-1]["loss"])
        if not math.isfinite(latest_loss):
            latest_loss = None
    except (KeyError, TypeError, ValueError):
        latest_loss = None
    loss_change_percent = (
        100.0 * (recent_loss_mean - early_loss_mean) / early_loss_mean
        if recent_loss_mean is not None and early_loss_mean not in (None, 0.0)
        else None
    )
    modified = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600.0
    if age_hours > 1.0:
        reasons.append(
            f"This local log copy is {age_hours:.1f} hours old; inspect the live server log before stopping the process."
        )

    return {
        "metrics_path": str(metrics_path.resolve()),
        "log_modified_utc": modified.isoformat(),
        "log_age_hours": round(age_hours, 3),
        "records": {"training": len(training), "validation": len(validation), "invalid": len(invalid)},
        "progress": {
            "latest_logged_step": latest_step,
            "configured_total_steps": total_steps,
            "percent": round(100.0 * latest_step / total_steps, 3) if total_steps else None,
            "semi_supervised_start_step": semi_start,
            "semi_supervised_ramp_end_step": semi_ramp_end,
        },
        "loss": {
            "first": losses[0] if losses else None,
            "latest": latest_loss,
            "early_window_mean": early_loss_mean,
            "recent_window_mean": recent_loss_mean,
            "recent_vs_early_percent": loss_change_percent,
            "recent_slope_per_1000_steps": linear_slope(recent, "loss"),
            "non_finite_values": non_finite_count,
        },
        "validation": {
            "points": [
                {"step": int(item["step"]), "validation_macro_dice": float(item["validation_macro_dice"])}
                for item in validation
            ],
            "full_volume_points": full_validation,
            "selection_metric": selection_key,
            "best": best_validation,
            "best_step": best_validation_step,
            "latest": float(selection_records[-1][selection_key]) if selection_records else None,
            "slope_per_1000_steps": validation_slope,
            "stale_checks_after_best": stale_validations,
            "configured_interval": validate_every,
            "early_stop_patience": patience,
        },
        "speed": {
            "median_steps_per_second": median_speed,
            "remaining_steps": remaining_steps,
            "estimated_remaining_hours": eta_hours,
        },
        "checkpoint": {
            "configured_interval": checkpoint_every,
            "latest_expected_periodic_checkpoint_step": expected_checkpoint_step,
            "steps_after_expected_checkpoint": latest_step - expected_checkpoint_step if expected_checkpoint_step else None,
            "warning": "The log cannot prove that checkpoint_last.pt exists or finished writing; verify it in the real run directory.",
        },
        "recommendation": {"decision": decision, "reasons": reasons},
        "invalid_records": invalid,
    }


def write_csv(path: Path, training: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in training:
        rows.append({"record_type": "training", **record})
    for record in validation:
        rows.append({"record_type": "validation", **record})
    rows.sort(key=lambda item: (int(item.get("step", 0)), item["record_type"]))
    keys = ["record_type", "step"]
    keys.extend(sorted({key for row in rows for key in row if key not in keys}))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_report(summary: dict[str, Any]) -> None:
    def formatted(value: Any, digits: int) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "non-finite"

    progress = summary["progress"]
    loss = summary["loss"]
    validation = summary["validation"]
    speed = summary["speed"]
    checkpoint = summary["checkpoint"]
    recommendation = summary["recommendation"]
    print(f"Decision: {recommendation['decision']}")
    print(
        f"Progress: step {progress['latest_logged_step']} / {progress['configured_total_steps']} "
        f"({progress['percent']}%)"
    )
    print(
        f"Loss: latest={formatted(loss['latest'], 6)}, "
        f"recent mean={formatted(loss['recent_window_mean'], 6)}, "
        f"recent/early change={formatted(loss['recent_vs_early_percent'], 2)}%"
    )
    if validation["points"]:
        print(
            f"Validation ({validation['selection_metric']}): latest={formatted(validation['latest'], 8)}, "
            f"best={formatted(validation['best'], 8)} at step {validation['best_step']}"
        )
    else:
        print("Validation: no validation records")
    if speed["median_steps_per_second"] is not None and speed["estimated_remaining_hours"] is not None:
        print(
            f"Speed: median={speed['median_steps_per_second']:.3f} steps/s, "
            f"estimated remaining={speed['estimated_remaining_hours']:.2f} h"
        )
    print(
        "Checkpoint: latest expected periodic step="
        f"{checkpoint['latest_expected_periodic_checkpoint_step']}, "
        f"uncheckpointed logged steps={checkpoint['steps_after_expected_checkpoint']}"
    )
    for reason in recommendation["reasons"]:
        print(f"- {reason}")
    print(f"- {checkpoint['warning']}")


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    training, validation, invalid = load_jsonl(metrics_path)
    config, config_path = load_config(metrics_path, args.config)
    summary = summarize(metrics_path, training, validation, invalid, config, args.window)
    summary["config_path"] = str(config_path) if config_path else None

    output_json = Path(args.output_json) if args.output_json else metrics_path.with_name("training_analysis.json")
    output_csv = Path(args.output_csv) if args.output_csv else metrics_path.with_name("training_metrics.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_csv, training, validation)
    print_report(summary)
    print(f"JSON report: {output_json}")
    print(f"CSV table: {output_csv}")


if __name__ == "__main__":
    main()
