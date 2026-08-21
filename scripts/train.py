from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.config import load_config, with_overrides
from cbct_ssl.engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone artifact-aware semi-supervised 3-D CBCT model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint_last.pt or checkpoint_best.pt.")
    parser.add_argument("--pretrained", default=None, help="Optional self-supervised pretrain_best.pt used to initialize training.")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. train.steps=1000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_overrides(load_config(args.config), args.set)
    work_dir = Path(config["data"]["work_dir"]).resolve()
    expected = [work_dir / "prepared_index.json", work_dir / "split.json", work_dir / "dataset_info.json"]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Prepared data is missing. Run scripts/prepare_data.py first:\n" + "\n".join(missing))
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = work_dir / "runs" / run_name
    train(
        config,
        work_dir,
        run_dir,
        Path(args.resume).resolve() if args.resume else None,
        Path(args.pretrained).resolve() if args.pretrained else None,
    )


if __name__ == "__main__":
    main()
