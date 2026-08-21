from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cbct_ssl.config import load_config, with_overrides
from cbct_ssl.engine import pretrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised disruptive autoencoder pretraining on all CBCT scans.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_overrides(load_config(args.config), args.set)
    work_dir = Path(config["data"]["work_dir"]).resolve()
    required = [work_dir / "prepared_index.json", work_dir / "dataset_info.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Prepared data is missing. Run scripts/prepare_data.py first:\n" + "\n".join(missing))
    run_name = args.run_name or datetime.now().strftime("dae_%Y%m%d_%H%M%S")
    pretrain(config, work_dir, work_dir / "pretrain" / run_name, Path(args.resume).resolve() if args.resume else None)


if __name__ == "__main__":
    main()
