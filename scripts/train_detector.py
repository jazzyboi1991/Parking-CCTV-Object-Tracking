#!/usr/bin/env python3
"""Train a small Ultralytics detector on the promoted YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=None, help="path to YOLO data.yaml"
    )
    parser.add_argument(
        "--model", default="yolo11n.pt", help="model weights or model YAML"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps", help="e.g. mps, cpu, or 0")
    parser.add_argument("--name", default="parking-cctv-baseline")
    args = parser.parse_args()

    data = args.data or (
        Path(__file__).parent.parent
        / "images"
        / "training_candidates"
        / "verified_yolo_dataset"
        / "data.yaml"
    )
    if not data.is_file():
        parser.error(
            f"dataset config not found: {data}; run verify_training_images.py --promote first"
        )

    model = YOLO(args.model)
    model.train(
        data=str(data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
