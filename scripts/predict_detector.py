#!/usr/bin/env python3
"""Run the trained detector on an image or video and save annotated output."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str, default: Path | None = None) -> Path:
    path = Path(value) if value else default
    if path is None:
        raise ValueError("a path is required")
    if not path.is_absolute():
        project_path = ROOT / path
        path = project_path if project_path.exists() else path.resolve()
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/detect/train/weights/best.pt",
        help="trained weights path (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        default="video/ViralHog-Shopping-Carts-Collide-with-Cars-clip-20-25s.mp4",
        help="image or video path (default: %(default)s)",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="confidence threshold; lower values retain small, low-confidence objects",
    )
    parser.add_argument("--device", default="mps", help="mps, cpu, or a CUDA device")
    parser.add_argument(
        "--name", default="predict", help="output folder under runs/detect"
    )
    args = parser.parse_args()

    weights = resolve_path(args.model)
    source = resolve_path(args.source)
    if not weights.is_file():
        parser.error(f"model weights not found: {weights}")
    if not source.exists():
        parser.error(f"source not found: {source}")
    if args.imgsz <= 0 or not 0 < args.conf <= 1:
        parser.error("--imgsz must be positive and --conf must be in (0, 1]")

    model = YOLO(str(weights))
    frame_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    max_confidence: defaultdict[str, float] = defaultdict(float)
    output_dir: Path | None = None
    frame_total = 0

    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        stream=True,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    for result in results:
        frame_total += 1
        output_dir = Path(result.save_dir)
        classes_in_frame: set[str] = set()
        if result.boxes is None:
            continue
        for cls, confidence in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            label = model.names[int(cls)]
            classes_in_frame.add(label)
            object_counts[label] += 1
            max_confidence[label] = max(max_confidence[label], float(confidence))
        frame_counts.update(classes_in_frame)

    print(f"frames={frame_total}")
    print(f"frames_with_class={dict(frame_counts)}")
    print(f"objects={dict(object_counts)}")
    print(f"max_confidence={dict(max_confidence)}")
    print(
        f"annotated_output={output_dir if output_dir else ROOT / 'runs' / 'detect' / args.name}"
    )


if __name__ == "__main__":
    main()
