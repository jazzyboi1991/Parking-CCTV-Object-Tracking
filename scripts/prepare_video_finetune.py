#!/usr/bin/env python3
"""Prepare a separate dataset using video label drafts and existing training data."""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "images/training_candidates/verified_yolo_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", type=Path, default=ROOT / "images/video_finetune/unlabeled"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "images/video_finetune/labels_draft"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "images/video_finetune/dataset"
    )
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    frames_dir = args.frames if args.frames.is_absolute() else ROOT / args.frames
    labels_dir = args.labels if args.labels.is_absolute() else ROOT / args.labels
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    old_images = sorted(
        p.resolve()
        for p in (BASE / "train/images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    frames = sorted(frames_dir.glob("frame-*.jpg"))
    if not old_images or not frames or not (BASE / "valid/images").is_dir():
        raise SystemExit(
            "Expected existing train/valid dataset and sampled video frames."
        )
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")
    for frame in frames:
        path = labels_dir / f"{frame.stem}.txt"
        if not path.is_file():
            raise SystemExit(f"Missing label: {path}")
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[0] not in {"0", "1", "2"}:
                raise SystemExit(f"Invalid label in {path}: {line}")
            x, y, w, h = map(float, fields[1:])
            if not (
                0 <= x - w / 2 < x + w / 2 <= 1 and 0 <= y - h / 2 < y + h / 2 <= 1
            ):
                raise SystemExit(f"Invalid bounding box in {path}: {line}")
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    # Distinct filenames retain repeated samples when the loader builds its cache.
    new_images = []
    for frame in frames:
        for repeat in range(args.repeat):
            stem = f"{frame.stem}-repeat-{repeat:02d}"
            destination = image_dir / f"{stem}.jpg"
            shutil.copy2(frame, destination)
            shutil.copy2(labels_dir / f"{frame.stem}.txt", label_dir / f"{stem}.txt")
            new_images.append(str(destination))
    (output_dir / "train.txt").write_text(
        "\n".join(map(str, old_images)) + "\n" + "\n".join(new_images) + "\n"
    )
    import json

    (output_dir / "data.yaml").write_text(
        f"path: {json.dumps(str(output_dir), ensure_ascii=False)}\n"
        f"train: train.txt\n"
        f"val: {json.dumps(str(BASE / 'valid/images'), ensure_ascii=False)}\n"
        "names: [car, person, shopping_cart]\n"
    )
    print(
        f"Prepared {len(old_images)} original and {len(frames)} video frames ({args.repeat} repeats) in {output_dir}"
    )
    print(
        "Video labels are draft annotations; assess the final video and use a different scene for independent evaluation."
    )


if __name__ == "__main__":
    main()
