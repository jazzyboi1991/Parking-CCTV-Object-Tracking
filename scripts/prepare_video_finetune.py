#!/usr/bin/env python3
"""Prepare a separate dataset using video label drafts and existing training data."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "images/training_candidates/verified_yolo_dataset"
FRAMES = ROOT / "images/video_finetune/unlabeled"
LABELS = ROOT / "images/video_finetune/labels_draft"
OUTPUT = ROOT / "images/video_finetune/dataset"


def main() -> None:
    old_images = sorted(p.resolve() for p in (BASE / "train/images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    frames = sorted(FRAMES.glob("frame-*.jpg"))
    if not old_images or len(frames) != 10 or not (BASE / "valid/images").is_dir():
        raise SystemExit("Expected existing train/valid dataset and 10 video frames.")
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists: {OUTPUT}")
    for frame in frames:
        path = LABELS / f"{frame.stem}.txt"
        if not path.is_file() or not path.read_text().strip():
            raise SystemExit(f"Missing or empty label: {path}")
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[0] not in {"0", "1", "2"}:
                raise SystemExit(f"Invalid label in {path}: {line}")
            x, y, w, h = map(float, fields[1:])
            if not (0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1):
                raise SystemExit(f"Invalid bounding box in {path}: {line}")
    image_dir = OUTPUT / "images"
    label_dir = OUTPUT / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    # Distinct filenames retain repeated samples when the loader builds its cache.
    new_images = []
    for frame in frames:
        for repeat in range(20):
            stem = f"{frame.stem}-repeat-{repeat:02d}"
            destination = image_dir / f"{stem}.jpg"
            shutil.copy2(frame, destination)
            shutil.copy2(LABELS / f"{frame.stem}.txt", label_dir / f"{stem}.txt")
            new_images.append(str(destination))
    (OUTPUT / "train.txt").write_text("\n".join(map(str, old_images)) + "\n" +
                                      "\n".join(new_images) + "\n")
    import json
    (OUTPUT / "data.yaml").write_text(
        f"path: {json.dumps(str(OUTPUT), ensure_ascii=False)}\n"
        f"train: train.txt\n"
        f"val: {json.dumps(str(BASE / 'valid/images'), ensure_ascii=False)}\n"
        "names: [car, person, shopping_cart]\n"
    )
    print(f"Prepared {len(old_images)} original and {len(frames)} video frames (20 repeats) in {OUTPUT}")
    print("Video labels are draft annotations; assess the final video and use a different scene for independent evaluation.")


if __name__ == "__main__":
    main()
