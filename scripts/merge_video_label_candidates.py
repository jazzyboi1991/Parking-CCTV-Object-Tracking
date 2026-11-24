#!/usr/bin/env python3
"""Combine two detector outputs into YOLO-format labels for manual review."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images/video_finetune/unlabeled"
DRAFT = ROOT / "runs/detect/runs/detect"
OUTPUT = ROOT / "images/video_finetune/labels_draft"


def read_boxes(path: Path, mapping: dict[int, int]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate labels: {path}")
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) not in (5, 6):
            raise ValueError(f"Invalid label: {path}:{number}")
        class_id = int(fields[0])
        if class_id not in mapping:
            raise ValueError(f"Unexpected class {class_id}: {path}:{number}")
        coords = [float(value) for value in fields[1:5]]
        if not all(0 <= value <= 1 for value in coords) or min(coords[2:]) <= 0:
            raise ValueError(f"Invalid coordinates: {path}:{number}")
        result.append(f"{mapping[class_id]} " + " ".join(fields[1:5]))
    return result


def main() -> None:
    images = sorted(IMAGES.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No frames found: {IMAGES}")
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists; preserve existing review: {OUTPUT}")
    combined = {}
    for image in images:
        name = image.stem + ".txt"
        car_person = read_boxes(DRAFT / "video-label-draft/labels" / name, {0: 1, 2: 0})
        carts = read_boxes(DRAFT / "video-cart-draft/labels" / name, {2: 2})
        combined[name] = car_person + carts
    OUTPUT.mkdir(parents=True)
    for name, labels in combined.items():
        (OUTPUT / name).write_text("\n".join(labels) + "\n")
    print(f"Created {len(combined)} UNREVIEWED label files in {OUTPUT}")
    print("Review every box and add missed objects before using these labels for training.")


if __name__ == "__main__":
    main()
