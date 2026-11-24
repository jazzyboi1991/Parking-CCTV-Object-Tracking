#!/usr/bin/env python3
"""Audit YOLO image/label pairs, class counts, and normalized boxes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


CLASSES = {0: "car", 1: "person", 2: "shopping_cart"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_split(root: Path, split: str) -> bool:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"[{split}] missing images/ or labels/ directory")
        return False

    images = {p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {p.stem for p in label_dir.glob("*.txt")}
    counts: Counter[int] = Counter()
    images_with_class: Counter[int] = Counter()
    invalid_rows: list[str] = []
    empty_labels = 0

    for stem in sorted(labels):
        label_path = label_dir / f"{stem}.txt"
        lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        seen: set[int] = set()

        for line_number, row in enumerate(lines, 1):
            try:
                class_id = int(row[0])
                box = list(map(float, row[1:]))
                if (
                    len(row) != 5
                    or class_id not in CLASSES
                    or any(value < 0 or value > 1 for value in box)
                    or box[2] <= 0
                    or box[3] <= 0
                ):
                    raise ValueError("invalid class or box")
                counts[class_id] += 1
                seen.add(class_id)
            except (ValueError, IndexError):
                invalid_rows.append(f"{label_path}:{line_number}")

        for class_id in seen:
            images_with_class[class_id] += 1

    missing_labels = images - labels
    missing_images = labels - images
    print(f"\n[{split}]")
    print(f"images={len(images)} labels={len(labels)}")
    print(f"images_without_labels={len(missing_labels)} labels_without_images={len(missing_images)}")
    print(f"empty_label_files={empty_labels} invalid_rows={len(invalid_rows)}")
    for class_id, name in CLASSES.items():
        print(f"{name}: objects={counts[class_id]} images={images_with_class[class_id]}")
    if invalid_rows:
        print("invalid examples:")
        for path in invalid_rows[:10]:
            print(f"  {path}")

    return not (missing_labels or missing_images or invalid_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent / "images" / "training_candidates" / "verified_yolo_dataset",
        help="YOLO dataset directory containing train/, valid/, and data.yaml",
    )
    args = parser.parse_args()

    root = args.root
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        print("data.yaml:")
        print(yaml_path.read_text(encoding="utf-8"))
    else:
        print(f"Missing dataset config: {yaml_path}")

    ok = all(audit_split(root, split) for split in ("train", "valid"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
