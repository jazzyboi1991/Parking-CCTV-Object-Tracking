#!/usr/bin/env python3
"""Merge detector drafts and render numbered images for manual review."""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "images/video_finetune/unlabeled"
DRAFTS = ROOT / "runs/detect/03-runs/detect"
LABELS = ROOT / "images/video_finetune/labels_draft"
OVERLAYS = ROOT / "images/video_finetune/review_overlays"
COLORS = {0: "red", 1: "lime", 2: "cyan"}


def read_boxes(path: Path, mapping: dict[int, int]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate labels: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) not in (5, 6):
            raise ValueError(f"Invalid label row: {path}:{number}")
        class_id = int(fields[0])
        if class_id not in mapping:
            raise ValueError(f"Unexpected class {class_id}: {path}:{number}")
        coords = [float(value) for value in fields[1:5]]
        x, y, width, height = coords
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid box coordinates: {path}:{number}")
        if not (0 <= x - width / 2 < x + width / 2 <= 1
                and 0 <= y - height / 2 < y + height / 2 <= 1):
            raise ValueError(f"Box extends outside image: {path}:{number}")
        rows.append(f"{mapping[class_id]} " + " ".join(fields[1:5]))
    return rows


def main() -> None:
    frames = sorted(FRAMES.glob("frame-*.jpg"))
    if not frames:
        raise SystemExit(f"No frames found: {FRAMES}")
    if LABELS.exists() or OVERLAYS.exists():
        raise SystemExit("Review output already exists; move it aside before regenerating.")

    merged: dict[str, list[str]] = {}
    for frame in frames:
        name = frame.stem + ".txt"
        car_person = read_boxes(
            DRAFTS / "01-video-label-draft/labels" / name, {0: 1, 2: 0}
        )
        carts = read_boxes(DRAFTS / "02-video-cart-draft/labels" / name, {2: 2})
        merged[name] = car_person + carts

    LABELS.mkdir(parents=True)
    OVERLAYS.mkdir(parents=True)
    frame_by_name = {frame.stem + ".txt": frame for frame in frames}
    for name, rows in merged.items():
        label_text = "\n".join(rows) + ("\n" if rows else "")
        (LABELS / name).write_text(label_text, encoding="utf-8")
        frame = frame_by_name[name]
        with Image.open(frame) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for index, row in enumerate(rows, 1):
            class_id, x, y, box_width, box_height = row.split()
            x, y, box_width, box_height = map(float, (x, y, box_width, box_height))
            bounds = (
                (x - box_width / 2) * width,
                (y - box_height / 2) * height,
                (x + box_width / 2) * width,
                (y + box_height / 2) * height,
            )
            color = COLORS[int(class_id)]
            draw.rectangle(bounds, outline=color, width=2)
            draw.text((bounds[0], bounds[1]), str(index), fill=color,
                      stroke_width=1, stroke_fill="black")
        image.save(OVERLAYS / frame.name, quality=95)

    print(f"Created {len(merged)} editable label files in {LABELS}")
    print(f"Created {len(merged)} review images in {OVERLAYS}")
    print("red=car, green=person, cyan=shopping_cart; numbers match label rows")
    print("Review every box and add missed objects before training.")


if __name__ == "__main__":
    main()
