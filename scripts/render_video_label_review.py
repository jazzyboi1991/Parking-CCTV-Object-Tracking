#!/usr/bin/env python3
"""Draw numbered candidate boxes for reviewing video frame annotations."""

from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "images/video_finetune/unlabeled"
LABELS = ROOT / "images/video_finetune/labels_draft"
OUTPUT = ROOT / "images/video_finetune/review_overlays"
COLORS = {0: "red", 1: "lime", 2: "cyan"}


def main() -> None:
    frames = sorted(FRAMES.glob("frame-*.jpg"))
    if not frames:
        raise SystemExit(f"No frames found: {FRAMES}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        label_path = LABELS / (frame.stem + ".txt")
        if not label_path.is_file():
            raise SystemExit(f"Missing label: {label_path}")
        with Image.open(frame) as src:
            image = src.convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for number, line in enumerate(label_path.read_text().splitlines(), 1):
            fields = line.split()
            cls = int(fields[0])
            x, y, bw, bh = map(float, fields[1:5])
            box = ((x - bw / 2) * w, (y - bh / 2) * h,
                   (x + bw / 2) * w, (y + bh / 2) * h)
            color = COLORS[cls]
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], box[1]), str(number), fill=color, stroke_width=1, stroke_fill="black")
        image.save(OUTPUT / frame.name, quality=95)
    print(f"Created {len(frames)} review images in {OUTPUT}")
    print("red=car, green=person, cyan=shopping_cart; numbers match label rows")


if __name__ == "__main__":
    main()
