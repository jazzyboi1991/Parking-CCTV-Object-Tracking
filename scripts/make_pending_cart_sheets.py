#!/usr/bin/env python3
"""Create contact sheets for unreviewed Mendeley trolley candidates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_pending(root: Path) -> list[tuple[Path, Path]]:
    review_csv = root / "review" / "review.csv"
    with review_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        pending = [
            row["image"]
            for row in rows
            if row.get("source") == "Mendeley trolley"
            and row.get("structural_check") == "PASS"
            and not row.get("approved", "").strip()
        ]

    return [
        (root / "shopping_trolley_mendeley" / "images" / name,
         root / "shopping_trolley_mendeley" / "labels" / f"{Path(name).stem}.txt")
        for name in sorted(pending)
    ]


def read_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        row = line.split()
        if len(row) != 5:
            continue
        try:
            _, cx, cy, width, height = map(float, row)
            boxes.append((cx, cy, width, height))
        except ValueError:
            continue
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent / "images" / "training_candidates",
        help="training_candidates directory",
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory for contact sheets")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    if args.cols < 1 or args.rows < 1:
        parser.error("--cols and --rows must be positive")

    records = read_pending(args.root)
    if not records:
        print("No structurally valid, unreviewed trolley candidates found.")
        return

    out_dir = args.out or args.root / "review" / "pending_contact_sheets" / "shopping_trolley_mendeley"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_page = args.cols * args.rows
    thumb_w, thumb_h, caption_h = 240, 180, 28
    font = ImageFont.load_default()

    for page_no, start in enumerate(range(0, len(records), per_page), 1):
        subset = records[start : start + per_page]
        sheet = Image.new("RGB", (args.cols * thumb_w, args.rows * (thumb_h + caption_h)), "white")
        draw = ImageDraw.Draw(sheet)

        for i, (image_path, label_path) in enumerate(subset):
            try:
                image = Image.open(image_path).convert("RGB")
            except OSError:
                continue
            original_w, original_h = image.size
            image.thumbnail((thumb_w, thumb_h))
            x, y = (i % args.cols) * thumb_w, (i // args.cols) * (thumb_h + caption_h)
            sheet.paste(image, (x, y))

            scale_x, scale_y = image.width / original_w, image.height / original_h
            for cx, cy, bw, bh in read_boxes(label_path):
                left = x + (cx - bw / 2) * original_w * scale_x
                top = y + (cy - bh / 2) * original_h * scale_y
                right = x + (cx + bw / 2) * original_w * scale_x
                bottom = y + (cy + bh / 2) * original_h * scale_y
                draw.rectangle((left, top, right, bottom), outline="#00cc44", width=2)
                draw.text((left + 2, top + 2), "shopping_cart", fill="#00cc44", font=font)

            draw.text((x + 3, y + thumb_h + 3), image_path.name, fill="black", font=font)

        sheet.save(out_dir / f"pending-{page_no:03d}.jpg", quality=88)

    print(f"pending trolley images={len(records)}")
    print(f"contact sheets={math.ceil(len(records) / per_page)} -> {out_dir}")
    print("This script only creates review sheets; it does not change approved values.")


if __name__ == "__main__":
    main()
