#!/usr/bin/env python3
"""Check image/YOLO file integrity and build human semantic-review contact sheets.

Structural checks can detect broken files and malformed boxes. Human approval is
required for semantic correctness (e.g. trolley vs horse cart, relevant context).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASSES = {0: "car", 1: "person", 2: "shopping_cart"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify_pair(image_path: Path, label_path: Path):
    issues = []
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 160 or height < 160:
                issues.append("low_resolution")
    except Exception as exc:
        return None, [f"image_unreadable:{exc}"]
    boxes = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"bad_label_fields:{line_no}")
                continue
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            cx, cy, bw, bh = vals
            if cls not in CLASSES:
                issues.append(f"unknown_class:{line_no}:{cls}")
            if any(v < 0 or v > 1 for v in vals) or bw <= 0 or bh <= 0:
                issues.append(f"bad_box_range:{line_no}")
            # Allow tiny floating-point drift from source boxes rounded to 6 decimals.
            eps = 1e-5
            if (
                cx - bw / 2 < -eps
                or cy - bh / 2 < -eps
                or cx + bw / 2 > 1 + eps
                or cy + bh / 2 > 1 + eps
            ):
                issues.append(f"box_outside_image:{line_no}")
            boxes.append((cls, cx, cy, bw, bh))
        if not boxes:
            issues.append("no_boxes")
    except Exception as exc:
        issues.append(f"label_unreadable:{exc}")
    return (width, height, boxes), issues


def make_sheets(records, out_dir: Path, source_name: str, cols=5, rows=4):
    sheet_dir = out_dir / source_name
    sheet_dir.mkdir(parents=True, exist_ok=True)
    per_page = cols * rows
    thumb_w, thumb_h, label_h = 240, 180, 28
    font = ImageFont.load_default()
    for page, start in enumerate(range(0, len(records), per_page), 1):
        subset = records[start : start + per_page]
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, rec in enumerate(subset):
            image_path = rec["image_path"]
            img = Image.open(image_path).convert("RGB")
            original_w, original_h = img.size
            img.thumbnail((thumb_w, thumb_h))
            x = (offset % cols) * thumb_w
            y = (offset // cols) * (thumb_h + label_h)
            sheet.paste(img, (x, y))
            if rec.get("boxes"):
                sx, sy = img.width / original_w, img.height / original_h
                for cls, cx, cy, bw, bh in rec["boxes"]:
                    if bw <= 0 or bh <= 0:
                        continue
                    left = x + (cx - bw / 2) * original_w * sx
                    top = y + (cy - bh / 2) * original_h * sy
                    right = x + (cx + bw / 2) * original_w * sx
                    bottom = y + (cy + bh / 2) * original_h * sy
                    if right < left or bottom < top:
                        continue
                    draw.rectangle(
                        (left, top, right, bottom), outline="#00ff55", width=2
                    )
                    class_name = CLASSES.get(cls, f"class_{cls}")
                    draw.text((left + 2, top + 2), class_name, fill="#00ff55", font=font)
            draw.text((x + 3, y + thumb_h + 3), rec["image"], fill="black", font=font)
        sheet.save(sheet_dir / f"review-{page:03d}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent / "images" / "training_candidates",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--promote",
        action="store_true",
        help="copy only structurally valid rows marked approved=yes into a YOLO dataset",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = args.root
    out = args.out or root / "review"
    review_path = out / "review.csv"
    previous_approvals = {}
    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8-sig") as f:
            for old in csv.DictReader(f):
                previous_approvals[(old.get("source", ""), old.get("image", ""))] = old.get("approved", "")
    rows = []
    records = {"shopping_trolley_mendeley": [], "parking_lot_epfl": []}
    cart_root = root / "shopping_trolley_mendeley"
    for image_path in sorted((cart_root / "images").glob("*.png")):
        label_path = cart_root / "labels" / (image_path.stem + ".txt")
        if not label_path.exists():
            rows.append(
                {
                    "source": "Mendeley trolley",
                    "image": image_path.name,
                    "structural_check": "FAIL",
                    "issues": "missing_label",
                    "approved": previous_approvals.get(("Mendeley trolley", image_path.name), ""),
                }
            )
            continue
        result, issues = verify_pair(image_path, label_path)
        status = "PASS" if not issues else "FAIL"
        rows.append(
            {
                "source": "Mendeley trolley",
                "image": image_path.name,
                "structural_check": status,
                "issues": ";".join(issues),
                "approved": previous_approvals.get(("Mendeley trolley", image_path.name), ""),
            }
        )
        if result:
            # In the source trolley dataset, its single class is numbered 0.
            mapped = [(2, *box[1:]) for box in result[2]]
            records["shopping_trolley_mendeley"].append(
                {"image": image_path.name, "image_path": image_path, "boxes": mapped}
            )
    epfl_root = root / "parking_lot_epfl"
    epfl_images = epfl_root / "images"
    epfl_labels = epfl_root / "labels"
    for image_path in sorted(epfl_images.glob("*.jpg")):
        label_path = epfl_labels / (image_path.stem + ".txt")
        if not label_path.exists():
            rows.append(
                {
                    "source": "EPFL parking/campus",
                    "image": image_path.name,
                    "structural_check": "FAIL",
                    "issues": "missing_label",
                    "approved": previous_approvals.get(("EPFL parking/campus", image_path.name), ""),
                }
            )
            continue
        result, issues = verify_pair(image_path, label_path)
        status = "PASS" if not issues else "FAIL"
        rows.append(
            {
                "source": "EPFL parking/campus",
                "image": image_path.name,
                "structural_check": status,
                "issues": ";".join(issues),
                "approved": previous_approvals.get(("EPFL parking/campus", image_path.name), ""),
            }
        )
        if result:
            records["parking_lot_epfl"].append(
                {"image": image_path.name, "image_path": image_path, "boxes": result[2]}
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "image", "structural_check", "issues", "approved"]
        )
        writer.writeheader()
        writer.writerows(rows)
    for source_name, source_records in records.items():
        make_sheets(source_records, out / "contact_sheets", source_name)
    passed = sum(r["structural_check"] == "PASS" for r in rows)
    failed = len(rows) - passed
    print(f"images={len(rows)} structural_pass={passed} structural_fail={failed}")
    print(f"review CSV: {out / 'review.csv'}")
    print(f"contact sheets: {out / 'contact_sheets'}")
    print(
        "Green boxes are source annotations. In review.csv, set approved=yes only when the object and scene are suitable."
    )
    if args.promote:
        output = promote_approved(rows, root, args.seed)
        if output is not None and not audit_yolo_dataset(output):
            raise SystemExit("Promoted dataset audit failed; review the issues above.")


def promote_approved(rows, root: Path, seed: int):
    """Build a clean YOLO folder from manually approved, structurally valid rows."""
    output = root / "verified_yolo_dataset"
    accepted = [
        r for r in rows
        if r["structural_check"] == "PASS" and r.get("approved", "").strip().lower() == "yes"
    ]
    if not accepted:
        print(
            "No images promoted: review.csv has no rows marked approved=yes. "
            "Approve suitable images in the CSV, then run this command again."
        )
        return None

    items = []
    for row in accepted:
        if row["source"] == "Mendeley trolley":
            image = root / "shopping_trolley_mendeley" / "images" / row["image"]
            label = root / "shopping_trolley_mendeley" / "labels" / (Path(row["image"]).stem + ".txt")
            class_map = {0: 2}  # Mendeley's one class is shopping-trolley
            group = ("cart", Path(row["image"]).stem)
        else:
            image = root / "parking_lot_epfl" / "images" / row["image"]
            label = root / "parking_lot_epfl" / "labels" / (Path(row["image"]).stem + ".txt")
            class_map = {0: 0, 1: 1}  # EPFL's car/person annotations
            match = re.match(r"frame(\d+)_cam\d+", Path(row["image"]).stem)
            group = ("epfl", match.group(1) if match else Path(row["image"]).stem)
        if not image.exists() or not label.exists():
            continue
        converted = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            original_class = int(parts[0])
            if original_class not in class_map:
                continue
            converted.append(" ".join([str(class_map[original_class]), *parts[1:]]))
        if converted:
            items.append({"image": image, "label": converted, "group": group})

    if not items:
        print("No images promoted: approved rows had no usable image/label pairs.")
        return None

    # Preserve any previous output when approval is empty or the source pairs are missing.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Split by EPFL frame ID so synchronized camera views never leak across splits.
    grouped = {}
    for item in items:
        grouped.setdefault(item["group"], []).append(item)
    group_keys = list(grouped)
    random.Random(seed).shuffle(group_keys)
    train_groups = set(group_keys[: max(1, int(len(group_keys) * 0.8))])
    for item in items:
        split = "train" if item["group"] in train_groups else "valid"
        image_out = output / split / "images" / item["image"].name
        label_out = output / split / "labels" / (item["image"].stem + ".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["image"], image_out)
        label_out.write_text("\n".join(item["label"]) + "\n", encoding="utf-8")

    # Use an absolute dataset root so Ultralytics resolves train/val paths
    # correctly regardless of the shell's current working directory.
    yaml_root = str(output.resolve()).replace("'", "''")
    (output / "data.yaml").write_text(
        f"path: '{yaml_root}'\ntrain: train/images\nval: valid/images\nnc: 3\nnames: [car, person, shopping_cart]\n",
        encoding="utf-8",
    )
    print(f"approved images promoted: {len(items)} -> {output}")
    return output


def audit_yolo_dataset(root: Path) -> bool:
    """Report train/valid image-label pairing and per-class label counts."""
    config = root / "data.yaml"
    if not config.is_file():
        print(f"Missing dataset config: {config}")
        return False
    print("data.yaml:")
    print(config.read_text(encoding="utf-8"))

    all_ok = True
    for split in ("train", "valid"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            print(f"[{split}] missing images/ or labels/ directory")
            all_ok = False
            continue

        images = {p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {p.stem for p in label_dir.glob("*.txt")}
        counts = Counter()
        images_with_class = Counter()
        invalid_rows = []
        empty_labels = 0

        for stem in sorted(labels):
            label_path = label_dir / f"{stem}.txt"
            lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            seen = set()
            for line_number, row in enumerate(lines, 1):
                try:
                    class_id = int(row[0])
                    box = list(map(float, row[1:]))
                    if (
                        len(row) != 5
                        or class_id not in CLASSES
                        or not all(math.isfinite(value) for value in box)
                        or any(value < 0 or value > 1 for value in box)
                        or box[2] <= 0
                        or box[3] <= 0
                        or box[0] - box[2] / 2 < -1e-5
                        or box[1] - box[3] / 2 < -1e-5
                        or box[0] + box[2] / 2 > 1 + 1e-5
                        or box[1] + box[3] / 2 > 1 + 1e-5
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
        for class_id, class_name in CLASSES.items():
            print(f"{class_name}: objects={counts[class_id]} images={images_with_class[class_id]}")
        if invalid_rows:
            print("invalid examples:")
            for path in invalid_rows[:10]:
                print(f"  {path}")
        all_ok &= not (missing_labels or missing_images or invalid_rows)

    return all_ok


if __name__ == "__main__":
    main()
