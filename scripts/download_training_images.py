#!/usr/bin/env python3
"""Download parking-scene car/person images and shopping-trolley examples.

Requires Python 3.10+ and Pillow. Car/person images and annotations are from
EPFL CVLab; trolley images and YOLO boxes are from Mendeley Data DOI
10.17632/vj9g4mn4vv.1 (CC BY 4.0). This script does not run unless invoked.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

API_URL = "https://data.mendeley.com/public-api/datasets/vj9g4mn4vv"
DATASET_URL = "https://data.mendeley.com/datasets/vj9g4mn4vv/1"
EPFL_IMAGES_URL = "https://documents.epfl.ch/groups/d/di/disal-sem/public/multiclass_ground_truth_images.zip"
EPFL_LABELS_URL = (
    "https://documents.epfl.ch/groups/d/di/disal-sem/public/multiclass_ground_truth.zip"
)
USER_AGENT = "ML-course-dataset-collector/1.0"


def read_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def download(url: str, path: Path, expected_sha256: str | None = None):
    if path.exists() and path.stat().st_size:
        data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is None or data_hash == expected_sha256:
            return
    err = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
            if not data:
                raise ValueError("empty response")
            if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ValueError("SHA-256 mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return
        except Exception as exc:
            err = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to download {url}: {err}")


def collect_carts(root: Path, limit: int, seed: int):
    metadata = read_json(API_URL)
    files = metadata["files"]
    by_name = {item["filename"]: item for item in files}
    positives = [f for f in files if f["filename"].endswith(".txt") and f["size"] > 0]
    random.Random(seed).shuffle(positives)
    positives = positives[:limit]
    target = root / "shopping_trolley_mendeley"
    image_dir, label_dir = target / "images", target / "labels"
    jobs = []
    for label in positives:
        image_name = label["filename"].rsplit(".", 1)[0] + ".png"
        image = by_name.get(image_name)
        if not image:
            continue
        jobs.extend(
            [
                (label, label_dir / label["filename"]),
                (image, image_dir / image_name),
            ]
        )

    def worker(job):
        item, path = job
        download(
            item["content_details"]["download_url"],
            path,
            item["content_details"].get("sha256_hash"),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(worker, jobs))

    manifest = target / "source_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "source", "doi", "license", "approved"])
        for label in positives:
            base = label["filename"].rsplit(".", 1)[0]
            image_name = base + ".png"
            if (image_dir / image_name).exists() and (
                label_dir / label["filename"]
            ).exists():
                writer.writerow(
                    [
                        image_name,
                        label["filename"],
                        DATASET_URL,
                        metadata["doi"]["id"],
                        "CC BY 4.0",
                        "",
                    ]
                )
    return len(positives), target


def safe_extract_jpegs(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            path = Path(name)
            if (
                path.suffix.lower() != ".jpg"
                or path.is_absolute()
                or ".." in path.parts
            ):
                continue
            if not (path.parts[0].startswith("c") and len(path.parts[0]) == 2):
                continue
            out = destination / path
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.write_bytes(zf.read(name))


def collect_epfl(root: Path):
    """Download EPFL's parking/campus frames and convert car/person boxes to YOLO."""
    archive_dir = root / "source_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    images_zip = archive_dir / "epfl_multiclass_images.zip"
    labels_zip = archive_dir / "epfl_multiclass_annotations.zip"
    if not images_zip.exists():
        download(EPFL_IMAGES_URL, images_zip)
    if not labels_zip.exists():
        download(EPFL_LABELS_URL, labels_zip)

    source = root / "parking_lot_epfl" / "source_images"
    safe_extract_jpegs(images_zip, source)
    annotations = zipfile.ZipFile(labels_zip)
    out_images = root / "parking_lot_epfl" / "images"
    out_labels = root / "parking_lot_epfl" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    grouped = {}
    # Bus annotations are intentionally omitted: the assignment uses car/person/cart.
    class_dirs = {"car": 0, "person": 1}
    import re

    for name in annotations.namelist():
        match = re.search(
            r"gt_files242_(car|person)/visible_frame/det_frame(\d+)_cam(\d+)\.txt$",
            name,
        )
        if not match:
            continue
        class_name, frame_s, camera_s = match.groups()
        frame, camera = int(frame_s), int(camera_s)
        image_id = f"frame{frame:03d}_cam{camera}"
        image_path = source / f"c{camera}" / f"{frame:08d}.jpg"
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            image_w, image_h = img.size
        boxes = grouped.setdefault(
            image_id, {"image": image_path, "w": image_w, "h": image_h, "labels": []}
        )
        for line in annotations.read(name).decode("utf-8").splitlines():
            vals = line.split()
            if len(vals) != 4:
                continue
            xmin, ymin, xmax, ymax = map(float, vals)
            xmin = min(max(xmin, 0.0), image_w)
            xmax = min(max(xmax, 0.0), image_w)
            ymin = min(max(ymin, 0.0), image_h)
            ymax = min(max(ymax, 0.0), image_h)
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = ((xmin + xmax) / 2) / image_w
            cy = ((ymin + ymax) / 2) / image_h
            bw = (xmax - xmin) / image_w
            bh = (ymax - ymin) / image_h
            boxes["labels"].append(
                f"{class_dirs[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

    for image_id, item in grouped.items():
        if not item["labels"]:
            continue
        image_out = out_images / f"{image_id}.jpg"
        label_out = out_labels / f"{image_id}.txt"
        if not image_out.exists():
            image_out.write_bytes(item["image"].read_bytes())
        label_out.write_text("\n".join(item["labels"]) + "\n", encoding="utf-8")
    return len(list(out_images.glob("*.jpg"))), root / "parking_lot_epfl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent / "images" / "training_candidates",
    )
    parser.add_argument(
        "--carts", type=int, default=200, help="number of trolley-positive images"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    count, cart_dir = collect_carts(args.root, args.carts, args.seed)
    parking_count, parking_dir = collect_epfl(args.root)
    summary = {
        "classes": ["car", "person", "shopping_cart"],
        "car_person_source": str(parking_dir),
        "car_person_dataset": "EPFL Multi-view Multi-class Detection dataset",
        "car_person_dataset_url": "https://www.epfl.ch/labs/cvlab/data/data-multiclass/",
        "car_person_license_note": "EPFL states research use is permitted; cite EPFL/CVLab when reporting results.",
        "car_person_images_downloaded": parking_count,
        "shopping_cart_source": str(cart_dir),
        "shopping_cart_dataset": DATASET_URL,
        "shopping_cart_doi": "10.17632/vj9g4mn4vv.1",
        "shopping_cart_license": "CC BY 4.0",
        "shopping_cart_images_downloaded": count,
        "seed": args.seed,
        "note": "Candidates only. Run verify_training_images.py and approve every image before training.",
    }
    (args.root / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
