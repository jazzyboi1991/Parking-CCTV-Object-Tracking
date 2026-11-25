#!/usr/bin/env python3
"""Sample a source video and create reviewable YOLO label drafts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = (
    ROOT / "original videos/ViralHog-Shopping-Carts-Collide-with-Cars-original.mp4"
)
DEFAULT_MODEL = ROOT / "runs/detect/05-parking-video-finetune/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "images/video_finetune/original_person_review"
COLORS = {0: "red", 1: "lime", 2: "cyan"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    video = args.video if args.video.is_absolute() else ROOT / args.video
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not video.is_file():
        parser.error(f"video not found: {video}")
    if not model_path.is_file():
        parser.error(f"fine-tuned model not found: {model_path}")
    if args.samples < 1 or args.imgsz < 1 or not 0 < args.conf <= 1:
        parser.error("samples/imgsz must be positive and conf must be in (0, 1]")
    if output.exists():
        parser.error(f"output already exists; refusing to overwrite: {output}")

    frames_dir = output / "frames"
    labels_dir = output / "labels"
    overlays_dir = output / "overlays"
    for directory in (frames_dir, labels_dir, overlays_dir):
        directory.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if total < 1 or fps <= 0:
        capture.release()
        parser.error(f"could not read video metadata: {video}")

    # Avoid nearly identical neighboring samples while covering the entire video.
    indices = sorted(
        {round(i * (total - 1) / max(args.samples - 1, 1)) for i in range(args.samples)}
    )
    sampled: list[tuple[Path, float]] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        image_path = frames_dir / f"frame-{index:08d}.jpg"
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise OSError(f"could not write frame: {image_path}")
        sampled.append((image_path, index / fps))
    capture.release()
    if not sampled:
        raise SystemExit("No video frames could be sampled.")

    # COCO person detections seed the person labels; the project's model supplies
    # car and shopping-cart candidates. Every box remains a draft for human review.
    person_model = YOLO("yolo26n.pt")
    project_model = YOLO(str(model_path))
    for image_path, _timestamp in sampled:
        person_result = person_model.predict(
            source=str(image_path),
            classes=[0],
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        project_result = project_model.predict(
            source=str(image_path),
            classes=[0, 2],
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        width, height = project_result.orig_shape[1], project_result.orig_shape[0]
        rows: list[tuple[int, float, float, float, float]] = []
        if project_result.boxes is not None:
            for box, cls in zip(
                project_result.boxes.xywhn.tolist(), project_result.boxes.cls.tolist()
            ):
                class_id = int(cls)
                if class_id in (0, 2):
                    rows.append((class_id, *map(float, box)))
        if person_result.boxes is not None:
            for box in person_result.boxes.xywhn.tolist():
                rows.append((1, *map(float, box)))
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text(
            "".join(
                f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for cls, x, y, w, h in rows
            ),
            encoding="utf-8",
        )

        with Image.open(image_path) as source:
            overlay = source.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for number, (cls, x, y, w, h) in enumerate(rows, 1):
            box = (
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            )
            color = COLORS[cls]
            draw.rectangle(box, outline=color, width=3)
            draw.text(
                (box[0], box[1]),
                str(number),
                fill=color,
                stroke_width=1,
                stroke_fill="black",
            )
        overlay.save(overlays_dir / image_path.name, quality=95)

    (output / "timestamps.txt").write_text(
        "\n".join(f"{image.name}\t{seconds:.3f}s" for image, seconds in sampled) + "\n",
        encoding="utf-8",
    )
    print(f"sampled_frames={len(sampled)}")
    print(f"review_overlays={overlays_dir}")
    print(f"editable_labels={labels_dir}")
    print(
        "Review every candidate: correct boxes and add missed people, cars, and carts before training."
    )


if __name__ == "__main__":
    main()
