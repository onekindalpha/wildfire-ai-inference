#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the USB wildfire YOLO dataset for local training.

The archived dataset stores labels with class id 15, which maps to `fire`
in its `classes.txt`. This script remaps class 15 to class 0 and builds a
single-class YOLO dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_split(src_root: Path, out_root: Path, split: str, prefix: str) -> tuple[int, int]:
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for label_path in sorted(src_labels.glob("*.txt")):
        stem = label_path.stem
        image_path = next((src_images / f"{stem}{ext}" for ext in IMAGE_EXTS if (src_images / f"{stem}{ext}").exists()), None)
        if image_path is None:
            continue

        remapped_lines: list[str] = []
        for line in label_path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = parts[0]
            if cls_id != "15":
                continue
            remapped_lines.append("0 " + " ".join(parts[1:]))

        if not remapped_lines:
            continue

        out_stem = f"{prefix}_{split}_{stem}"
        shutil.copy2(image_path, out_images / f"{out_stem}{image_path.suffix.lower()}")
        (out_labels / f"{out_stem}.txt").write_text("\n".join(remapped_lines) + "\n")
        image_count += 1
        label_count += len(remapped_lines)

    return image_count, label_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Extracted USB dataset root")
    ap.add_argument("--output", required=True, type=Path, help="Prepared YOLO dataset root")
    args = ap.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    sources = [
        ("wildfireData", args.input / "wildfireData"),
        ("wildfireData_1", args.input / "wildfireData_1"),
    ]
    summary: list[tuple[str, str, int, int]] = []
    for prefix, src in sources:
        for split in ("train", "valid", "test"):
            if (src / split / "labels").exists():
                images, labels = copy_split(src, args.output, split, prefix)
                summary.append((prefix, split, images, labels))

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output.resolve()}",
                "train: images/train",
                "val: images/valid",
                "test: images/test",
                "names:",
                "  0: fire",
            ]
        )
        + "\n"
    )

    for prefix, split, images, labels in summary:
        print(f"{prefix}/{split}: images={images} labels={labels}")
    print(f"data_yaml={data_yaml}")


if __name__ == "__main__":
    main()

