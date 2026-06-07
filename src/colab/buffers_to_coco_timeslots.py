#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buffers_to_coco_timeslots.py
----------------------------
GeoJSON 버퍼 폴리곤(예: firms_to_buffers_geojson.py 출력)을 **UTC 10분 슬롯** 단위로 묶어
**COCO 인스턴스 세그멘테이션** 포맷으로 내보냅니다.
(선택) 각 슬롯마다 **빈 PNG 캔버스**도 생성하여 Mask R-CNN 등에서 바로 사용할 수 있도록 합니다.

입력:
  --geojson <FeatureCollection> : 각 Feature는 geometry(Polygon/MultiPolygon)와
      properties에 최소한 다음 키를 포함해야 함:
        - ts_utc: 문자열 타임스탬프 (예: "2025-03-25 03:20:00+00:00")
        - instrument: "VIIRS_*" 또는 "MODIS_*"
        - confidence_norm: 0~100 (선택)
        - AOI, latitude, longitude 등은 있으면 attributes로 포함

그리드 정의(둘 중 하나):
  1) --reference_raster <.tif> : 해상도/크기/CRS/transform 상속
  2) --bbox W S E N --width W --height H : EPSG:4326 격자 생성

카테고리:
  - 기본: 단일 카테고리 "fire" (id=1)
  - --class_by_instrument 사용 시: "fire_modis"(id=1), "fire_viirs"(id=2)

출력:
  --out_dir/coco/
    - annotations/instances_timeslots.json
    - images/<slot_id>.png (옵션 --write_blank_images)
    - meta/grid.json (그리드/변환 정보)

설치(콜랩 1회):
  !pip -q install pandas shapely rasterio numpy pillow

예시:
  python buffers_to_coco_timeslots.py \
    --geojson "/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v2/buffers/all_aois_buffers.geojson" \
    --reference_raster "/content/drive/MyDrive/wildfire_ml/some_ref.tif" \
    --out_dir "/content/drive/MyDrive/wildfire_ml/coco_timeslots" \
    --class_by_instrument --write_blank_images
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
import rasterio
from rasterio.transform import from_bounds
from PIL import Image, ImageDraw


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, type=str, help="Input GeoJSON (FeatureCollection) from buffer step")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory root (creates coco/ subdirs)")

    # Grid choice
    ap.add_argument("--reference_raster", type=str, default=None, help="Reference GeoTIFF to inherit size/transform/CRS")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, help="west south east north (EPSG:4326)")
    ap.add_argument("--width", type=int, default=None, help="width if using bbox grid")
    ap.add_argument("--height", type=int, default=None, help="height if using bbox grid")

    # Categories
    ap.add_argument("--class_by_instrument", action="store_true", help="Split MODIS/VIIRS categories")

    # Images
    ap.add_argument("--write_blank_images", action="store_true", help="Write blank PNG per time slot")
    ap.add_argument("--bg_value", type=int, default=0, help="Background pixel value (0-255) for blank PNG")
    ap.add_argument("--fg_value", type=int, default=0, help="(unused) Foreground; annotations live only in JSON")

    # Filtering
    ap.add_argument("--min_area_px", type=float, default=10.0, help="Drop tiny polygons under this pixel area")
    ap.add_argument("--limit_slots", type=int, default=None, help="Process at most N earliest slots (debug)")

    return ap.parse_args()


def load_features(geojson_path: Path) -> List[Dict[str, Any]]:
    with geojson_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")
    return gj["features"]


def grid_from_reference(ref_path: Path):
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = str(src.crs) if src.crs else "EPSG:4326"
        width, height = src.width, src.height
    return transform, crs, width, height


def grid_from_bbox(bbox: Tuple[float, float, float, float], width: int, height: int):
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    crs = "EPSG:4326"
    return transform, crs, width, height


def lonlat_to_xy(transform: rasterio.Affine, lon: float, lat: float) -> Tuple[float, float]:
    # rasterio.transform.rowcol returns (row, col). We need x(col), y(row).
    col, row = ~transform * (lon, lat)
    return float(col), float(row)


def poly_to_pixels(poly, transform) -> List[List[float]]:
    # Returns list of flattened [x1,y1,...] arrays for each ring (only exteriors for COCO)
    polys = []
    if isinstance(poly, Polygon):
        candidates = [poly]
    elif isinstance(poly, MultiPolygon):
        candidates = list(poly.geoms)
    else:
        return polys

    for p in candidates:
        exterior = list(p.exterior.coords)
        xy = []
        for lon, lat in exterior:
            x, y = lonlat_to_xy(transform, lon, lat)
            xy.extend([x, y])
        if len(xy) >= 6:  # at least 3 points
            polys.append(xy)
    return polys


def coco_categories(split_by_instrument: bool):
    if not split_by_instrument:
        return [{"id": 1, "name": "fire", "supercategory": "fire"}]
    return [
        {"id": 1, "name": "fire_modis", "supercategory": "fire"},
        {"id": 2, "name": "fire_viirs", "supercategory": "fire"},
    ]


def category_id_for(instr: str, split: bool) -> int:
    if not split:
        return 1
    s = (instr or "").upper()
    if "MODIS" in s:
        return 1
    # VIIRS / SNPP / NOAA*
    return 2


def flatten_bbox(seg_xy: List[float]) -> Tuple[float, float, float, float]:
    xs = seg_xy[0::2]
    ys = seg_xy[1::2]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return x0, y0, (x1 - x0), (y1 - y0)


def main():
    args = parse_args()
    geojson_path = Path(args.geojson)
    out_root = Path(args.out_dir)
    anno_dir = out_root / "coco" / "annotations"
    img_dir = out_root / "coco" / "images"
    meta_dir = out_root / "coco" / "meta"
    for d in (anno_dir, img_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    feats = load_features(geojson_path)

    # Build grid
    if args.reference_raster:
        transform, crs, width, height = grid_from_reference(Path(args.reference_raster))
    else:
        if not (args.bbox and args.width and args.height):
            raise SystemExit("Provide --reference_raster OR (--bbox W S E N --width W --height H)")
        transform, crs, width, height = grid_from_bbox(tuple(args.bbox), int(args.width), int(args.height))

    # Save grid meta for reproducibility
    with (meta_dir / "grid.json").open("w", encoding="utf-8") as f:
        json.dump({
            "transform": list(transform),
            "crs": crs,
            "width": width,
            "height": height
        }, f)

    # Group by 10-min slot (floor assumed in ts_utc of previous step)
    # We still floor here to be safe.
    def to_slot(ts: str) -> str:
        try:
            t = pd.to_datetime(ts, utc=True)
            # floor to 10 min
            floored = (t - pd.Timedelta(minutes=t.minute % 10, seconds=t.second, microseconds=t.microsecond)).strftime("%Y%m%d_%H%M")
            return floored
        except Exception:
            return "unknown"

    slots = defaultdict(list)
    for ft in feats:
        props = ft.get("properties", {})
        slot = to_slot(props.get("ts_utc", ""))
        if slot == "unknown":
            continue
        slots[slot].append(ft)

    # Optional: limit number of slots processed
    all_slot_keys = sorted(slots.keys())
    if args.limit_slots is not None:
        all_slot_keys = all_slot_keys[: int(args.limit_slots)]

    # COCO skeleton
    coco = {
        "info": {"description": "FIRMS buffers per 10-min UTC slot", "version": "1.0"},
        "licenses": [],
        "categories": coco_categories(args.class_by_instrument),
        "images": [],
        "annotations": []
    }

    ann_id = 1
    img_id = 1

    # Optionally prepare blank image
    def ensure_blank(path: Path, w: int, h: int, bg: int):
        if path.exists():
            return
        img = Image.new("L", (w, h), color=int(bg))
        img.save(path)

    for slot in all_slot_keys:
        img_filename = f"{slot}.png"
        coco["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })
        if args.write_blank_images:
            ensure_blank(img_dir / img_filename, width, height, args.bg_value)

        for ft in slots[slot]:
            geom = ft.get("geometry")
            if not geom:
                continue
            poly = shape(geom)
            segs = poly_to_pixels(poly, transform)
            if not segs:
                continue

            for seg_xy in segs:
                # area and tiny filter
                # rough area in pixel units using polygon area formula
                xs = seg_xy[0::2]
                ys = seg_xy[1::2]
                if len(xs) < 3:
                    continue
                area = 0.0
                for i in range(len(xs)):
                    j = (i + 1) % len(xs)
                    area += xs[i] * ys[j] - xs[j] * ys[i]
                area = abs(area) / 2.0
                if area < float(args.min_area_px):
                    continue

                bbox = flatten_bbox(seg_xy)
                cat_id = category_id_for(ft.get("properties", {}).get("instrument", ""), args.class_by_instrument)
                attribs = {
                    "confidence_norm": ft.get("properties", {}).get("confidence_norm", None),
                    "instrument": ft.get("properties", {}).get("instrument", ""),
                    "AOI": ft.get("properties", {}).get("AOI", ""),
                    "ts_utc": ft.get("properties", {}).get("ts_utc", ""),
                }

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": [seg_xy],  # single exterior ring
                    "area": float(area),
                    "bbox": [float(b) for b in bbox],
                    "iscrowd": 0,
                    "attributes": attribs
                })
                ann_id += 1

        img_id += 1

    anno_path = anno_dir / "instances_timeslots.json"
    with anno_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    print("COCO annotations written:", anno_path)
    print("Images dir:", img_dir)

if __name__ == "__main__":
    main()
