#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buffers_to_raster_masks.py
--------------------------
GeoJSON 버퍼 폴리곤(예: firms_to_buffers_geojson.py 출력)을 **라스터 마스크**로 변환합니다.

두 가지 방식 지원
1) 기준 래스터(GeoTIFF)를 참조: 해상도/크기/CRS/변환을 그대로 상속
   --reference_raster /path/to/reference.tif
2) BBOX + 사이즈로 신규 래스터 생성(EPSG:4326)
   --bbox west south east north --width 2000 --height 2000

버닝 옵션
- 기본은 value=1 (불 픽셀)
- --class_by_instrument: MODIS=1, VIIRS=2
- --burn_confidence: confidence_norm(0~100)을 0~255로 선형 스케일링하여 번(연속값)

출력
- GeoTIFF 기본 출력 (uint8)
- --write_png 추가 시 PNG로도 저장(팔레트 없음, 흑백/그레이)

설치(콜랩 1회):
    !pip -q install rasterio shapely numpy pillow

사용 예:
    # 기준 래스터 사용
    python buffers_to_raster_masks.py \
        --geojson "/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v2/buffers/all_aois_buffers.geojson" \
        --reference_raster "/content/drive/MyDrive/wildfire_ml/some_ref.tif" \
        --out_dir "/content/drive/MyDrive/wildfire_ml/masks" \
        --class_by_instrument

    # BBOX+사이즈
    python buffers_to_raster_masks.py \
        --geojson "/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v2/buffers/aoi_Sancheong-Hadong.geojson" \
        --bbox 127.5 34.8 128.2 35.5 --width 2200 --height 2200 \
        --out_dir "/content/drive/MyDrive/wildfire_ml/masks" \
        --burn_confidence --write_png
"""
import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
from shapely.geometry import shape, mapping
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image


def read_geojson(path: Path):
    with path.open('r', encoding='utf-8') as f:
        gj = json.load(f)
    if gj.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")
    return gj["features"]


def make_grid_from_bbox(bbox: Tuple[float, float, float, float], width: int, height: int):
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    crs = "EPSG:4326"
    return transform, crs, (height, width)


def grid_from_reference(ref_path: Path):
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = src.crs
        shape_hw = (src.height, src.width)
    return transform, crs, shape_hw


def burn_value(props: dict, class_by_instrument: bool, burn_confidence: bool) -> int:
    if burn_confidence:
        # confidence_norm (0~100) -> 0~255
        c = float(props.get("confidence_norm", 0.0))
        c = max(0.0, min(100.0, c))
        return int(round(c * 255.0 / 100.0))
    if class_by_instrument:
        instr = str(props.get("instrument", "")).upper()
        if "MODIS" in instr:
            return 1
        if "VIIRS" in instr or "SNPP" in instr or "NOAA20" in instr or "NOAA21" in instr:
            return 2
        return 1  # default to 1
    return 1  # simple binary mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, type=str, help="Input GeoJSON (FeatureCollection)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    # reference grid OR bbox grid
    ap.add_argument("--reference_raster", type=str, default=None, help="Reference GeoTIFF to inherit grid/CRS")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, help="west south east north (EPSG:4326)")
    ap.add_argument("--width", type=int, default=None, help="raster width (if using bbox)")
    ap.add_argument("--height", type=int, default=None, help="raster height (if using bbox)")

    # burn options
    ap.add_argument("--class_by_instrument", action="store_true", help="MODIS=1, VIIRS=2")
    ap.add_argument("--burn_confidence", action="store_true", help="Burn confidence 0~255 instead of 1/2 classes")

    # IO formats
    ap.add_argument("--write_png", action="store_true", help="Also write PNG alongside GeoTIFF")
    ap.add_argument("--tiff_name", type=str, default=None, help="Output GeoTIFF base name (default derives from input)")

    args = ap.parse_args()

    geojson_path = Path(args.geojson)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = read_geojson(geojson_path)
    if not feats:
        print("[INFO] No features; nothing to rasterize.")
        return

    # Grid
    if args.reference_raster:
        transform, crs, shape_hw = grid_from_reference(Path(args.reference_raster))
    else:
        if not (args.bbox and args.width and args.height):
            raise SystemExit("Provide --reference_raster OR (--bbox west south east north --width W --height H)")
        transform, crs, shape_hw = make_grid_from_bbox(tuple(args.bbox), args.width, args.height)

    height, width = shape_hw

    # Build (geometry, value) tuples
    shapes: List[Tuple[dict, int]] = []
    for ft in feats:
        geom = ft.get("geometry")
        if not geom:
            continue
        props = ft.get("properties", {})
        val = burn_value(props, args.class_by_instrument, args.burn_confidence)
        shapes.append((geom, val))

    # Rasterize (uint8)
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True  # be generous; firefighters prefer recall
    )

    # Output names
    base = args.tiff_name or geojson_path.stem
    tif_path = out_dir / f"{base}.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(mask, 1)

    print("Wrote GeoTIFF:", tif_path)

    if args.write_png:
        png_path = out_dir / f"{base}.png"
        Image.fromarray(mask).save(png_path)
        print("Wrote PNG:", png_path)


if __name__ == "__main__":
    main()
