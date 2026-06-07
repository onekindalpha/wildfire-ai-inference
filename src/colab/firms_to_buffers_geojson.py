#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
firms_to_buffers_geojson.py
---------------------------------
FIRMS raw 포인트 CSV(무디듀프)를 **센서별 버퍼 반경(미터)** 로 둘러싼 Polygon으로 변환하여
**GeoJSON**(AOI별 + 전역)과 **CSV 매니페스트**를 생성합니다.
(훈련용: Mask R-CNN/세그멘테이션 전처리 단계로 활용)

기본 동작
- OUT_DIR 아래 모든 AOI/<firms_points_raw.csv>를 읽음
- 포인트별 UTC 타임스탬프, 정규화 신뢰도(conf_norm) 계산
- 센서별 반경(m)으로 현지 투영(AEQD) 버퍼 → WGS84 Polygon 좌표로 변환
- AOI별 GeoJSON과 전역 GeoJSON, 그리고 평면 CSV(manifest) 저장
- 기본: **디듀프 하지 않음**

설치(콜랩 1회)
    !pip -q install pandas shapely pyproj

실행 예시
    python firms_to_buffers_geojson.py \
        --out_dir "/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v2" \
        --min_conf 0 \
        --include_viirs --include_modis \
        --viirs_radius 375 --modis_radius 1000 \
        --points_per_circle 64

옵션
    --out_dir            결과가 있는 루트 폴더(각 AOI 하위에 firms_points_raw.csv)
    --min_conf           최소 신뢰도(0~100). None이면 필터 없음
    --include_viirs      VIIRS 계열 포함 (기본 True)
    --include_modis      MODIS 계열 포함 (기본 True)
    --viirs_radius       VIIRS 버퍼 반경(m) (기본 375)
    --modis_radius       MODIS 버퍼 반경(m) (기본 1000)
    --points_per_circle  버퍼 폴리곤 분할 포인트 개수(기본 64, 낮추면 용량↓)
    --aoi                특정 AOI 이름만 처리(여러 번 지정 가능). 미지정시 전체
    --instrument_like    instrument 부분일치 필터(여러 번 지정 가능)

출력
    <OUT_DIR>/buffers/
      ├─ aoi_<AOI>.geojson           # AOI별 버퍼 폴리곤
      ├─ all_aois_buffers.geojson    # 전체 AOI 합본
      └─ buffers_manifest.csv        # 평면 테이블(라벨/메타)

주의
- 정확한 미터 버퍼를 위해 포인트별로 **AEQD(방위방위등거리)** 로 로컬 투영 후 버퍼, 다시 WGS84로 역투영.
- FIRMS 컬럼 스키마가 다소 변형될 수 있어 latitude/longitude, product/instrument, confidence 등은
  가능한 한 자동 보정합니다.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from shapely.geometry import Point, mapping
from shapely.ops import transform
from pyproj import CRS, Transformer


# ---------------------- 유틸 ----------------------
SENSOR_GROUPS_VIIRS = {"VIIRS", "SNPP", "NOAA20", "NOAA21"}
SENSOR_PREFIXES = [
    "VIIRS_NOAA21_NRT","VIIRS_NOAA21_SP",
    "VIIRS_SNPP_NRT","VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT","VIIRS_NOAA20_SP",
    "MODIS_NRT","MODIS_SP"
]

def is_viirs(instr: str) -> bool:
    s = (instr or "").upper()
    return s.startswith("VIIRS_") or any(k in s for k in ("SNPP","NOAA20","NOAA21"))

def is_modis(instr: str) -> bool:
    s = (instr or "").upper()
    return s.startswith("MODIS_") or "MODIS" in s

def normalize_conf(row: pd.Series) -> float:
    # VIIRS numeric (0~100) 우선, MODIS 텍스트(H/N/L) 지원
    for col in ("confidence","confidence_text","confidence_level"):
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if 0 <= v <= 100:
                    return v
            except Exception:
                txt = str(row[col]).strip().upper()
                if txt in ("HIGH","H"):
                    return 100.0
                if txt in ("NOMINAL","N"):
                    return 70.0
                if txt in ("LOW","L"):
                    return 30.0
    return 0.0

def build_ts_utc(row: pd.Series):
    try:
        hhmm = int(row.get("acq_time"))
        h, m = divmod(hhmm, 100)
        return pd.Timestamp(f"{row.get('acq_date')} {h:02d}:{m:02d}:00", tz='UTC')
    except Exception:
        return pd.NaT

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    if "lat" in df.columns and "latitude" not in df.columns:
        ren["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        ren["lon"] = "longitude"
    if "product" in df.columns and "instrument" not in df.columns:
        ren["product"] = "instrument"
    if ren:
        df = df.rename(columns=ren)
    for col in ("latitude","longitude","instrument","acq_date","acq_time"):
        if col not in df.columns:
            df[col] = pd.NA
    return df

def local_buffer_polygon_wgs84(lat: float, lon: float, radius_m: float, n_points: int = 64):
    """
    포인트별 로컬 투영(AEQD, 중심 lat/lon)으로 정확한 미터 버퍼 후 WGS84로 역투영.
    """
    # 중심점 기준의 방위방위등거리(AEQD) 투영 (EPSG:4326 -> AEQD)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform

    pt = Point(float(lon), float(lat))  # note: shapely uses (x=lon, y=lat)
    poly = transform(fwd, pt).buffer(radius_m, resolution=max(4, int(n_points/4)))  # resolution controls circle smoothness
    poly_wgs84 = transform(inv, poly)
    return poly_wgs84


# ---------------------- 메인 로직 ----------------------
def process_aoi(aoi_dir: Path,
                viirs_radius: float,
                modis_radius: float,
                include_viirs: bool,
                include_modis: bool,
                min_conf: float = None,
                points_per_circle: int = 64,
                instrument_like: List[str] = None) -> Dict[str, Any]:
    csv_path = aoi_dir / "firms_points_raw.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {"features": [], "count": 0}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] {aoi_dir.name}: read failed: {e}")
        return {"features": [], "count": 0}

    if df.empty:
        return {"features": [], "count": 0}

    df = ensure_columns(df)
    df["__conf__"] = df.apply(normalize_conf, axis=1)
    df["__ts_utc__"] = df.apply(build_ts_utc, axis=1)

    # 필터: 센서/신뢰도/부분일치
    if not include_viirs:
        df = df[~df["instrument"].astype(str).apply(is_viirs)]
    if not include_modis:
        df = df[~df["instrument"].astype(str).apply(is_modis)]
    if min_conf is not None:
        df = df[df["__conf__"] >= float(min_conf)]
    if instrument_like:
        like = [s.upper() for s in instrument_like]
        df = df[df["instrument"].astype(str).str.upper().apply(lambda s: any(k in s for k in like))]

    df = df.dropna(subset=["latitude","longitude"])
    if df.empty:
        return {"features": [], "count": 0}

    feats = []
    for _, r in df.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        instr = str(r["instrument"] or "")
        radius_m = viirs_radius if is_viirs(instr) else modis_radius if is_modis(instr) else viirs_radius

        try:
            poly = local_buffer_polygon_wgs84(lat, lon, radius_m, n_points=points_per_circle)
            feat = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "AOI": aoi_dir.name,
                    "instrument": instr,
                    "confidence_norm": float(r["__conf__"]),
                    "acq_date": str(r.get("acq_date")),
                    "acq_time": str(r.get("acq_time")),
                    "ts_utc": str(r.get("__ts_utc__")),
                    "latitude": lat,
                    "longitude": lon,
                    "radius_m": float(radius_m)
                }
            }
            feats.append(feat)
        except Exception as e:
            # 점 하나 실패해도 계속 진행
            print(f"[WARN] {aoi_dir.name}: buffer failed at ({lat},{lon}) instr={instr}: {e}")
            continue

    return {"features": feats, "count": len(feats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Root directory that contains AOI subfolders with firms_points_raw.csv")
    ap.add_argument("--min_conf", type=float, default=None, help="Minimum confidence (0~100); None for no filter")
    ap.add_argument("--include_viirs", action="store_true", default=True, help="Include VIIRS family")
    ap.add_argument("--no_include_viirs", action="store_true", help="Disable VIIRS")
    ap.add_argument("--include_modis", action="store_true", default=True, help="Include MODIS family")
    ap.add_argument("--no_include_modis", action="store_true", help="Disable MODIS")
    ap.add_argument("--viirs_radius", type=float, default=375.0, help="Buffer radius for VIIRS (meters)")
    ap.add_argument("--modis_radius", type=float, default=1000.0, help="Buffer radius for MODIS (meters)")
    ap.add_argument("--points_per_circle", type=int, default=64, help="Polygon smoothness per circle")
    ap.add_argument("--aoi", action="append", default=None, help="Only process specific AOI folder names (repeatable)")
    ap.add_argument("--instrument_like", action="append", default=None, help="Substring filter for instrument (repeatable)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise SystemExit(f"OUT_DIR not found: {out_dir}")

    include_viirs = False if args.no_include_viirs else True
    include_modis = False if args.no_include_modis else True
    # (include_viirs/include_modis flags default True; user can disable with --no_*)

    aoi_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if args.aoi:
        allow = set(args.aoi)
        aoi_dirs = [p for p in aoi_dirs if p.name in allow]

    out_base = out_dir / "buffers"
    out_base.mkdir(parents=True, exist_ok=True)

    global_feats = []
    manifest_rows = []

    for aoi in sorted(aoi_dirs, key=lambda p: p.name.lower()):
        res = process_aoi(
            aoi_dir=aoi,
            viirs_radius=args.viirs_radius,
            modis_radius=args.modis_radius,
            include_viirs=include_viirs,
            include_modis=include_modis,
            min_conf=args.min_conf,
            points_per_circle=args.points_per_circle,
            instrument_like=args.instrument_like,
        )
        feats = res["features"]
        if not feats:
            print(f"[AOI] {aoi.name}: no features")
            continue

        # AOI GeoJSON
        aoi_geo = {"type": "FeatureCollection", "name": aoi.name, "features": feats}
        aoi_path = out_base / f"aoi_{aoi.name}.geojson"
        with aoi_path.open("w", encoding="utf-8") as f:
            json.dump(aoi_geo, f)

        print(f"[AOI] {aoi.name}: {len(feats)} polygons  -> {aoi_path.name}")
        global_feats.extend(feats)

        # manifest rows
        for ft in feats:
            prop = ft["properties"]
            manifest_rows.append({
                "AOI": prop.get("AOI"),
                "instrument": prop.get("instrument"),
                "confidence_norm": prop.get("confidence_norm"),
                "ts_utc": prop.get("ts_utc"),
                "latitude": prop.get("latitude"),
                "longitude": prop.get("longitude"),
                "radius_m": prop.get("radius_m")
            })

    # Global GeoJSON + CSV
    if global_feats:
        all_geo = {"type": "FeatureCollection", "name": "all_aois", "features": global_feats}
        all_path = out_base / "all_aois_buffers.geojson"
        with all_path.open("w", encoding="utf-8") as f:
            json.dump(all_geo, f)
        print(f"[ALL] {len(global_feats)} polygons  -> {all_path.name}")

        man_df = pd.DataFrame(manifest_rows)
        man_csv = out_base / "buffers_manifest.csv"
        man_df.to_csv(man_csv, index=False)
        print(f"[ALL] manifest rows: {len(man_df)} -> {man_csv.name}")
    else:
        print("[ALL] no features generated")

if __name__ == "__main__":
    main()
