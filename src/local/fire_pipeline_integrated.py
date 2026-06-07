#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-End: 경남 산청군(시천면 신천리 산39 일원) 2025-03-21 15:26 KST 산불
→ 스냅샷(위성영상) 수집 → dNBR 생성 → (선택) VIIRS 불점
→ COCO 데이터셋 구성(약식: burn_scar) → Mask R-CNN (Swin-T) 설정 파일 생성
→ 학습/추론 커맨드 안내까지 '한 파일'에서 자동화.

필수/권장 패키지
    pip install -U pip setuptools wheel
    # Core EO
    pip install pystac-client stackstac rioxarray xarray rasterio geopandas shapely pyproj numpy pandas requests tqdm
    # (선택) mmdetection 학습
    pip install -U openmim && mim install "mmengine>=0.10.4" && mim install "mmcv>=2.1.0" && mim install "mmdet>=3.3.0"
"""
from __future__ import annotations
import os, json, argparse, math, tempfile
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, mapping
import requests
from tqdm import tqdm

# Set environment variables for unsigned S3 access with rasterio/GDAL
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['AWS_S3_ALLOW_READ_WITHOUT_AWS_CONFIG'] = 'TRUE'

# EO & raster
import xarray as xr
import rioxarray as rxr
from pystac_client import Client
import stackstac
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

# ---------------------------
# Utils
# ---------------------------
KST = timezone.utc


def kst_to_utc(dt_kst_iso: str) -> datetime:
    """Convert 'YYYY-MM-DDTHH:MM:SS+09:00' or naive local KST to UTC datetime."""
    if "+" in dt_kst_iso[10:]:
        dt = datetime.fromisoformat(dt_kst_iso)
    else:
        dt = datetime.fromisoformat(dt_kst_iso + "+09:00")
    return dt.astimezone(timezone.utc)


def make_aoi(lat: float, lon: float, buffer_km: float = 5.0) -> gpd.GeoDataFrame:
    """Create circular AOI buffer in a local UTM CRS and return as GeoDataFrame."""
    utm_zone = math.floor((lon + 180) / 6) + 1
    utm_epsg = 32600 + utm_zone
    if lat < 0:
        utm_epsg = 32700 + utm_zone

    point_gdf = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs="EPSG:4326")
    point_utm = point_gdf.to_crs(f"EPSG:{utm_epsg}")
    aoi_utm = point_utm.buffer(buffer_km * 1000)
    return gpd.GeoDataFrame(geometry=aoi_utm, crs=f"EPSG:{utm_epsg}")


from shapely import affinity as _aff
import shapely

# ---------------------------
# STAC / Sentinel-2 helpers
# ---------------------------
EARTHSEARCH = "https://earth-search.aws.element84.com/v1"


def stac_s2_items(aoi_geojson: dict, dt_from: str, dt_to: str, cloud_lt: int = 40):
    """Search for Sentinel-2 items using STAC API."""
    cat = Client.open(EARTHSEARCH)
    search = cat.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{dt_from}/{dt_to}",
        intersects=aoi_geojson,
        query={"eo:cloud_cover": {"lt": cloud_lt}},
    )
    # FIX: Use items() instead of get_items()
    return list(search.items())


def pick_by_date(items, start: str, end: str):
    def within(i):
        d = i.datetime.date().isoformat()
        return start <= d <= end
    subset = [i for i in items if within(i)]
    subset.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    return subset[0] if subset else None


def load_bands_with_rioxarray(item, aoi: gpd.GeoDataFrame, assets: list[str]) -> xr.DataArray:
    """Load Sentinel-2 bands using rioxarray directly, clipped to AOI."""
    if item is None:
        return None

    try:
        # Remove '-jp2' suffix for asset access
        asset_names = [a.replace('-jp2', '') for a in assets]
        
        print(f"[INFO] Loading {item.id} with assets: {asset_names}")
        
        bands_list = []
        target_crs = aoi.crs
        bounds = aoi.total_bounds
        
        for asset_name in asset_names:
            if asset_name not in item.assets:
                print(f"[WARN] Asset {asset_name} not found in item {item.id}")
                continue
            
            href = item.assets[asset_name].href
            
            # Open with rioxarray
            band_data = rxr.open_rasterio(href, masked=True)
            
            # Squeeze band dimension if it exists (usually dimension 0)
            if 'band' in band_data.dims and len(band_data.band) == 1:
                band_data = band_data.squeeze('band', drop=True)
            
            # Clip to AOI bounds in original CRS first
            try:
                band_data = band_data.rio.clip_box(
                    minx=bounds[0], miny=bounds[1],
                    maxx=bounds[2], maxy=bounds[3],
                    crs=target_crs
                )
            except Exception as clip_err:
                print(f"[WARN] Clip failed for {asset_name}, trying reproject first: {clip_err}")
                # If clip fails, reproject first then clip
                band_data = band_data.rio.reproject(target_crs)
                band_data = band_data.rio.clip_box(
                    minx=bounds[0], miny=bounds[1],
                    maxx=bounds[2], maxy=bounds[3]
                )
            
            # Reproject to target resolution (10m) if not already
            if band_data.rio.resolution()[0] != 10:
                band_data = band_data.rio.reproject(
                    target_crs,
                    resolution=10,
                    resampling=Resampling.bilinear
                )
            
            bands_list.append(band_data)
        
        if not bands_list:
            raise ValueError("No bands successfully loaded")
        
        # Stack bands along new 'band' dimension
        da = xr.concat(bands_list, dim='band')
        da = da.assign_coords(band=assets)  # Use original names with -jp2 suffix
        
        print(f"[OK] Loaded shape: {da.shape}, bands: {list(da.coords['band'].values)}")
        return da

    except Exception as e:
        print(f"[ERROR] Failed to load bands for item {item.id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------
# FIRMS VIIRS helpers (optional)
# ---------------------------
FIRMS_CNTRY_TMPL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/country/csv/{API_KEY}/VIIRS_NOAA20_NRT/"
    "KOR/{DATE_FROM}/{DATE_TO}"
)


def get_viirs_csv(api_key: str, date_from: str, date_to: str, out_csv: Path) -> Path | None:
    if not api_key:
        return None
    url = FIRMS_CNTRY_TMPL.format(API_KEY=api_key, DATE_FROM=date_from, DATE_TO=date_to)
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"[WARN] FIRMS request failed: {r.status_code} {r.text[:160]}")
        return None
    out_csv.write_text(r.text, encoding="utf-8")
    return out_csv


def filter_viirs_by_aoi(csv_path: Path, center: tuple[float,float], radius_km=10.0) -> gpd.GeoDataFrame:
    if not csv_path or not csv_path.exists():
        return gpd.GeoDataFrame(columns=["latitude","longitude","acq_date","acq_time","frp","geometry"], geometry="geometry", crs=4326)
    df = pd.read_csv(csv_path)
    if df.empty:
        return gpd.GeoDataFrame(columns=["latitude","longitude","acq_date","acq_time","frp","geometry"], geometry="geometry", crs=4326)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    
    lat, lon = center[0], center[1]
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2*math.radians(lat)) + 1.175 * math.cos(4*math.radians(lat))
    m_per_deg_lon = 111_412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(3*math.radians(lat))
    deg_lat = (radius_km*1000)/m_per_deg_lat
    deg_lon = (radius_km*1000)/m_per_deg_lon
    circ = Point(lon, lat).buffer(1.0, resolution=64)
    circ = _aff.scale(circ, xfact=deg_lon, yfact=deg_lat, origin=(lon, lat))
    return gdf[gdf.geometry.within(circ)]


# ---------------------------
# COCO dataset builder
# ---------------------------

def dnbr_to_polygons(dnbr: xr.DataArray, thresh: float = 0.27, min_area_pix: int = 200) -> gpd.GeoDataFrame:
    """Convert dNBR to polygons where dnbr >= thresh."""
    from rasterio.features import shapes
    
    arr = dnbr.values
    arr = np.where(np.isfinite(arr), arr, -9999)
    mask = (arr >= thresh).astype("uint8")
    transform = dnbr.rio.transform()
    
    geoms = []
    with rasterio.Env():
        for geom, val in shapes(mask, mask==1, transform=transform):
            if val != 1:
                continue
            poly = shapely.geometry.shape(geom)
            if min_area_pix and (poly.area < min_area_pix):
                continue
            geoms.append(poly)
    
    if not geoms:
        print("[WARN] No polygons found above threshold")
        return gpd.GeoDataFrame(geometry=[], crs=dnbr.rio.crs)
    
    return gpd.GeoDataFrame(geometry=geoms, crs=dnbr.rio.crs)


def save_coco(geo_polys: gpd.GeoDataFrame, img_path: Path, out_json: Path, category_name="burn_scar"):
    """Create minimal COCO instance json."""
    from PIL import Image
    
    with Image.open(img_path) as im:
        width, height = im.size
    
    anns = []
    with rasterio.open(img_path) as src:
        for ann_id, geom in enumerate(geo_polys.to_crs(src.crs).geometry, start=1):
            if geom.is_empty:
                continue
            exterior = np.array(geom.exterior.coords)
            inv = ~src.transform
            px = [inv * (x, y) for x, y in exterior]
            seg = [c for xy in px for c in xy]
            if len(seg) < 6:
                continue
            xmin, ymin, xmax, ymax = geom.bounds
            bbox_px_min = inv * (xmin, ymin)
            bbox_px_max = inv * (xmax, ymax)
            x0, y0 = bbox_px_min
            x1, y1 = bbox_px_max
            w = max(1.0, x1 - x0)
            h = max(1.0, y1 - y0)
            anns.append({
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [seg],
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "iscrowd": 0,
                "area": float(geom.area),
            })
    
    coco = {
        "images": [{"id": 1, "file_name": img_path.name, "width": width, "height": height}],
        "categories": [{"id": 1, "name": category_name}],
        "annotations": anns,
    }
    out_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------
# MMDetection config writer
# ---------------------------

def write_mmdet_cfg(out_dir: Path, dataset_root: Path, cls_name: str = "burn_scar") -> Path:
    cfg = f'''\
_base_ = ["mmdet::mask_rcnn/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py"]
num_classes = 1

model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint="https://download.openmmlab.com/pretrain/third_party/swin_tiny_patch4_window7_224.pth")
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes),
    )
)

data_root = r"{dataset_root.as_posix()}/"
train_img_prefix = data_root + "firetrain/images/"
val_img_prefix   = data_root + "fireval/images/"
train_ann_file   = data_root + "firetrain/annotations.json"
val_ann_file     = data_root + "fireval/annotations.json"

metainfo = dict(classes=("{cls_name}",))

train_dataloader = dict(
    batch_size=2, num_workers=4,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_prefix),
        metainfo=metainfo
    )
)
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_prefix),
        metainfo=metainfo
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={{
        "absolute_pos_embed": dict(decay_mult=0.0),
        "relative_position_bias_table": dict(decay_mult=0.0),
        "norm": dict(decay_mult=0.0)}})
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_evaluator = dict(type="CocoMetric", ann_file=val_ann_file, metric=["bbox","segm"])
test_evaluator = val_evaluator
'''
    cfg_path = out_dir / "mask_rcnn_swin_t_fire.py"
    cfg_path.write_text(cfg, encoding="utf-8")
    return cfg_path


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    # 사용자 설정값 - 3월 21일 오후 3시 ~ 3월 30일 오후 1시
    LAT = 35.2469
    LON = 127.7927
    EVENT_TIME = "2025-03-21T15:00:00+09:00"  # 3월 21일 오후 3시 시작
    EVENT_END = "2025-03-30T13:00:00+09:00"   # 3월 30일 오후 1시 종료
    OUTPUT_DIR = "./sancheong_run"
    AOI_BUFFER_KM = 10.0  # 더 넓은 범위로 불점 검색
    FIRMS_API_KEY = os.getenv("FIRMS_MAP_KEY", "")  # set before running
    PRE_DAYS = 14
    POST_DAYS = 14
    DNBR_THRESHOLD = 0.27

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out/"data/firetrain/images").mkdir(parents=True, exist_ok=True)
    (out/"data/fireval/images").mkdir(parents=True, exist_ok=True)

    # 1) AOI + time window
    dt_utc_start = kst_to_utc(EVENT_TIME)
    dt_utc_end = kst_to_utc(EVENT_END)
    
    # FIRMS 검색 날짜 범위 (UTC)
    firms_from = dt_utc_start.date().isoformat()
    firms_to = dt_utc_end.date().isoformat()
    
    # Sentinel-2 검색 날짜 범위
    from_date = (dt_utc_start - pd.Timedelta(days=PRE_DAYS)).date().isoformat()
    to_date   = (dt_utc_end + pd.Timedelta(days=POST_DAYS)).date().isoformat()

    print(f"\n{'='*60}")
    print(f"산불 분석 설정:")
    print(f"  분석 기간: {EVENT_TIME} ~ {EVENT_END} (KST)")
    print(f"  FIRMS 검색: {firms_from} ~ {firms_to} (UTC)")
    print(f"  중심 좌표: {LAT}, {LON}")
    print(f"  검색 반경: {AOI_BUFFER_KM} km")
    print(f"{'='*60}\n")

    aoi = make_aoi(LAT, LON, buffer_km=AOI_BUFFER_KM)
    aoi_geojson = mapping(aoi.to_crs(4326).geometry[0])

    # 1.5) VIIRS 불점 데이터 먼저 수집
    viirs_gdf = None
    if FIRMS_API_KEY:
        print(f"[INFO] Fetching VIIRS fire points from FIRMS API...")
        csv_path = get_viirs_csv(FIRMS_API_KEY, firms_from, firms_to, out/"viirs_kor.csv")
        if csv_path:
            viirs_gdf = filter_viirs_by_aoi(csv_path, (LAT, LON), radius_km=AOI_BUFFER_KM)
            if not viirs_gdf.empty:
                viirs_path = out/"viirs_points.geojson"
                viirs_gdf.to_file(viirs_path, driver="GeoJSON")
                print(f"[OK] Found {len(viirs_gdf)} VIIRS fire points")
                
                # 불점 위경도 출력
                print(f"\n{'='*60}")
                print("🔥 감지된 불점 위치 (위도, 경도, 날짜, 시간, FRP):")
                print(f"{'='*60}")
                for idx, row in viirs_gdf.iterrows():
                    print(f"  {row['latitude']:.4f}, {row['longitude']:.4f} | "
                          f"{row['acq_date']} {row['acq_time']} | FRP: {row.get('frp', 'N/A')} MW")
                print(f"{'='*60}\n")
                
                # CSV로도 저장
                viirs_summary = viirs_gdf[['latitude', 'longitude', 'acq_date', 'acq_time', 'frp']].copy()
                viirs_summary.to_csv(out/"fire_points_summary.csv", index=False)
                print(f"[OK] Fire points saved: {out/'fire_points_summary.csv'}")
                print(f"[OK] GeoJSON saved: {viirs_path}\n")
            else:
                print("[INFO] No VIIRS points found within the specified area/time.\n")
    else:
        print("[WARN] FIRMS API key not provided. Skipping VIIRS fire point detection.\n")

    # 2) STAC search & pick
    print(f"[INFO] Searching Sentinel-2 from {from_date} to {to_date}")
    items = stac_s2_items(aoi_geojson, from_date, to_date, cloud_lt=60)
    if not items:
        raise SystemExit("No Sentinel-2 items found in the given window.")

    print(f"[INFO] Found {len(items)} Sentinel-2 items")
    pre = pick_by_date(items, from_date, dt_utc_start.date().isoformat())
    post = pick_by_date(items, dt_utc_end.date().isoformat(), to_date)
    
    if pre is None or post is None:
        print("[WARN] Could not find both pre and post images; trying full set.")
        if pre is None and items:
            pre = items[0]
        if post is None and len(items) > 1:
            post = items[-1]
    
    print(f"Pre-fire image:  {pre.id if pre else None}")
    print(f"Post-fire image: {post.id if post else None}\n")

    # 3) Compute NBR & dNBR
    aoi_projected = make_aoi(LAT, LON, buffer_km=AOI_BUFFER_KM)
    
    print("[INFO] Loading pre-fire bands...")
    nbr_pre  = load_bands_with_rioxarray(pre, aoi_projected, assets=["nir-jp2", "swir16-jp2"])
    print("[INFO] Loading post-fire bands...")
    nbr_post = load_bands_with_rioxarray(post, aoi_projected, assets=["nir-jp2", "swir16-jp2"])

    if nbr_pre is None or nbr_post is None:
        raise SystemExit("Failed to load pre or post fire bands.")

    # Compute NBR
    print("[INFO] Computing NBR...")
    nir_pre = nbr_pre.sel(band="nir-jp2").astype("float32")
    swir_pre = nbr_pre.sel(band="swir16-jp2").astype("float32")
    nbr_pre_calc = (nir_pre - swir_pre) / (nir_pre + swir_pre + 1e-6)

    nir_post = nbr_post.sel(band="nir-jp2").astype("float32")
    swir_post = nbr_post.sel(band="swir16-jp2").astype("float32")
    nbr_post_calc = (nir_post - swir_post) / (nir_post + swir_post + 1e-6)

    # Align and compute dNBR
    print("[INFO] Computing dNBR...")
    try:
        nbr_post_aligned = nbr_post_calc.rio.reproject_match(nbr_pre_calc)
        dnbr = nbr_pre_calc - nbr_post_aligned
        dnbr_path = out / "dnbr.tif"
        dnbr.rio.to_raster(dnbr_path)
        print(f"[OK] dNBR saved: {dnbr_path}")
    except Exception as e:
        print(f"[ERROR] Failed to compute dNBR: {e}")
        raise

    # 4) Save RGB preview
    print("[INFO] Loading RGB bands...")
    rgb_data = load_bands_with_rioxarray(post, aoi_projected, assets=["red-jp2","green-jp2","blue-jp2"])
    if rgb_data is None:
        raise SystemExit("Failed to load post fire RGB bands.")

    def norm_band(b):
        vmin = np.nanpercentile(b, 2)
        vmax = np.nanpercentile(b, 98)
        b = (b - vmin)/(vmax - vmin + 1e-6)
        return np.clip(b, 0, 1)

    rgbn = xr.concat([
        norm_band(rgb_data.sel(band="red-jp2")),
        norm_band(rgb_data.sel(band="green-jp2")),
        norm_band(rgb_data.sel(band="blue-jp2"))
    ], dim="band")

    rgbn = (rgbn*255).astype("uint8").transpose("band","y","x")
    rgb_path = out/"rgb_post.tif"
    rgbn.rio.to_raster(rgb_path)
    print(f"[OK] RGB saved: {rgb_path}")

    # 5) dNBR -> polygons
    print("[INFO] Extracting burn scar polygons...")
    polys = dnbr_to_polygons(dnbr, thresh=DNBR_THRESHOLD, min_area_pix=200)
    if not polys.empty:
        polys = polys.to_crs(4326)
    print(f"[INFO] Detected {len(polys)} burn scar polygons")

    # 6) COCO dataset
    print("[INFO] Creating COCO dataset...")
    train_img = out/"data/firetrain/images/scene_post.tif"
    val_img   = out/"data/fireval/images/scene_post_val.tif"
    import shutil
    shutil.copyfile(rgb_path, train_img)
    shutil.copyfile(rgb_path, val_img)

    save_coco(polys, train_img, out/"data/firetrain/annotations.json", category_name="burn_scar")
    save_coco(polys, val_img, out/"data/fireval/annotations.json", category_name="burn_scar")
    print("[OK] COCO annotations written.")

    # 7) MMDetection config
    cfg_path = write_mmdet_cfg(out, out/"data", cls_name="burn_scar")
    print(f"[OK] MMDetection config: {cfg_path}")

    # 8) Summary report
    print(f"\n{'='*60}")
    print("📊 분석 결과 요약:")
    print(f"{'='*60}")
    if viirs_gdf is not None and not viirs_gdf.empty:
        print(f"✓ VIIRS 불점: {len(viirs_gdf)}개 감지")
        print(f"  - CSV: {out/'fire_points_summary.csv'}")
        print(f"  - GeoJSON: {out/'viirs_points.geojson'}")
    else:
        print("✗ VIIRS 불점: 데이터 없음")
    print(f"✓ 소실 피해 폴리곤: {len(polys)}개")
    print(f"✓ dNBR 래스터: {out/'dnbr.tif'}")
    print(f"✓ RGB 영상: {out/'rgb_post.tif'}")
    print(f"✓ COCO 데이터셋: {out/'data'}")
    print(f"{'='*60}\n")
    
    # 9) Next steps
    cmds = f"""
# ==== Next steps ============================================
# 1) Install training dependencies (if not already installed)
python -m pip install -U openmim && mim install "mmengine>=0.10.4" && mim install "mmcv>=2.1.0" && mim install "mmdet>=3.3.0"

# 2) Train Mask R-CNN model
mim run mmdet train {cfg_path} --work-dir {out/'work_dirs/swin_t_fire'}

# 3) Run inference
mim run mmdet test {cfg_path} --checkpoint {out/'work_dirs/swin_t_fire/epoch_12.pth'}
# ===========================================================
"""
    print(cmds)
    print("\n[DONE] ✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
