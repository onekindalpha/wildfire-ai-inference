
"""
Build FIRMS–Himawari8 (AHI) pairs for Korean wildfire incidents.

Usage:
  pip install pandas pytz requests boto3 botocore s3fs
  python build_firms_h8_dataset.py --firms_api_key YOUR_KEY

What it does:
  1) Pulls FIRMS active-fire points for three AOIs & date ranges you can edit
  2) Rounds each detection time to the nearest 10-min Himawari-8 full-disk slot
  3) Lists matching H8 objects in the public NOAA S3 bucket (B03, B07, B13 by default)
  4) Saves per-AOI CSVs and a manifest_summary.csv

Next step (not included here):
  - Download H8 data (--also_download) and tile to images; convert FIRMS points to YOLO boxes or masks.
"""

from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import io
import pandas as pd
import pytz
import requests

import boto3
from botocore import UNSIGNED
from botocore.client import Config

# -----------------------------
# Config & AOIs
# -----------------------------

KST = pytz.timezone("Asia/Seoul")

@dataclass
class AOI:
    name: str
    # bbox: (min_lon, min_lat, max_lon, max_lat)
    bbox: Tuple[float, float, float, float]
    # (start_date, end_date) local date strings (KST)
    start_date: str  # "YYYY-MM-DD"
    end_date: str    # "YYYY-MM-DD"


# Approximate bounding boxes—adjust as needed for your study areas.
AOIS = [
    AOI(
        name="Sancheong-Hadong",
        bbox=(127.5, 34.8, 128.2, 35.5),
        start_date="2025-03-21",
        end_date="2025-03-30",
    ),
    AOI(
        name="Uiseong-Andong-Cluster",
        bbox=(128.0, 36.0, 129.6, 37.2),  # Uiseong, Andong, Cheongsong, Yeongdeok, Yeongyang
        start_date="2025-03-22",
        end_date="2025-03-30",
    ),
    AOI(
        name="Daegu-Buk",
        bbox=(128.4, 35.8, 128.8, 36.2),
        start_date="2025-04-28",
        end_date="2025-05-01",
    ),
]


# -----------------------------
# FIRMS helpers
# -----------------------------

def fetch_firms_csv(
    api_key: str,
    start_date: str,
    end_date: str,
    instrument: str = "VIIRS_NRT",
    region: str = "world",
    conf: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull FIRMS CSV via area query (world, date range).
    Filters spatially later using bbox.
    instrument: "VIIRS_NRT" or "MODIS_NRT" etc.
    """
    base = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    url = f"{base}/{instrument}/{region}/{start_date}/{end_date}?key={api_key}"
    if conf:
        url += f"&confidence={conf}"  # optional
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df


def filter_by_bbox(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    return df[(df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
              & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)].copy()


def to_utc_timestamp(row: pd.Series) -> pd.Timestamp:
    # FIRMS fields: 'acq_date' (YYYY-MM-DD), 'acq_time' (HHMM, UTC)
    date = str(row["acq_date"])
    hhmm = int(row["acq_time"])
    hour, minute = divmod(hhmm, 100)
    ts = pd.Timestamp(f"{date} {hour:02d}:{minute:02d}:00", tz="UTC")
    return ts


def nearest_h8_10min(ts_utc: pd.Timestamp) -> pd.Timestamp:
    # Himawari-8 full disk cadence: every 10 minutes
    base = ts_utc.floor("10min")
    # decide nearest (floor vs floor+10)
    delta = abs((ts_utc - base).total_seconds())
    return base if delta <= 300 else (base + pd.Timedelta(minutes=10))


# -----------------------------
# H8 S3 listing (public)
# -----------------------------

def list_h8_keys_for_time(hh: int, mm: int, yyyy: int, mon: int, dd: int, bands: List[str]) -> List[str]:
    """
    Returns S3 keys for AHI-L1b-FLDK for the given time/bands.
    """
    prefix = f"AHI-L1b-FLDK/{yyyy:04d}/{mon:02d}/{dd:02d}/{hh:02d}{mm:02d}/"
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    page_it = paginator.paginate(Bucket="noaa-himawari8", Prefix=prefix)
    keys = []
    for page in page_it:
        for it in page.get("Contents", []):
            key = it["Key"]
            if any(b in key for b in bands):  # crude band filter by substring
                keys.append(key)
    return keys


def download_h8_keys(keys: List[str], out_dir: Path) -> None:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        fn = out_dir / key.split("/")[-1]
        if fn.exists():
            continue
        s3.download_file("noaa-himawari8", key, str(fn))


# -----------------------------
# Main orchestration
# -----------------------------

def run(firms_api_key: str, out_dir: Path, bands: List[str], also_download: bool, instrument: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    for aoi in AOIS:
        print(f"[AOI] {aoi.name}")
        # 1) FIRMS pull for date range (world), then spatial filter by bbox
        df = fetch_firms_csv(
            api_key=firms_api_key,
            start_date=aoi.start_date,
            end_date=aoi.end_date,
            instrument=instrument
        )

        needed = {"latitude", "longitude", "acq_date", "acq_time"}
        missing = needed - set(df.columns)
        if missing:
            print(f"  Warning: Missing columns in FIRMS response: {missing}")

        df_aoi = filter_by_bbox(df, aoi.bbox)
        if df_aoi.empty:
            print("  No FIRMS points in bbox/dates.")
            # still write empty CSVs for traceability
            aoi_dir = out_dir / aoi.name.replace(" ", "_")
            aoi_dir.mkdir(exist_ok=True, parents=True)
            df_aoi.to_csv(aoi_dir / "firms_points.csv", index=False)
            pd.DataFrame(columns=["utc_time", "s3_key"]).to_csv(aoi_dir / "himawari_keys.csv", index=False)
            continue

        df_aoi["ts_utc"] = df_aoi.apply(to_utc_timestamp, axis=1)
        df_aoi["h8_slot_utc"] = df_aoi["ts_utc"].apply(nearest_h8_10min)
        slot_times = sorted(pd.unique(df_aoi["h8_slot_utc"]))

        # 2) List H8 keys for each slot
        h8_rows = []
        for ts in slot_times:
            ts = pd.Timestamp(ts)  # ensure Timestamp
            y, m, d = ts.year, ts.month, ts.day
            h, mm = ts.hour, ts.minute
            keys = list_h8_keys_for_time(h, mm, y, m, d, bands=bands)
            for k in keys:
                h8_rows.append({"utc_time": ts, "s3_key": k})
        df_h8 = pd.DataFrame(h8_rows)

        # 3) Save per-AOI outputs
        aoi_dir = out_dir / aoi.name.replace(" ", "_")
        aoi_dir.mkdir(exist_ok=True, parents=True)
        df_aoi.to_csv(aoi_dir / "firms_points.csv", index=False)
        df_h8.to_csv(aoi_dir / "himawari_keys.csv", index=False)

        if also_download and not df_h8.empty:
            band_dir = aoi_dir / "H8_raw"
            print(f"  Downloading {len(df_h8)} H8 files to {band_dir} ...")
            download_h8_keys(df_h8["s3_key"].tolist(), band_dir)

        manifests.append({
            "aoi": aoi.name,
            "bbox": aoi.bbox,
            "start_date": aoi.start_date,
            "end_date": aoi.end_date,
            "firms_csv": str((aoi_dir / "firms_points.csv").resolve()),
            "h8_keys_csv": str((aoi_dir / "himawari_keys.csv").resolve()),
            "bands": ",".join(bands),
            "instrument": instrument,
        })

    pd.DataFrame(manifests).to_csv(out_dir / "manifest_summary.csv", index=False)
    print("Done. Wrote:", out_dir / "manifest_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--firms_api_key", required=True, help="Your FIRMS API key")
    parser.add_argument("--out_dir", default="firms_h8_out", help="Output directory")
    parser.add_argument("--bands", nargs="+", default=["B03", "B07", "B13"],
                        help="Himawari-8 band codes to include (e.g., B03 B07 B13)")
    parser.add_argument("--also_download", action="store_true", help="Also download H8 files from S3")
    parser.add_argument("--instrument", default="VIIRS_NRT", help="FIRMS instrument feed (e.g., VIIRS_NRT, MODIS_NRT)")
    args = parser.parse_args()

    run(
        firms_api_key=args.firms_api_key,
        out_dir=Path(args.out_dir),
        bands=args.bands,
        also_download=args.also_download,
        instrument=args.instrument
    )
