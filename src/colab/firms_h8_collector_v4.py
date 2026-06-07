#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIRMS -> Himawari key collector (v4, robust)
- Pulls FIRMS (VIIRS+MODIS; NRT/SP selectable) using the Area CSV API
- Computes AHI 10-min slot per detection
- Lists Himawari-9 first, then Himawari-8 (anonymous S3)
- Tries minutes [-10, 0, +10, +20], then scans whole hour if needed
- Writes per-AOI:
    firms_points_raw.csv
    himawari_keys.csv  (full 'bucket/key')
    slots_debug.csv    (key counts per slot)
- Writes global manifest_summary.csv

Usage:
    python firms_h8_collector_v4.py
Edit the USER SETTINGS section below before running if needed.
"""

import os

# ========= USER SETTINGS =========
MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")  # set before running
OUT_DIR = "/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v4"  # absolute path for Colab/Drive

# Which FIRMS feeds to include:
INCLUDE_NRT = True
INCLUDE_SP  = True
VIIRS_ONLY  = False   # if True, ignore MODIS feeds

# AHI listing behavior
BANDS = None                  # None = list ALL bands; or e.g. ["B03","B07","B13"]
CHECK_NEIGHBORS = [-10, 0, +10, +20]
FALLBACK_SCAN_HOUR = True
ALSO_DOWNLOAD = False         # BIG files; set True if you really want raw downloads

# AOIs (edit as needed)
AOIS = [
    ("Sancheong-Hadong",        (127.5, 34.8, 128.2, 35.5), "2025-03-21", "2025-03-30"),
    ("Uiseong-Andong-Cluster",  (128.0, 36.0, 129.6, 37.2), "2025-03-22", "2025-03-30"),
    ("Daegu-Buk (Apr28–May1)",  (128.4, 35.8, 128.8, 36.2), "2025-04-28", "2025-05-01"),
    ("Ulju–Onyang/Yangsan",     (128.9, 35.25, 129.3, 35.70), "2025-03-22", "2025-03-27"),
    ("Gimhae",                  (128.7, 35.10, 129.0, 35.35), "2025-03-22", "2025-03-25"),
    ("Okcheon–Yeongdong",       (127.4, 36.00, 128.0, 36.50), "2025-03-23", "2025-03-25"),
    ("Gochang–Jeongeup",        (126.5, 35.30, 127.0, 35.70), "2025-03-25", "2025-03-25"),
    ("Ulju–Eonyang",            (129.0, 35.45, 129.3, 35.70), "2025-03-25", "2025-03-26"),
    ("Bonghwa",                 (128.5, 36.70, 129.0, 37.10), "2025-03-25", "2025-03-26"),
    ("Dalseong (Daegu)",        (128.3, 35.55, 128.6, 35.85), "2025-03-26", "2025-03-27"),
    ("Muju",                    (127.5, 35.80, 127.9, 36.10), "2025-03-26", "2025-03-27"),
    ("Daegu Buk (Apr6)",        (128.5, 35.90, 128.7, 36.05), "2025-04-06", "2025-04-06"),
    ("Hadong (Apr7–8)",         (127.6, 34.90, 127.9, 35.20), "2025-04-07", "2025-04-08"),
    ("Inje",                    (127.9, 37.90, 128.5, 38.30), "2025-04-26", "2025-04-27"),
]

# ----- deps (auto-install if needed) -----
import importlib, subprocess, sys, io, datetime as dt
def ensure(p):
    try:
        return importlib.import_module(p.split("==")[0].split(">=")[0].split("<")[0])
    except Exception:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
        return importlib.import_module(p.split("==")[0].split(">=")[0].split("<")[0])
pd       = ensure("pandas==2.2.2")
requests = ensure("requests")
boto3    = ensure("boto3>=1.34,<2")

from pathlib import Path
import boto3 as _boto3, botocore
from botocore.config import Config
from botocore.handlers import disable_signing

# feeds
INSTRUMENTS_VIIRS_NRT = ["VIIRS_SNPP_NRT","VIIRS_NOAA20_NRT","VIIRS_NOAA21_NRT"]
INSTRUMENTS_VIIRS_SP  = ["VIIRS_SNPP_SP","VIIRS_NOAA20_SP","VIIRS_NOAA21_SP"]
INSTRUMENTS_MODIS_NRT = ["MODIS_NRT"]
INSTRUMENTS_MODIS_SP  = ["MODIS_SP"]

# Prefer Himawari-9 (primary), then Himawari-8
BUCKETS = ["noaa-himawari9", "noaa-himawari8"]

def make_s3_client():
    try:
        return _boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))
    except Exception:
        s3 = _boto3.client("s3")
        s3.meta.events.register("choose-signer.s3.*", disable_signing)
        return s3
S3 = make_s3_client()

# ----- helpers -----
def daterange_chunks(start_date, end_date, max_days=10):
    d0 = dt.date.fromisoformat(start_date); d1 = dt.date.fromisoformat(end_date)
    cur = d0
    while cur <= d1:
        left = (d1 - cur).days + 1
        span = min(max_days, left)
        yield cur, span
        cur += dt.timedelta(days=span)

def fetch_area_csv(map_key, source, bbox, day_range, start_date):
    west,south,east,north = bbox
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{west},{south},{east},{north}/{day_range}/{start_date}"
    r = requests.get(url, timeout=120); r.raise_for_status()
    txt = r.text.strip()
    if (not txt) or txt.lower().startswith(("no data","error","invalid","map key")):
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(txt))
    except Exception:
        return pd.read_csv(io.StringIO(txt), skiprows=1)

def to_utc_timestamp(row):
    try:
        hhmm = int(row.get("acq_time"))
        h, m = divmod(hhmm, 100)
        return pd.Timestamp(f"{row.get('acq_date')} {h:02d}:{m:02d}:00", tz="UTC")
    except Exception:
        return pd.NaT

def list_ahi_keys(hh, mm, yyyy, mon, dd, bands=None, neighbors=(-10,0,10,20), fallback_scan_hour=True):
    """Return full 'bucket/key' strings for H9->H8. Try neighbor minutes and fallback scan."""
    tried = set(); keys = []
    for delta in neighbors:
        m = (mm + delta) % 60; tried.add(m)
        for bucket in BUCKETS:
            prefix = f"AHI-L1b-FLDK/{yyyy:04d}/{mon:02d}/{dd:02d}/{hh:02d}{m:02d}/"
            paginator = S3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for it in page.get("Contents", []):
                    k = it["Key"]
                    if bands is None or any(b in k for b in bands):
                        keys.append(f"{bucket}/{k}")
    if not keys and fallback_scan_hour:
        for m in [0,10,20,30,40,50]:
            if m in tried: continue
            for bucket in BUCKETS:
                prefix = f"AHI-L1b-FLDK/{yyyy:04d}/{mon:02d}/{dd:02d}/{hh:02d}{m:02d}/"
                paginator = S3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for it in page.get("Contents", []):
                        k = it["Key"]
                        if bands is None or any(b in k for b in bands):
                            keys.append(f"{bucket}/{k}")
    return keys

def download_ahi(keys, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for full in keys:
        bucket, key = full.split("/", 1)
        fn = out_dir / key.split("/")[-1]
        if fn.exists(): 
            continue
        S3.download_file(bucket, key, str(fn))

# ----- main -----
def main():
    out_base = Path(OUT_DIR); out_base.mkdir(parents=True, exist_ok=True)
    instruments = []
    if INCLUDE_NRT:
        instruments += INSTRUMENTS_VIIRS_NRT
        if not VIIRS_ONLY: instruments += INSTRUMENTS_MODIS_NRT
    if INCLUDE_SP:
        instruments += INSTRUMENTS_VIIRS_SP
        if not VIIRS_ONLY: instruments += INSTRUMENTS_MODIS_SP

    manifests = []

    for name, bbox, d0, d1 in AOIS:
        print(f"[AOI] {name}  {d0}..{d1}")
        aoi_dir = out_base / name.replace(" ", "_"); aoi_dir.mkdir(parents=True, exist_ok=True)

        # ---- FIRMS pulls ----
        dfs = []
        for src in instruments:
            for start, span in daterange_chunks(d0, d1, max_days=10):
                try:
                    df = fetch_area_csv(MAP_KEY, src, bbox, span, start.isoformat())
                    if not df.empty:
                        df["instrument"] = src
                        dfs.append(df)
                except Exception as e:
                    print(f"  WARN: {src} {start}+{span-1}d fetch failed: {e}")

        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        df_all.to_csv(aoi_dir/"firms_points_raw.csv", index=False)

        # ---- AHI key listing ----
        if df_all.empty:
            pd.DataFrame(columns=["utc_time","s3_key"]).to_csv(aoi_dir/"himawari_keys.csv", index=False)
            pd.DataFrame(columns=["slot_utc","keys_found","notes"]).to_csv(aoi_dir/"slots_debug.csv", index=False)
        else:
            df_all["ts_utc"] = df_all.apply(to_utc_timestamp, axis=1)
            df_all = df_all.dropna(subset=["ts_utc"]).copy()
            df_all["ahi_slot_utc"] = df_all["ts_utc"].dt.floor("10min")

            rows, dbg = [], []
            for ts in sorted(pd.unique(df_all["ahi_slot_utc"])):  # per 10-min slot
                ts = pd.Timestamp(ts)
                keys = list_ahi_keys(ts.hour, ts.minute, ts.year, ts.month, ts.day,
                                     bands=BANDS, neighbors=CHECK_NEIGHBORS,
                                     fallback_scan_hour=FALLBACK_SCAN_HOUR)
                for k in keys:
                    rows.append({"utc_time": ts, "s3_key": k})
                dbg.append({
                    "slot_utc": ts.isoformat(),
                    "keys_found": len(keys),
                    "notes": f"bands={'all' if BANDS is None else ','.join(BANDS)}; "
                             f"neighbors={CHECK_NEIGHBORS}; fallback={FALLBACK_SCAN_HOUR}"
                })
            pd.DataFrame(rows).to_csv(aoi_dir/"himawari_keys.csv", index=False)
            pd.DataFrame(dbg).to_csv(aoi_dir/"slots_debug.csv", index=False)

            if ALSO_DOWNLOAD and rows:
                print(f"  Downloading {len(rows)} AHI files …")
                download_ahi([r["s3_key"] for r in rows], aoi_dir/"H8_raw")

        manifests.append({
            "aoi": name, "bbox": bbox, "start_date": d0, "end_date": d1,
            "firms_csv": str((aoi_dir/'firms_points_raw.csv').resolve()),
            "h8_keys_csv": str((aoi_dir/'himawari_keys.csv').resolve()),
            "bands": "all" if BANDS is None else ",".join(BANDS),
            "feeds": ",".join(instruments)
        })

    pd.DataFrame(manifests).to_csv(Path(OUT_DIR)/"manifest_summary.csv", index=False)
    print("Done. Wrote:", Path(OUT_DIR)/"manifest_summary.csv")

if __name__ == "__main__":
    main()
