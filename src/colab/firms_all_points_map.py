#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALL-DOTS MAP for firefighters (no dedup)

- Loads every AOI firms_points_raw.csv from OUT_DIR
- Keeps ALL detections (no dedup), optional min-confidence filter
- Builds two maps with pydeck (standalone HTML):
    1) Scatter (colored by instrument or confidence)
    2) Heatmap (density)
- Time is encoded in tooltip; you can also filter by date range

Usage:
    python firms_all_points_map.py
Edit USER SETTINGS below as needed.
"""
import pandas as pd
from pathlib import Path
import pydeck as pdk

# ===== USER SETTINGS =====
OUT_DIR = Path("/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v4")
OUTPUT_HTML_SCATTER = OUT_DIR / "all_points_scatter.html"
OUTPUT_HTML_HEATMAP = OUT_DIR / "all_points_heatmap.html"

# Filter
DATE_START = None   # e.g., "2025-03-21" or None
DATE_END   = None   # e.g., "2025-03-31" or None
MIN_CONFIDENCE = None  # e.g., 20 (0~100), None to include all

# Visualization:
COLOR_BY = "instrument"  # "instrument" or "confidence"
POINT_RADIUS = 200  # meters for scatter
# =========================

SENSOR_COLOR = {
    "VIIRS_NOAA21_NRT": [255, 80, 80],
    "VIIRS_NOAA21_SP":  [255, 120, 120],
    "VIIRS_SNPP_NRT":   [255, 170, 0],
    "VIIRS_SNPP_SP":    [255, 200, 80],
    "VIIRS_NOAA20_NRT": [120, 180, 255],
    "VIIRS_NOAA20_SP":  [160, 210, 255],
    "MODIS_NRT":        [160, 255, 120],
    "MODIS_SP":         [200, 255, 160],
}

def load_all_points(out_dir: Path) -> pd.DataFrame:
    rows = []
    for aoi_dir in sorted([p for p in out_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        f = aoi_dir / "firms_points_raw.csv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            df["__AOI__"] = aoi_dir.name
            rows.append(df)
        except Exception as e:
            print(f"[WARN] skip {aoi_dir.name}: {e}")
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

def to_ts_utc(row):
    try:
        t = int(row.get("acq_time"))
        h, m = divmod(t, 100)
        return pd.Timestamp(f"{row.get('acq_date')} {h:02d}:{m:02d}:00", tz='UTC')
    except Exception:
        return pd.NaT

def normalize_conf(row):
    # numeric if possible
    for col in ("confidence","confidence_text","confidence_level"):
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if 0 <= v <= 100:
                    return v
            except Exception:
                txt = str(row[col]).strip().upper()
                if txt in ("HIGH","H"): return 100.0
                if txt in ("NOMINAL","N"): return 70.0
                if txt in ("LOW","L"): return 30.0
    return 0.0

def color_for(row):
    if COLOR_BY == "instrument":
        return SENSOR_COLOR.get(str(row.get("instrument")), [200, 200, 200])
    # confidence -> ramp (blue->green->yellow->red)
    c = float(row.get("__conf__", 0))
    if c >= 90:  return [255, 80, 80]
    if c >= 70:  return [255, 200, 80]
    if c >= 40:  return [160, 220, 120]
    return [120, 180, 255]

def main():
    if not OUT_DIR.exists():
        raise SystemExit(f"OUT_DIR not found: {OUT_DIR}")

    df = load_all_points(OUT_DIR)
    if df.empty:
        print("No FIRMS points found under", OUT_DIR)
        return

    # Required columns
    if "latitude" not in df.columns and "lat" in df.columns:
        df = df.rename(columns={"lat": "latitude"})
    if "longitude" not in df.columns and "lon" in df.columns:
        df = df.rename(columns={"lon": "longitude"})
    if "instrument" not in df.columns and "product" in df.columns:
        df = df.rename(columns={"product": "instrument"})

    # timestamps and confidence
    df["__ts_utc__"] = df.apply(to_ts_utc, axis=1)
    df["__conf__"] = df.apply(normalize_conf, axis=1)

    # date/conf filter
    if DATE_START:
        df = df[df["__ts_utc__"] >= pd.Timestamp(DATE_START, tz='UTC')]
    if DATE_END:
        df = df[df["__ts_utc__"] <= pd.Timestamp(DATE_END + " 23:59:59", tz='UTC')]
    if MIN_CONFIDENCE is not None:
        df = df[df["__conf__"] >= float(MIN_CONFIDENCE)]

    df = df.dropna(subset=["latitude","longitude"]).copy()

    if df.empty:
        print("All points filtered out; nothing to map.")
        return

    # Color column for scatter
    df["__color__"] = df.apply(color_for, axis=1)

    # View state: center over mean of points
    try:
        center_lat = df["latitude"].astype(float).mean()
        center_lon = df["longitude"].astype(float).mean()
    except Exception:
        center_lat, center_lon = 36.3, 127.9

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6.2, bearing=0, pitch=0)

    # Scatter layer (all dots)
    scatter = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["longitude","latitude"],
        get_radius=POINT_RADIUS,
        get_fill_color="__color__",
        pickable=True,
        radius_min_pixels=1,
        radius_max_pixels=10,
        stroked=False,
        opacity=0.6,
    )

    tooltip = {
        "html": "<b>AOI:</b> {__AOI__}<br/>"
                "<b>Instrument:</b> {instrument}<br/>"
                "<b>Conf:</b> {__conf__}<br/>"
                "<b>UTC:</b> {__ts_utc__}<br/>"
                "<b>Lat/Lon:</b> {latitude}, {longitude}",
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}
    }

    r_scatter = pdk.Deck(
        layers=[scatter],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=tooltip,
    )
    r_scatter.to_html(str(OUTPUT_HTML_SCATTER), open_browser=False)
    print("Saved scatter map:", OUTPUT_HTML_SCATTER)

    # Heatmap (density)
    heat = pdk.Layer(
        "HeatmapLayer",
        df,
        get_position=["longitude","latitude"],
        get_weight="__conf__",  # confidence-weighted density
        aggregation=pdk.types.String("MEAN"),
        radius_pixels=40,
    )
    r_heat = pdk.Deck(
        layers=[heat],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "Density map (confidence-weighted)"},
    )
    r_heat.to_html(str(OUTPUT_HTML_HEATMAP), open_browser=False)
    print("Saved heatmap:", OUTPUT_HTML_HEATMAP)

if __name__ == "__main__":
    main()
