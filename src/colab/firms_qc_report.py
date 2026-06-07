#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QC report for FIRMS/Himawari outputs.

- Scans OUT_DIR for AOIs.
- Counts rows in firms_points_raw.csv, himawari_keys.csv, slots_debug.csv.
- If Himawari keys are empty but FIRMS has rows, prints top 10 UTC 10-min slots to help debug minute alignment.
- If FIRMS is empty, prints a per-product count (if present) to hint which feeds were empty.
"""
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/content/drive/MyDrive/wildfire_ml/firms_h8_out_allfeeds_v2")  # change if needed

def main():
    base = Path(OUT_DIR)
    if not base.exists():
        raise SystemExit(f"OUT_DIR not found: {base}")

    aoi_dirs = [p for p in base.iterdir() if p.is_dir()]
    rows = []
    for aoi in sorted(aoi_dirs, key=lambda p: p.name.lower()):
        firms = aoi / "firms_points_raw.csv"
        h8    = aoi / "himawari_keys.csv"
        dbg   = aoi / "slots_debug.csv"

        firms_n = h8_n = dbg_n = 0
        has_ts  = False
        top_slots = None
        product_counts = None

        if firms.exists():
            try:
                df = pd.read_csv(firms)
                firms_n = len(df)
                has_ts = "acq_date" in df.columns and "acq_time" in df.columns
            except Exception as e:
                print(f"[WARN] {aoi.name}: read firms_points_raw.csv failed: {e}")

        if h8.exists():
            try:
                dh = pd.read_csv(h8)
                h8_n = len(dh)
            except Exception as e:
                print(f"[WARN] {aoi.name}: read himawari_keys.csv failed: {e}")

        if dbg.exists():
            try:
                dd = pd.read_csv(dbg)
                dbg_n = len(dd)
            except Exception as e:
                print(f"[WARN] {aoi.name}: read slots_debug.csv failed: {e}")

        # Diagnose when FIRMS exists but H8 keys are empty
        if firms_n > 0 and h8_n == 0 and has_ts:
            try:
                df = pd.read_csv(firms, usecols=["acq_date","acq_time"], dtype={"acq_time":"int64"})
                # build UTC timestamps and 10-min floors
                ts = []
                for _, r in df.iterrows():
                    hhmm = int(r["acq_time"])
                    h, m = divmod(hhmm, 100)
                    ts.append(pd.Timestamp(f"{r['acq_date']} {h:02d}:{m:02d}:00", tz="UTC"))
                s = pd.Series(ts).dt.floor("10min").value_counts().head(10)
                top_slots = [f"{k.isoformat()}Z: {v}" for k,v in s.items()]
            except Exception as e:
                top_slots = [f"(failed to compute slots: {e})"]

        # Diagnose when FIRMS is empty
        if firms_n == 0 and firms.exists():
            try:
                df = pd.read_csv(firms)  # empty frame but we can still show columns
                cols = [c for c in df.columns if c.lower() in ("instrument","satellite","sensor","product")]
                if cols:
                    product_counts = df[cols[0]].value_counts().to_dict()
            except Exception:
                product_counts = None

        rows.append({
            "AOI": aoi.name,
            "FIRMS_rows": firms_n,
            "H8_keys": h8_n,
            "slots_debug_rows": dbg_n,
            "top_10_utc_slots": "; ".join(top_slots) if top_slots else "",
            "hint_product_counts": product_counts or {},
        })

    rep = pd.DataFrame(rows).sort_values("AOI")
    out_csv = base / "qc_report.csv"
    rep.to_csv(out_csv, index=False)
    print("QC report written to:", out_csv)
    print(rep.to_string(index=False, max_colwidth=80))

if __name__ == "__main__":
    main()
