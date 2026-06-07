
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize Ultralytics/YOLO runs.
- Walks the given root (default: ./runs) and finds args.yaml/opt.yaml, hyp.yaml, results.csv
- Extracts: batch, imgsz, workers, optimizer, amp, lr0, lrf, box, cls, dfl
- Reads results.csv to find the last valid (non-NaN) epoch and losses/metrics
- Outputs a summary CSV and prints a concise table

Usage:
  python summarize_yolo_runs.py --root ./runs --out summary.csv

Tip:
  If you have multiple experiment roots, run this per root.
"""

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # We'll parse minimal YAML if PyYAML is missing

YOLO_ARG_FILENAMES = ["args.yaml", "opt.yaml"]
YOLO_HYP_FILENAMES = ["hyp.yaml", "hyps.yaml"]
YOLO_RESULTS_FILENAMES = ["results.csv"]

KEYS_ARGS = [
    "batch", "imgsz", "workers", "optimizer", "amp",
    "lr0", "lrf", "momentum", "weight_decay", "patience", "epochs", "device",
    "project", "name"
]
KEYS_HYP = ["box", "cls", "dfl"]
RESULT_COLS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
]

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        if yaml is not None:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    # Fallback: extremely naive parser (key: value on single lines)
    data: Dict[str, Any] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.strip().startswith("#"):
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() in {"true", "false"}:
                    data[k] = (v.lower() == "true")
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            data[k] = float(v)
                        else:
                            data[k] = int(v)
                    except Exception:
                        data[k] = v.strip("'\"")
    except Exception:
        return {}
    return data

def find_first_existing(d: Path, candidates) -> Optional[Path]:
    for fn in candidates:
        p = d / fn
        if p.exists():
            return p
    return None

def find_runs(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        files = set(filenames)
        if (set(YOLO_RESULTS_FILENAMES) & files) or (set(YOLO_ARG_FILENAMES) & files):
            yield Path(dirpath)

def read_results_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Normalize possible whitespace column names
        df.columns = [c.strip() for c in df.columns]
        # Ensure numeric
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
        return df
    except Exception:
        return pd.DataFrame()

def last_valid_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    # Prefer last row with a non-NaN in any of the RESULT_COLS
    mask = None
    for c in RESULT_COLS:
        if c in df.columns:
            m = df[c].notna()
            mask = m if mask is None else (mask | m)
    if mask is None:
        return df.iloc[-1]
    valid = df[mask]
    if valid.empty:
        return df.iloc[-1]
    # If duplicate epochs exist, keep last occurrence by natural order
    return valid.iloc[-1]

def summarize_run(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
    }
    arg_file = find_first_existing(run_dir, YOLO_ARG_FILENAMES)
    hyp_file = find_first_existing(run_dir, YOLO_HYP_FILENAMES)
    res_file = find_first_existing(run_dir, YOLO_RESULTS_FILENAMES)

    args = load_yaml(arg_file) if arg_file else {}
    hyps = load_yaml(hyp_file) if hyp_file else {}
    out["args_path"] = str(arg_file) if arg_file else ""
    out["hyp_path"] = str(hyp_file) if hyp_file else ""
    out["results_path"] = str(res_file) if res_file else ""

    for k in KEYS_ARGS:
        out[k] = args.get(k, "")
    for k in KEYS_HYP:
        out[k] = hyps.get(k, "")

    df = read_results_csv(res_file) if res_file else pd.DataFrame()
    last = last_valid_row(df)

    # Copy last metrics/losses if present
    for c in RESULT_COLS + ["epoch"]:
        out[c] = last.get(c, "") if isinstance(last, pd.Series) else ""

    # Quick flags
    out["has_nan_in_results"] = bool(df.isna().any().any()) if not df.empty else False
    out["epochs_logged"] = int(df.shape[0]) if not df.empty else 0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./runs", help="Root folder to scan")
    ap.add_argument("--out", type=str, default="summary.csv", help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows = []
    for d in find_runs(root):
        rows.append(summarize_run(d))

    if not rows:
        print(f"No runs found under: {root}")
        return

    df = pd.DataFrame(rows)

    # Reorder columns for readability
    pref = [
        "run_dir", "args_path", "hyp_path", "results_path",
        "project", "name", "optimizer", "device", "amp",
        "batch", "imgsz", "workers",
        "lr0", "lrf", "momentum", "weight_decay", "patience", "epochs",
        "box", "cls", "dfl",
        "epoch",
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "epochs_logged", "has_nan_in_results",
    ]
    cols = [c for c in pref if c in df.columns] + [c for c in df.columns if c not in pref]
    df = df[cols]

    out_path = Path(args.out).expanduser().resolve()
    df.to_csv(out_path, index=False, encoding="utf-8")
    # Pretty print a compact view
    view_cols = [c for c in [
        "run_dir","batch","imgsz","workers","optimizer","amp","lr0","lrf","box","cls","dfl",
        "epoch","train/box_loss","train/cls_loss","train/dfl_loss",
        "val/box_loss","val/cls_loss","val/dfl_loss","epochs_logged","has_nan_in_results"
    ] if c in df.columns]
    print(df[view_cols].to_string(index=False))
    print(f"\nSaved summary to: {out_path}")

if __name__ == "__main__":
    main()
