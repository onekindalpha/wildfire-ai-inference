
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epoch-by-epoch change report for YOLO results.csv

- Reads results.csv with columns like:
  epoch, train/box_loss, train/cls_loss, train/dfl_loss,
  val/box_loss, val/cls_loss, val/dfl_loss,
  metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
  lr/pg0, lr/pg1, lr/pg2, ...

- Computes deltas and percent changes per epoch (current - previous, and delta/previous).
- Saves a CSV with *_delta and *_pct columns.
- Also creates simple matplotlib line charts (one per figure) for LR, train losses, val losses, and metrics.

Usage:
  python epoch_change_report.py --results path/to/results.csv --out epoch_deltas.csv --plots out_dir

Notes:
  - NaN rows are kept but pct deltas will be NaN accordingly.
  - Duplicate epochs are allowed; the natural file order is preserved.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

GROUPS = {
    "lr": ["lr/pg0","lr/pg1","lr/pg2"],
    "train_losses": ["train/box_loss","train/cls_loss","train/dfl_loss"],
    "val_losses": ["val/box_loss","val/cls_loss","val/dfl_loss"],
    "metrics": ["metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)","metrics/mAP50-95(B)"],
}

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if c != "epoch"]
    for c in numeric_cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[f"{c}_delta"] = out[c].diff()
            out[f"{c}_pct"] = out[c].pct_change()
    return out

def plot_group(df: pd.DataFrame, cols: list, title: str, out_dir: Path):
    present = [c for c in cols if c in df.columns]
    if not present:
        return
    plt.figure()
    for c in present:
        plt.plot(df["epoch"], df[c], label=c)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True, help="Path to results.csv")
    ap.add_argument("--out", type=str, default="epoch_deltas.csv", help="Output CSV path")
    ap.add_argument("--plots", type=str, default="plots", help="Directory to save plots")
    args = ap.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    plots_dir = Path(args.plots).expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_path)
    out = add_deltas(df)
    out.to_csv(out_csv, index=False, encoding="utf-8")

    # Plots
    for group_name, cols in GROUPS.items():
        plot_group(df, cols, f"{group_name} over epochs", plots_dir)

    print(f"Saved delta report to: {out_csv}")
    print(f"Saved plots to: {plots_dir}")

if __name__ == "__main__":
    main()
