# # yolo_run_forensics.py
# import os, glob, json, time, math, re
# from datetime import datetime
# from collections import defaultdict
# import pandas as pd

# # optional: torch/tensorboard import (있으면 사용, 없으면 건너뜀)
# try:
#     import torch
# except:
#     torch = None

# try:
#     from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# except:
#     EventAccumulator = None

# RUN_DIRS = [
# r"C:\Users\607-11\anaconda3\dev\project\dev\yolo_project\datasets\yoloseries\FASDD (2)\FASDD_yolo_resized\final_balanced_dataset\runs\multi_smoke_test\YOLOv8s",
# r"C:\Users\607-11\anaconda3\dev\project\dev\yolo_project\datasets\yoloseries\FASDD (2)\FASDD_yolo_resized\final_balanced_dataset\YOLOV8S_runs\multi_smoke_test\YOLOv8s",
# ]

# KEYS_INTEREST = {
#     "model","imgsz","epochs","batch","device","optimizer","lr0","lrf","momentum","weight_decay",
#     "warmup_epochs","warmup_momentum","warmup_bias_lr","seed",
#     "patience","save_period",
#     # aug
#     "mosaic","mixup","hsv_h","hsv_s","hsv_v","degrees","translate","scale","shear","perspective","flipud","fliplr"
# }

# def mtime(p):
#     try: return os.path.getmtime(p)
#     except: return None

# def walltime_to_iso(wt):
#     try: return datetime.fromtimestamp(wt).isoformat(sep=" ", timespec="seconds")
#     except: return None

# def scan_files(root):
#     files = []
#     for dirpath, _, filenames in os.walk(root):
#         for fn in filenames:
#             p = os.path.join(dirpath, fn)
#             lower = fn.lower()
#             kind = None
#             if lower.endswith(".pt"):
#                 kind = "ckpt"
#             elif lower.startswith("events.out.tfevents"):
#                 kind = "tfevents"
#             elif lower.startswith("results") and lower.endswith(".csv"):
#                 kind = "results"
#             elif lower in {"args.yaml","hyp.yaml","cfg.yaml"}:
#                 kind = "yaml"
#             if kind:
#                 files.append({"path":p, "kind":kind, "mtime":mtime(p)})
#     return files

# def read_yaml_text(p):
#     try:
#         import yaml
#         with open(p, "r", encoding="utf-8") as f:
#             return yaml.safe_load(f) or {}
#     except Exception:
#         return {}

# def read_ckpt_meta(p):
#     out = {"train_args":{}, "model_yaml":None}
#     if torch is None: return out
#     try:
#         ckpt = torch.load(p, map_location="cpu")
#         if isinstance(ckpt, dict):
#             out["train_args"] = ckpt.get("train_args", {}) or {}
#             m = ckpt.get("model", None)
#             # try get yaml
#             if hasattr(m, "yaml"):
#                 out["model_yaml"] = getattr(m, "yaml", None)
#             elif isinstance(m, dict):
#                 out["model_yaml"] = m.get("yaml")
#     except Exception:
#         pass
#     return out

# def parse_tb_file(p):
#     """Extract wall_time, scalars (lr/pg*), text summaries; hparams if present."""
#     if EventAccumulator is None:
#         return None
#     try:
#         ea = EventAccumulator(p, size_guidance={
#             'scalars': 0, 'images': 0, 'histograms': 0, 'tensors': 0, 'compressedHistograms': 0, 'audio': 0
#         })
#         ea.Reload()
#         wt0 = None
#         try:
#             wt0 = ea._first_event_timestamp  # earliest event wall_time
#         except Exception:
#             wt0 = None
#         scalars = {}
#         for tag in ea.Tags().get('scalars', []):
#             scalars[tag] = [(s.step, s.wall_time, s.value) for s in ea.Scalars(tag)]
#         texts = {}
#         for tag in ea.Tags().get('tensors', []):
#             # try decode small text tensors (ultralytics sometimes logs text)
#             try:
#                 ts = ea.Tensors(tag)
#                 if ts:
#                     texts[tag] = len(ts)  # count only (avoid heavy decode)
#             except Exception:
#                 pass
#         return {"wall_time": wt0, "scalars": scalars, "texts_count": texts}
#     except Exception:
#         return None

# def infer_warmup_lr_params(scalars):
#     """Heuristic: use lr/pg0 curve to infer warmup_epochs and lr0/lrf-ish."""
#     if not scalars or "lr/pg0" not in scalars: return {}
#     seq = scalars["lr/pg0"]
#     # step ~ epoch, value ~ lr
#     steps = [s for (s, wt, v) in seq]
#     vals  = [v for (s, wt, v) in seq]
#     if not steps: return {}
#     # warmup: initial linear rise (first monotonic increasing segment)
#     warm_len = 0
#     for i in range(1, min(20, len(vals))):
#         if vals[i] >= vals[i-1]:
#             warm_len += 1
#         else:
#             break
#     warmup_epochs_est = warm_len  # lower-bound estimate
#     lr0_est = max(vals[:max(1, warm_len+1)]) if vals else None
#     lrf_est = min(vals[int(len(vals)*0.9):]) if len(vals) > 5 else (vals[-1] if vals else None)
#     return {"warmup_epochs_est": warmup_epochs_est, "lr0_est": lr0_est, "lrf_est": lrf_est}

# def collect_all(run_root):
#     files = scan_files(run_root)
#     records = []
#     for f in files:
#         rec = {"path": f["path"], "kind": f["kind"], "mtime": f["mtime"], "wall_time": None, "meta":{}}
#         if f["kind"] == "yaml":
#             y = read_yaml_text(f["path"])
#             meta = {k:v for k,v in (y or {}).items() if k in KEYS_INTEREST}
#             rec["meta"]["yaml"] = meta
#         elif f["kind"] == "ckpt":
#             meta = read_ckpt_meta(f["path"])
#             ta = {k:v for k,v in meta.get("train_args",{}).items() if k in KEYS_INTEREST}
#             rec["meta"]["ckpt_train_args"] = ta
#         elif f["kind"] == "results":
#             try:
#                 df = pd.read_csv(f["path"])
#                 rec["meta"]["results_cols"] = list(df.columns)
#                 if "epoch" in df.columns:
#                     rec["meta"]["last_epoch_logged"] = int(df["epoch"].iloc[-1])
#             except Exception:
#                 pass
#         elif f["kind"] == "tfevents":
#             tb = parse_tb_file(f["path"])
#             if tb:
#                 rec["wall_time"] = tb.get("wall_time")
#                 lr_inf = infer_warmup_lr_params(tb.get("scalars",{}))
#                 rec["meta"]["lr_curve_infer"] = lr_inf
#         records.append(rec)
#     return records

# def choose_initial_config(records):
#     # collect candidates with source+time, then pick earliest
#     cand = []
#     for r in records:
#         base_time = r["wall_time"] or r["mtime"] or 9e18
#         # yaml keys
#         for src in ["yaml","ckpt_train_args"]:
#             if src in r["meta"]:
#                 for k,v in r["meta"][src].items():
#                     cand.append((k, v, base_time, src, r["path"]))
#         # lr curve inferred
#         if "lr_curve_infer" in r["meta"]:
#             for k,v in r["meta"]["lr_curve_infer"].items():
#                 if v is not None:
#                     cand.append((k, v, base_time, "lr_curve", r["path"]))
#     # pick earliest for each key
#     best = {}
#     history = defaultdict(list)
#     for k,v,t,src,p in sorted(cand, key=lambda x:x[2]):
#         if k not in best:
#             best[k] = {"value": v, "source": src, "time": t, "file": p}
#         history[k].append({"value":v, "source":src, "time":t, "file":p})
#     return best, history

# def main():
#     all_records = []
#     for root in RUN_DIRS:
#         all_records += collect_all(root)
#     best, hist = choose_initial_config(all_records)

#     summary = {
#         "initial_config": {
#             k: {
#                 "value": v["value"],
#                 "source": v["source"],
#                 "first_seen_at": walltime_to_iso(v["time"]) if v["time"] else None,
#                 "file": v["file"]
#             } for k,v in best.items()
#         },
#         "notes": {
#             "warning": "lr*_est 값은 곡선 기반 추정이므로 참고용",
#             "confidence": "yaml > ckpt_train_args > lr_curve(추정) 순"
#         }
#     }

#     os.makedirs("./forensics_out", exist_ok=True)
#     with open("./forensics_out/initial_config.json","w",encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

#     # timeline
#     rows = []
#     for k, entries in hist.items():
#         for e in entries:
#             rows.append({
#                 "key": k,
#                 "value": e["value"],
#                 "source": e["source"],
#                 "seen_at": walltime_to_iso(e["time"]) if e["time"] else None,
#                 "file": e["file"]
#             })
#     df = pd.DataFrame(rows).sort_values(by=["key","seen_at"], na_position="last")
#     df.to_csv("./forensics_out/timeline.csv", index=False, encoding="utf-8-sig")

#     # also dump raw inventory
#     inv = []
#     for r in all_records:
#         inv.append({
#             "path": r["path"], "kind": r["kind"],
#             "mtime": walltime_to_iso(r["mtime"]) if r["mtime"] else None,
#             "wall_time": walltime_to_iso(r["wall_time"]) if r["wall_time"] else None,
#             "has_yaml": "yaml" in r["meta"],
#             "has_ckpt_args": "ckpt_train_args" in r["meta"],
#             "has_lr_curve": "lr_curve_infer" in r["meta"],
#             "last_epoch_logged": r["meta"].get("last_epoch_logged")
#         })
#     pd.DataFrame(inv).to_csv("./forensics_out/inventory.csv", index=False, encoding="utf-8-sig")

#     print("Wrote: forensics_out/initial_config.json, timeline.csv, inventory.csv")

# if __name__ == "__main__":
#     main()

import torch, json
p = r"C:\Users\607-11\anaconda3\dev\project\dev\yolo_project\datasets\yoloseries\FASDD (2)\FASDD_yolo_resized\final_balanced_dataset\YOLOV8S_runs\multi_smoke_test\YOLOv8s\weights\best.pt"
ckpt = torch.load(p, map_location="cpu")
ta = ckpt.get("train_args", {})
keys = ["model","epochs","patience","batch","imgsz","optimizer","lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr","mosaic","mixup","fliplr","flipud","scale","translate","degrees","shear","perspective","hsv_h","hsv_s","hsv_v","seed","device","save_period"]
print(json.dumps({k:ta.get(k) for k in keys}, indent=2))
