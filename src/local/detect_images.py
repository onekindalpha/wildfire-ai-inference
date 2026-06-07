# scripts/detect_images.py
from pathlib import Path
from glob import glob
import json, time, os, cv2, torch, numpy as np
from ultralytics import YOLO

WEIGHTS   = os.getenv("YOLO_WEIGHTS", r".\weights\best.pt")
INPUT_DIR = Path(os.getenv("YOLO_INPUT_DIR", r".\datasets\test_images"))
OUT_JSON  = Path(os.getenv("YOLO_OUT_JSON", r".\outputs\predictions.jsonl"))
DEVICE    = 0 if torch.cuda.is_available() else "cpu"
IMGSZ     = int(os.getenv("YOLO_IMGSZ", "736"))
CONF      = float(os.getenv("GLOBAL_CONF", "0.40"))
IOU       = float(os.getenv("GLOBAL_IOU", "0.60"))

def imread(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(WEIGHTS)
    model.to(DEVICE)
    names = model.model.names if hasattr(model, "model") else {}

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob(str(INPUT_DIR / ext)))
    paths = sorted(paths)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        for p in paths:
            img = imread(p)
            t0 = time.time()
            r = model.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, device=DEVICE, verbose=False)
            dets = []
            if len(r) > 0 and r[0].boxes is not None:
                for b in r[0].boxes:
                    xyxy = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item())
                    cls  = int(b.cls[0].item())
                    dets.append({
                        "bbox": [round(x, 2) for x in xyxy],
                        "conf": round(conf, 4),
                        "cls": cls,
                        "label": names.get(cls, str(cls)),
                    })
            rec = {
                "image": p,
                "num": len(dets),
                "detections": dets,
                "infer_ms": round((time.time() - t0) * 1000, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[SAVED] {OUT_JSON}")

if __name__ == "__main__":
    main()
