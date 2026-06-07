# scripts/train_yolo.py
from pathlib import Path
import os
from ultralytics import YOLO

# ---- 경로/설정 ----
DATA_YAML = os.getenv("YOLO_DATA", r".\configs\mydataset.yaml")
OUT_DIR   = Path(os.getenv("YOLO_OUT_DIR", r".\runs\train\exp_swinless"))
MODEL     = os.getenv("YOLO_MODEL", "yolov8n.pt")  # 시작은 작은 모델 추천
IMGSZ     = int(os.getenv("YOLO_IMGSZ", "736"))
EPOCHS    = int(os.getenv("YOLO_EPOCHS", "50"))
BATCH     = int(os.getenv("YOLO_BATCH", "8"))
WORKERS   = int(os.getenv("YOLO_WORKERS", "0"))     # Windows면 0 권장(충돌 방지)
SEED      = int(os.getenv("YOLO_SEED", "27"))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL)  # coco-pretrained base

    # 학습
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        seed=SEED,
        project=str(OUT_DIR.parent),
        name=OUT_DIR.name,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        patience=20,
        save=True,
        save_period=5,
        device=0 if os.getenv("YOLO_DEVICE", "") == "" else os.getenv("YOLO_DEVICE"),
        val=True,
        # augmentation 예시 (필요 시 조정)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        mosaic=1.0, mixup=0.1, copy_paste=0.0,
        # class weights / focal loss 등은 추후 필요 시 커스텀
    )
    print(results)

    # 최종 가중치 경로 안내
    ckpt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[BEST] {ckpt} ")
    print("Use it in server/app.py via YOLO_WEIGHTS env var.")

if __name__ == "__main__":
    main()
