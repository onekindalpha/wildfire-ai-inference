# detect_worker.py
# -*- coding: utf-8 -*-
"""
Headless YOLO → Elasticsearch 파이프라인 워커
- 지정 폴더의 신규 이미지 감지
- YOLO(best.pt) 추론
- 어노테이션 이미지 저장(static/)
- 결과 메타데이터/박스/점수 Elasticsearch 인덱싱

사용:
  python detect_worker.py --images ./samples --weights ./best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
from dateutil.tz import tzlocal
from elasticsearch import Elasticsearch, helpers
from ultralytics import YOLO
# detect_worker.py

# 기본 이미지 폴더 경로를 직접 지정
DEFAULT_IMAGES_DIR = r"C:\Users\607-11\anaconda3\dev\project\dev\yolo_project\datasets\yoloseries\merged_dfire_y11\valid\images"   # ← 여기만 원하는 경로로 수정

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default=DEFAULT_IMAGES_DIR, help="감시할 이미지 폴더")
    ap.add_argument("--weights", type=str, default="C:\Users\607-11\anaconda3\dev\project\dev\yolo_project\datasets\yoloseries\yolo11\동결해제runs\11n_unfreeze_e12_800_b8_flameOS\weights\best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--index", type=str, default="wildfire_detections")
    args = ap.parse_args()

    # 나머지 코드는 동일...

# -----------------------------
# Config
# -----------------------------
@dataclass
class ESConfig:
    hosts: List[str] = None
    index: str = "wildfire_detections"
    verify_certs: bool = False
    api_key: Optional[Tuple[str, str]] = None  # (id, key), 보안 ON 시 사용

    def client(self) -> Elasticsearch:
        kwargs: Dict[str, Any] = dict(
            hosts=self.hosts or ["http://localhost:9200"],
            verify_certs=self.verify_certs,
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return Elasticsearch(**kwargs)


@dataclass
class WorkerConfig:
    images_dir: Path
    static_dir: Path = Path("./static")
    weights_path: Path = Path("./best.pt")
    imgsz: int = 960
    conf: float = 0.4
    device: Optional[str] = None  # "cuda:0" | "cpu" | None(auto)
    poll_sec: float = 2.0
    batch_size: int = 1
    save_annotated: bool = True


# -----------------------------
# Utilities
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_now_iso() -> str:
    return datetime.now(tzlocal()).isoformat()


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def draw_boxes(img_bgr, detections, class_names):
    for xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{class_names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            img_bgr, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA
        )
    return img_bgr


def to_es_doc(
    src_path: Path,
    detections: List[Tuple[List[float], float, int]],
    class_names: List[str],
    annotated_relpath: Optional[str],
) -> Dict[str, Any]:
    # 집계 필드
    flame_scores = [conf for _, conf, cid in detections if class_names[int(cid)] == "flame"]
    smoke_scores = [conf for _, conf, cid in detections if class_names[int(cid)] == "smoke"]

    doc = {
        "@timestamp": utc_now_iso(),              # Kibana timefield
        "local_time": local_now_iso(),
        "source_path": str(src_path.as_posix()),
        "annotated_path": annotated_relpath,     # static/ 이하 상대 경로
        "num_detections": len(detections),
        "classes": [class_names[int(c)] for _, _, c in detections],
        "scores": [float(s) for _, s, _ in detections],
        "bboxes_xyxy": [[float(v) for v in xy] for xy, _, _ in detections],
        # 요약 필드(대시보드/알림에 유용)
        "flame_max": max(flame_scores) if flame_scores else 0.0,
        "smoke_max": max(smoke_scores) if smoke_scores else 0.0,
        "has_flame": len(flame_scores) > 0,
        "has_smoke": len(smoke_scores) > 0,
    }
    return doc


# -----------------------------
# Main worker
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True, help="감시할 이미지 폴더")
    ap.add_argument("--weights", type=str, default="./best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--poll", type=float, default=2.0, help="폴링 주기(초)")
    ap.add_argument("--index", type=str, default="wildfire_detections")
    args = ap.parse_args()

    cfg = WorkerConfig(
        images_dir=Path(args.images),
        weights_path=Path(args.weights),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        poll_sec=args.poll,
    )
    es_cfg = ESConfig(index=args.index)

    assert cfg.images_dir.exists(), f"images_dir not found: {cfg.images_dir}"
    cfg.static_dir.mkdir(parents=True, exist_ok=True)

    # ES 클라이언트 준비 + 인덱스 보장
    es = es_cfg.client()
    ensure_index(es, es_cfg.index)

    # YOLO 모델 로드
    model = YOLO(str(cfg.weights_path))

    seen: set[str] = set()
    class_names = model.model.names if hasattr(model.model, "names") else {0: "obj"}

    print(f"[WORKER] Watching: {cfg.images_dir.resolve()}")
    print(f"[WORKER] Indexing to: {es_cfg.index} @ {es.transport.hosts}")

    while True:
        try:
            new_files = [
                p for p in sorted(cfg.images_dir.iterdir())
                if p.is_file() and is_image_file(p) and str(p) not in seen
            ]
            if not new_files:
                time.sleep(cfg.poll_sec)
                continue

            actions = []
            for img_path in new_files:
                seen.add(str(img_path))
                # 추론
                res = model.predict(
                    source=str(img_path),
                    imgsz=cfg.imgsz,
                    conf=cfg.conf,
                    device=cfg.device,
                    verbose=False,
                )
                dets: List[Tuple[List[float], float, int]] = []
                if res and res[0].boxes is not None:
                    for b in res[0].boxes:
                        xyxy = b.xyxy[0].tolist()
                        conf = float(b.conf[0].item())
                        cls_id = int(b.cls[0].item())
                        dets.append((xyxy, conf, cls_id))

                # 어노테이션 저장
                annotated_rel = None
                if cfg.save_annotated:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is not None:
                        img_bgr = draw_boxes(img_bgr, dets, class_names)
                        out_name = f"{img_path.stem}_annotated.jpg"
                        out_path = cfg.static_dir / out_name
                        cv2.imwrite(str(out_path), img_bgr)
                        annotated_rel = f"static/{out_name}"

                # ES 문서화
                doc = to_es_doc(img_path, dets, class_names, annotated_rel)
                actions.append({
                    "_index": es_cfg.index,
                    "_op_type": "index",
                    "_source": doc,
                })

            if actions:
                helpers.bulk(es, actions)
                print(f"[WORKER] Indexed {len(actions)} docs")
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1.0)


def ensure_index(es: Elasticsearch, index: str):
    if es.indices.exists(index=index):
        return
    # 간단 매핑 (Kibana time series에 유용)
    mapping = {
        "mappings": {
            "properties": {
                "@timestamp": {"type": "date"},
                "local_time": {"type": "date"},
                "source_path": {"type": "keyword"},
                "annotated_path": {"type": "keyword"},
                "num_detections": {"type": "integer"},
                "classes": {"type": "keyword"},
                "scores": {"type": "float"},
                "bboxes_xyxy": {"type": "float"},
                "flame_max": {"type": "float"},
                "smoke_max": {"type": "float"},
                "has_flame": {"type": "boolean"},
                "has_smoke": {"type": "boolean"},
            }
        }
    }
    es.indices.create(index=index, **mapping)


if __name__ == "__main__":
    main()
