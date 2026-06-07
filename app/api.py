#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal YOLO demo service for wildfire image/video inference."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO


WEIGHTS = os.getenv("YOLO_WEIGHTS", "models/best.pt")
OUTPUT_DIR = Path(os.getenv("DEMO_OUTPUT_DIR", "outputs/service_demo"))
UPLOAD_DIR = OUTPUT_DIR / "uploads"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
JSON_DIR = OUTPUT_DIR / "json"

for directory in (UPLOAD_DIR, ANNOTATED_DIR, JSON_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Wildfire YOLO Demo Service")
model: YOLO | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    weights_path = Path(WEIGHTS)
    if not weights_path.exists():
        model = None
        return
    model = YOLO(str(weights_path))


def require_model() -> YOLO:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"YOLO model is not loaded. Set YOLO_WEIGHTS to an existing file. Current: {WEIGHTS}",
        )
    return model


def detection_record(source_name: str, detections: list[dict[str, Any]], infer_ms: float) -> dict[str, Any]:
    return {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "infer_ms": round(infer_ms, 2),
        "num_detections": len(detections),
        "detections": detections,
    }


def parse_boxes(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": names.get(cls_id, str(cls_id)),
                "cls": cls_id,
                "conf": round(conf, 4),
                "bbox_xyxy": [round(v, 2) for v in xyxy],
            }
        )
    return detections


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "weights": WEIGHTS,
        "model_loaded": model is not None,
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    yolo = require_model()
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image extension")

    job_id = uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    annotated_path = ANNOTATED_DIR / f"{job_id}.jpg"
    json_path = JSON_DIR / f"{job_id}.json"

    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    results = yolo.predict(str(input_path), imgsz=960, conf=0.4, verbose=False)
    infer_ms = (time.time() - t0) * 1000
    result = results[0]
    names = yolo.model.names if hasattr(yolo, "model") else {}
    detections = parse_boxes(result, names)

    annotated = result.plot()
    cv2.imwrite(str(annotated_path), annotated)

    record = detection_record(file.filename or input_path.name, detections, infer_ms)
    record["annotated_url"] = f"/outputs/{annotated_path.name}"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return JSONResponse(record)


@app.get("/outputs/{filename}")
def get_output(filename: str) -> FileResponse:
    path = ANNOTATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(path)

