#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLO inference on a video, webcam, or RTSP stream.

Examples:
  python src/local/stream_infer.py --source demo.mp4 --weights best.pt
  python src/local/stream_infer.py --source 0 --weights best.pt --display
  python src/local/stream_infer.py --source rtsp://... --weights best.pt --sample-every 5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def parse_source(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def make_record(
    frame_id: int,
    video_ms: float,
    infer_ms: float,
    detections: list[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    return {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "frame_id": frame_id,
        "video_timestamp_ms": round(video_ms, 2),
        "infer_ms": round(infer_ms, 2),
        "num_detections": len(detections),
        "detections": detections,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Video path, RTSP URL, or webcam index such as 0")
    ap.add_argument("--weights", required=True, help="YOLO weights path, e.g. best.pt")
    ap.add_argument("--output-dir", default="outputs/stream_demo")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sample-every", type=int, default=1, help="Run inference every N frames")
    ap.add_argument("--display", action="store_true", help="Show annotated frames in a window")
    ap.add_argument("--save-video", action="store_true", help="Write annotated mp4 output")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "detections.jsonl"
    video_path = output_dir / "annotated.mp4"

    model = YOLO(args.weights)
    names = model.model.names if hasattr(model, "model") else {}

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if args.save_video:
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not determine frame size for video writer")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    frame_id = 0
    processed = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            if frame_id % max(1, args.sample_every) != 0:
                if writer is not None:
                    writer.write(frame)
                continue

            t0 = time.time()
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            infer_ms = (time.time() - t0) * 1000

            detections: list[dict[str, Any]] = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
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

            annotated = results[0].plot() if results else frame
            record = make_record(
                frame_id=frame_id,
                video_ms=cap.get(cv2.CAP_PROP_POS_MSEC),
                infer_ms=infer_ms,
                detections=detections,
                source=args.source,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            if writer is not None:
                writer.write(annotated)
            if args.display:
                cv2.imshow("wildfire-yolo-demo", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"[DONE] processed_frames={processed}")
    print(f"[JSONL] {jsonl_path}")
    if args.save_video:
        print(f"[VIDEO] {video_path}")


if __name__ == "__main__":
    main()

