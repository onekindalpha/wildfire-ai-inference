# YOLO Inference Notes

이 프로젝트의 YOLO 추론은 이미지 입력을 기준으로 작성되어 있습니다. 영상이나 스트림 데이터도 내부적으로는 frame 이미지의 연속이기 때문에 같은 모델과 후처리 구조를 사용할 수 있습니다.

## Current Implementation

### Batch image inference

`src/local/detect_images.py`

- input folder에서 `.jpg`, `.jpeg`, `.png` 이미지를 읽음
- `YOLO_WEIGHTS`로 지정한 모델을 로드
- detection 결과를 `predictions.jsonl`에 저장
- 각 record에는 image path, detection count, bbox, confidence, class label, inference time이 포함됨

### Folder-watching inference worker

`src/local/detect_worker.py`

- 지정한 이미지 폴더를 polling
- 새 이미지가 들어오면 YOLO 추론 실행
- bounding box를 그린 annotated image 저장
- detection metadata를 Elasticsearch index에 저장

이 구조는 camera stream을 직접 붙이지 않아도, frame producer가 이미지를 폴더에 저장하면 준실시간 탐지 파이프라인처럼 동작할 수 있습니다.

## Stream/Video Extension

영상 입력을 처리하려면 다음 단계만 추가하면 됩니다.

```text
video / rtsp / webcam
    -> OpenCV VideoCapture
    -> frame sampling
    -> YOLO predict
    -> draw boxes
    -> save image or emit event
    -> Elasticsearch / dashboard / alert
```

구현 선택지는 두 가지입니다.

1. `detect_worker.py` 재사용
   - stream reader가 frame을 `incoming_frames/`에 저장
   - worker가 새 frame을 감지해서 기존 방식으로 추론
   - 장점: 기존 Elasticsearch 색인 로직을 그대로 사용

2. dedicated stream script 작성
   - `cv2.VideoCapture()`로 frame을 읽고 바로 `model.predict(frame)` 실행
   - 장점: latency가 낮고, frame skip / FPS 제어 / display 처리가 쉬움

## Minimal Stream Pseudocode

```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("rtsp://...")  # or video.mp4, 0 for webcam

frame_id = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame_id += 1
    if frame_id % 5 != 0:
        continue

    results = model.predict(frame, imgsz=960, conf=0.4, verbose=False)
    # parse results[0].boxes, save metadata, draw boxes, emit alert
```

## Detection Event Schema

`detect_worker.py` currently emits fields similar to:

- `@timestamp`
- `source_path`
- `annotated_path`
- `num_detections`
- `classes`
- `scores`
- `bboxes_xyxy`
- `flame_max`
- `smoke_max`
- `has_flame`
- `has_smoke`

For video, the same schema can be extended with:

- `stream_id`
- `frame_id`
- `video_timestamp_ms`
- `camera_location`
- `lat`
- `lon`

