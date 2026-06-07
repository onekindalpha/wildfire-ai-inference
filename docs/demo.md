# Demo Plan

가장 간단한 데모는 이미 학습된 YOLO weight를 사용해 이미지/영상/웹캠 입력에서 flame 또는 smoke detection 결과를 보여주는 것입니다.

## Demo Option 1: Image Batch

준비물:

- `best.pt`
- 테스트 이미지 폴더

실행:

```bash
export YOLO_WEIGHTS=/path/to/best.pt
export YOLO_INPUT_DIR=/path/to/test_images
export YOLO_OUT_JSON=outputs/predictions.jsonl
python src/local/detect_images.py
```

보여줄 것:

- 입력 이미지
- JSONL detection 결과
- class, confidence, bbox

## Demo Option 2: Video Inference

준비물:

- `best.pt`
- 산불/연기 영상 또는 일반 테스트 영상

실행:

```bash
python src/local/stream_infer.py \
  --source /path/to/demo_video.mp4 \
  --weights /path/to/best.pt \
  --output-dir outputs/video_demo \
  --sample-every 3 \
  --save-video
```

결과:

- `outputs/video_demo/detections.jsonl`
- `outputs/video_demo/annotated.mp4`

이 방식이 포트폴리오 데모로 가장 보기 쉽습니다. 영상이 재생되고, flame/smoke box가 찍힌 결과물을 바로 보여줄 수 있습니다.

## Demo Option 3: Webcam / RTSP

웹캠:

```bash
python src/local/stream_infer.py \
  --source 0 \
  --weights /path/to/best.pt \
  --display
```

RTSP:

```bash
python src/local/stream_infer.py \
  --source rtsp://user:password@host:554/stream \
  --weights /path/to/best.pt \
  --sample-every 5
```

## Demo Storyline

1. FIRMS/Himawari 기반으로 산불 이벤트 후보와 학습 데이터를 구성했다.
2. 라벨을 YOLO 형식으로 정리하고 모델을 학습했다.
3. 학습된 모델을 이미지/영상 frame에 적용했다.
4. 탐지 결과는 bbox, confidence, class metadata로 저장된다.
5. 같은 event schema를 Elasticsearch/Kibana 또는 별도 dashboard에 연결할 수 있다.

## Recommended Portfolio Demo

가장 안정적인 구성:

```text
demo_video.mp4
    -> stream_infer.py
    -> annotated.mp4
    -> detections.jsonl
```

실시간성까지 보여주고 싶으면:

```text
webcam/rtsp
    -> stream_infer.py --display
    -> detection event log
```

라이브 데모는 환경 영향을 많이 받기 때문에, 발표/면접용으로는 `annotated.mp4`를 먼저 준비해두는 편이 안전합니다.

