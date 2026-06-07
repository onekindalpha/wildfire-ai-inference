# Wildfire AI Inference

산불 탐지 실험에서 사용한 FIRMS/Himawari 데이터 수집, 라벨 생성, YOLO 학습/추론, 서버 산출물을 정리한 저장소입니다.

초기 목표는 위성 기반 산불 이벤트 데이터를 모으고, 불점 좌표를 학습 가능한 라벨 형태로 변환한 뒤, YOLO 모델과 간단한 추론 워커로 이미지/영상 입력을 탐지하는 흐름을 만드는 것이었습니다.

## What This Repository Contains

- FIRMS active fire point 수집 스크립트
- Himawari 위성 시간 슬롯 매칭 및 manifest 생성 스크립트
- FIRMS point → buffer polygon → raster mask / COCO annotation 변환 스크립트
- COCO annotation → YOLO detection/segmentation label 변환 샘플
- YOLO 학습, 이미지 추론, 폴더 감시형 추론 워커
- Elasticsearch/Kibana 연동을 가정한 서버 출력 HTML 산출물
- 프로젝트 발표/포트폴리오 문서

## Data Processing Flow

```text
FIRMS fire points
    -> AOI/date filtering
    -> Himawari 10-minute slot matching
    -> sensor-aware buffer polygons
    -> raster masks or COCO segmentation
    -> YOLO labels
    -> YOLO training / inference
    -> annotated outputs + detection metadata
```

현재 레포에 있는 YOLO 샘플 데이터는 `data/yolo_dataset/`에 들어 있습니다. 원본 COCO annotation의 y좌표가 이미지 높이보다 2001px 밀려 있어, 변환 과정에서 `y - 2001` 보정을 적용했습니다. 자세한 내용은 `data/yolo_dataset/README.md`에 남겨두었습니다.

## Inference Approach

YOLO 모델이 준비되어 있으면 입력 방식은 크게 세 가지로 나눌 수 있습니다.

1. 이미지 배치 추론
   - `src/local/detect_images.py`
   - 폴더 안의 이미지들을 순회하며 YOLO 예측 결과를 JSONL로 저장합니다.

2. 폴더 감시형 준실시간 추론
   - `src/local/detect_worker.py`
   - 새 이미지가 들어오면 YOLO로 탐지하고, bounding box가 그려진 이미지를 저장한 뒤 Elasticsearch에 metadata를 색인합니다.

3. 영상/스트림 추론
   - CCTV, RTSP, mp4 같은 입력을 OpenCV로 frame 단위로 읽습니다.
   - 일정 FPS 또는 일정 간격으로 frame을 추출합니다.
   - 추출된 frame을 YOLO 모델에 넣고, 같은 detection schema로 저장합니다.
   - 실제 배포에서는 `detect_worker.py`를 확장하거나, stream reader가 frame 이미지를 watched folder에 저장하도록 구성할 수 있습니다.

예시:

```bash
export YOLO_WEIGHTS=/path/to/best.pt
export YOLO_INPUT_DIR=/path/to/images
export YOLO_OUT_JSON=outputs/predictions.jsonl
python src/local/detect_images.py
```

폴더 감시형 워커:

```bash
python src/local/detect_worker.py \
  --images /path/to/incoming_frames \
  --weights /path/to/best.pt \
  --imgsz 960 \
  --conf 0.4
```

## Repository Layout

```text
src/
  colab/        FIRMS, Himawari, buffer, mask, COCO 변환 스크립트
  local/        YOLO 학습/추론 및 로컬 파이프라인 스크립트

data/
  yolo_dataset/ 변환 검증용 YOLO 샘플 데이터

deployments/
  server_outputs/  서버/지도 시각화 HTML 산출물

docs/
  portfolio/       발표 및 포트폴리오 문서
  data_pipeline.md 데이터 처리 및 라벨 생성 과정 메모
  demo.md          YOLO 이미지/영상 데모 실행 방법
  inference.md     영상/스트림 추론 구성 메모
  service_demo.md  FastAPI 데모 서비스 실행 방법

app/
  api.py           이미지 업로드 기반 YOLO 데모 API

configs/       USB 재마운트 후 Docker/conda 설정 파일 보관 예정
models/        placeholder only; model weights are not committed
```

## Setup

```bash
python -m pip install -r requirements.txt
```

FIRMS API를 사용하는 스크립트는 환경변수로 key를 받습니다.

```bash
export FIRMS_MAP_KEY="your_firms_map_key"
```

## Training

Detection:

```bash
yolo detect train data=data/yolo_dataset/detect/data.yaml model=yolo11n.pt imgsz=736 epochs=50
```

Segmentation:

```bash
yolo segment train data=data/yolo_dataset/segment/data.yaml model=yolo11n-seg.pt imgsz=736 epochs=50
```

## Demo Service

학습된 `best.pt`가 있으면 FastAPI로 이미지 업로드 데모를 실행할 수 있습니다.

```bash
export YOLO_WEIGHTS=/path/to/best.pt
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000/docs`를 열고 `POST /predict/image`에 테스트 이미지를 업로드하면 됩니다.

## Notes

- `.pem`, `.env`, API key, private server credential은 커밋하지 않습니다.
- `.pt`, `.7z`, `.zip`, `.nc`, 대형 GeoTIFF는 기본적으로 GitHub 일반 커밋에서 제외합니다.
- USB에 있던 일부 설정 파일과 산출물은 현재 세션에서 마운트되지 않아 `ARTIFACTS.md`에 추가 확인 항목으로 남겨두었습니다.
