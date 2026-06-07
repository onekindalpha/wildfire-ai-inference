# Demo Service

This is a small FastAPI service for testing a trained YOLO wildfire model from a browser or API client.

## What It Demonstrates

- Upload an image.
- Run YOLO inference with `best.pt`.
- Return class, confidence, and bounding boxes.
- Save an annotated image.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Set the trained model path:

```bash
export YOLO_WEIGHTS=/path/to/best.pt
```

Start the service:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Use `POST /predict/image` to upload a test image.

## Test With curl

```bash
curl -X POST \
  -F "file=@/path/to/test_fire_image.jpg" \
  http://localhost:8000/predict/image
```

The response includes:

- `num_detections`
- `detections`
- `bbox_xyxy`
- `conf`
- `label`
- `annotated_url`

## Current Limitation

This service needs a wildfire-trained YOLO weight file. Base models such as `yolo11n.pt` or `yolov8n.pt` can test whether the API runs, but they are COCO-pretrained models and should not be used as evidence of wildfire/fire-smoke detection performance.

