# Wildfire AI Inference

Satellite/FIRMS-based wildfire data collection, dataset preparation, YOLO training, and inference artifact repository.

This repository is organized from local/USB project outputs for a GitHub-ready portfolio structure. It keeps source code, small verified artifacts, and documentation together while excluding private keys, large model weights, archives, and raw geospatial data.

## Project Scope

- Collect FIRMS wildfire detections and Himawari satellite time slots.
- Build wildfire/burn-scar datasets from geospatial buffers and raster masks.
- Convert available COCO-style annotations into YOLO detection/segmentation labels.
- Train or validate YOLO models for wildfire-related detection tasks.
- Preserve server/demo outputs such as Elasticsearch/Kibana-linked HTML maps.

## Repository Layout

```text
.
├── src/
│   ├── colab/              # FIRMS, Himawari, mask, COCO, and QC scripts
│   └── local/              # YOLO training/inference and integrated local pipeline scripts
├── data/
│   └── yolo_dataset/       # Small generated YOLO dataset sample
├── deployments/
│   └── server_outputs/     # Exported server/demo HTML outputs
├── docs/
│   └── portfolio/          # Portfolio PDF/PPTX artifacts
├── configs/                # Docker/conda/deployment configs when USB is mounted
├── models/                 # Placeholder only; weights are excluded from git
└── ARTIFACTS.md            # Artifact status and missing USB checklist
```

## Quick Start

Install the core Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Set your FIRMS key before running collector scripts:

```bash
export FIRMS_MAP_KEY="your_firms_map_key"
```

Train YOLO detection from the prepared sample dataset:

```bash
yolo detect train data=data/yolo_dataset/detect/data.yaml model=yolo11n.pt imgsz=736 epochs=50
```

Train YOLO segmentation:

```bash
yolo segment train data=data/yolo_dataset/segment/data.yaml model=yolo11n-seg.pt imgsz=736 epochs=50
```

## Notes

- Private SSH keys such as `wildfire-infer.pem` must never be committed.
- Model weights such as `.pt`, archives such as `.7z`/`.zip`, and large geospatial files such as `.nc` are ignored by default.
- The included YOLO labels were generated from a COCO annotation set whose y-coordinates were offset by exactly 2001 px. See `data/yolo_dataset/README.md`.

