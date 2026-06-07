# Data Pipeline Notes

## Collection

FIRMS active-fire points are collected by AOI and date range. The collector scripts support VIIRS/MODIS feeds and write per-AOI CSV files plus manifest summaries.

Relevant scripts:

- `src/colab/firms_h8_collector_v4.py`
- `src/colab/firms_h8_colab_allfeeds_v2.py`
- `src/colab/build_firms_h8_dataset.py`

## Time Slot Matching

Each FIRMS detection timestamp is rounded/matched to a Himawari 10-minute observation slot. The scripts list matching Himawari objects from public NOAA S3 buckets and store the matched keys for later download or image preparation.

## Label Generation

Two label-building paths are represented in the repository:

1. Point buffer path
   - FIRMS point
   - sensor-specific buffer radius
   - GeoJSON polygon
   - raster mask or COCO segmentation

2. Existing COCO conversion path
   - COCO annotation
   - coordinate validation/correction
   - YOLO detection labels
   - YOLO segmentation labels

Relevant scripts:

- `src/colab/firms_to_buffers_geojson.py`
- `src/colab/buffers_to_raster_masks.py`
- `src/colab/buffers_to_coco_timeslots.py`
- `scripts/prepare_usb_yolo_dataset.py`

## USB YOLO Dataset Cleanup

The USB `yolo.7z` dataset contains labels with class id `15`. In its `classes.txt`, class `15` maps to `fire`, so the local training cleanup remaps `15 -> 0` and builds a single-class `fire` dataset.

This cleanup is intentionally kept as a script instead of committing the extracted dataset:

```bash
python scripts/prepare_usb_yolo_dataset.py \
  --input data/external/yolo_usb \
  --output data/external/yolo_fire_clean
```

## Quality Checks

QC scripts inspect whether FIRMS rows, Himawari keys, masks, and annotation outputs line up as expected. This is useful because remote sensing data often fails in quiet ways: empty feed responses, wrong time slots, missing satellite keys, or coordinates outside image bounds.

Relevant scripts:

- `src/colab/firms_qc_report.py`
- `src/local/yolo_run_forensics.py`
- `src/local/summarize_yolo_runs.py`
