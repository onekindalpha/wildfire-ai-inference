# YOLO Wildfire Dataset

Generated from `/Users/velocitygoal/wildfire_google_drive/scripts_local/sancheong_run/data`.

Important note: the COCO annotation y-coordinates were outside the TIFF image height by exactly 2001 px, so labels here apply a `y - 2001` correction. Source files were not modified.

Folders:
- `detect/`: YOLO detection labels (`class x_center y_center width height`).
- `segment/`: YOLO segmentation polygon labels (`class x1 y1 x2 y2 ...`).

Summary:
| split | image | size | source anns | detect labels | segment labels | dropped |
|---|---:|---:|---:|---:|---:|---:|
| train | scene_post.png | 21x2001 | 13 | 13 | 13 | 0 |
| val | scene_post_val.png | 21x2001 | 13 | 13 | 13 | 0 |

Example commands:
```bash
yolo detect train data=yolo_dataset/detect/data.yaml model=yolo11n.pt imgsz=736 epochs=50
yolo segment train data=yolo_dataset/segment/data.yaml model=yolo11n-seg.pt imgsz=736 epochs=50
```
