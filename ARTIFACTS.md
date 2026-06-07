# Artifact Inventory

## Included

| Artifact | Status | Location |
|---|---|---|
| YOLO sample dataset | Included | `data/yolo_dataset/` |
| Local YOLO/pipeline scripts | Included | `src/local/` |
| FIRMS/Himawari/data prep scripts | Included | `src/colab/` |
| Server HTML outputs | Included | `deployments/server_outputs/` |
| Portfolio PDF/PPTX from local backup | Included | `docs/portfolio/` |

## Confirm When USB Is Mounted

The USB volume was not visible under `/Volumes` during this cleanup pass, so these user-mentioned files were not copied yet:

| Expected file | Recommended repo destination | Git policy |
|---|---|---|
| `yolo-gpu.yml` | `configs/yolo-gpu.yml` | Commit |
| `yolo.yml` | `configs/yolo.yml` | Commit |
| `fire-detect.yml` | `configs/fire-detect.yml` | Commit |
| `elastic search.png` | `assets/elastic-search.png` | Commit |
| `yolo.7z` | external release / Git LFS only | Do not commit normally |
| `kim_g128_latlon.nc` | external data storage / Git LFS only | Do not commit normally |
| `wildfire-infer.pem` | local secret storage only | Never commit |
| `ai_detection_kimeungi_v1.pptx` | `docs/portfolio/` | Commit if size is acceptable |
| `251017_kwedth_portfolio_kimeungi_v1.pdf` | `docs/portfolio/` | Already included from local backup |

## Security Cleanup

Hardcoded FIRMS keys found in copied scripts were replaced with `FIRMS_MAP_KEY` environment-variable reads.

