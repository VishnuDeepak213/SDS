# SDS Full Project

This repository contains the full person detection and tracking project, including:

- the Streamlit dashboard
- YOLO-based image and video analysis
- training and evaluation scripts
- dataset conversion helpers
- model weights already present in the repo root

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, set the main file path to `streamlit_app.py`.
3. Keep `runtime.txt` and `requirements.txt` at the repository root.
4. Deploy.

## What Streamlit uses

- `streamlit_app.py` imports the real app from `person_yolo.web.app`
- `person_yolo/web/utils.py` loads local weights from the repo root when available
- `runtime.txt` pins Python 3.12.8 for stable package support

## Notes

- Large model files such as `yolo26n.pt` and `yolov8l.pt` are currently stored in the repo root so the app can load them directly.
- The dashboard supports image and video analysis, heatmaps, and tracking summaries.
- Training scripts remain under `person_yolo/scripts/` if you want to retrain or evaluate locally.