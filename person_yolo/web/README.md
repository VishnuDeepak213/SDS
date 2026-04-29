# Person Analytics Dashboard (Streamlit)

A lightweight web dashboard for human-only image and video analysis.

Features:
- Image: person counting and density heatmap visualization.
- Video: detection + per-frame metrics (count, density), tracking with the ability to select one person and highlight only that person through the whole video.

Quick start:

```
V:\SSD\.venv\Scripts\pip.exe install -r person_yolo/web/requirements.txt
V:\SSD\.venv\Scripts\streamlit.exe run person_yolo/web/app.py
```

Notes:
- The app auto-loads the latest `best.pt` under `runs/detect/**/weights/best.pt` if found; otherwise it falls back to `yolov8l.pt`.
- Adjust confidence and optional custom weights path from the sidebar.
- Outputs for selected-person videos are saved to a temporary directory and offered as a download.
