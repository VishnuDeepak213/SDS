# Person Detection and Tracking (YOLOv8)

## Overview
- Converts WiderPerson to YOLO format (person-only).
- Trains YOLOv8-L with strong augmentations.
- Saves best weights under `runs/detect/yolo_person_wider/weights/best.pt`.

## 1) Convert WiderPerson to YOLO
- Wider root is expected at `V:/SSD/Datasets/wide` with folders `Images/`, `Annotations/`, and files `train.txt`, `val.txt`.
- Run:
  - `python v:/SSD/person_yolo/scripts/convert_wider.py --wider_root V:/SSD/Datasets/wide --out_root V:/SSD/person_yolo/datasets/wider_yolo`

Notes:
- Maps classes 1 (person) and 3 (head) to a single class `person`.
- Generates `train/val` with `images/` and `labels/` (empty labels kept for negatives).

## 2) Train YOLOv8-L (person-only)
- Ensure ultralytics and torch installed in your environment.
- Run:
  - `python v:/SSD/person_yolo/scripts/train.py`

Training settings:
- `epochs=300`, `imgsz=960`, `optimizer=AdamW`, `lr0=0.002`, `box=8.0`, `cls=0.3`
- Augs: mosaic=1.0, copy_paste=0.5, fliplr=0.5, hsv_h/s/v tuned.

## 3) Inference & Tracking (optional)
- Once trained, you can run Ultralytics' built-in tracker with ByteTrack using:
  - `yolo task=detect mode=track model=runs/detect/yolo_person_wider/weights/best.pt source=<video.mp4> tracker=bytetrack.yaml conf=0.3 classes=0 imgsz=960`
- Apply downstream filters (min height/aspect ratio) in your own post-processing if desired.

## Dataset Notes
- The MOT directory present (`V:/SSD/Datasets/MOT/mot_format/`) appears to contain detection files (not ground truth) and no images; it is not used for training here. If you provide MOT frames + GT annotations, I can add a converter.
