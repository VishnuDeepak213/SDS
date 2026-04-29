import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Simple IOU-based tracker for consistent IDs
class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_missing=30):
        self.tracks = {}  # id -> {bbox, missing}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detections):
        # detections: list of (x1,y1,x2,y2,score)
        assigned = {}
        det_used = set()
        # Greedy assignment by IOU
        for tid, t in list(self.tracks.items()):
            best_iou, best_j = 0.0, None
            for j, d in enumerate(detections):
                if j in det_used:
                    continue
                i = self.iou(t["bbox"], d[:4])
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_j is not None and best_iou >= self.iou_thresh:
                self.tracks[tid] = {"bbox": detections[best_j][:4], "missing": 0}
                det_used.add(best_j)
                assigned[tid] = detections[best_j]
            else:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
        # Create new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j in det_used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": d[:4], "missing": 0}
            assigned[tid] = d
        return assigned


def filter_detections(boxes, confs, img_h, img_w, conf_thresh=0.28, min_h=30, min_ar=1.5):
    out = []
    for b, s in zip(boxes, confs):
        if s < conf_thresh:
            continue
        x1, y1, x2, y2 = map(float, b)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        # height threshold
        if h < min_h:
            continue
        # aspect ratio height/width
        ar = h / w
        if ar < min_ar:
            continue
        # clip to image bounds
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
        out.append((x1, y1, x2, y2, s))
    return out


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 person detect+track with filtering")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt")
    parser.add_argument("--source", type=str, required=True, help="Video file or camera index")
    parser.add_argument("--conf", type=float, default=0.28, help="Confidence threshold")
    parser.add_argument("--min_h", type=int, default=30, help="Min box height (px)")
    parser.add_argument("--min_ar", type=float, default=1.5, help="Min height/width aspect ratio")
    parser.add_argument("--save", type=str, default="outputs/tracked.mp4", help="Output video path")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Open source
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    tracker = SimpleTracker(iou_thresh=0.35, max_missing=20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Predict persons only
        results = model.predict(frame, classes=[0], conf=args.conf, imgsz=IMGSZ := 960, verbose=False)
        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,))

        dets = filter_detections(boxes_xyxy, confs, frame.shape[0], frame.shape[1], args.conf, args.min_h, args.min_ar)
        assigned = tracker.update(dets)

        # Draw
        for tid, d in assigned.items():
            x1, y1, x2, y2, s = d
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid} {s:.2f}", (p1[0], max(0, p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)
        # Optional preview: comment out for non-interactive
        # cv2.imshow("tracked", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    writer.release()
    print(f"Saved tracked output to: {args.save}")


if __name__ == "__main__":
    main()
