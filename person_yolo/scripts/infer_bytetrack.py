import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack tracking with post-filtering")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt")
    parser.add_argument("--source", type=str, required=True, help="Video file or camera index")
    parser.add_argument("--conf", type=float, default=0.28, help="Confidence threshold")
    parser.add_argument("--min_h", type=int, default=30, help="Min box height (px)")
    parser.add_argument("--min_ar", type=float, default=1.5, help="Min height/width aspect ratio")
    parser.add_argument("--save", type=str, default="outputs/tracked_bytetrack.mp4", help="Output video path")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Setup writer lazily after first frame
    writer = None
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    # Track with Ultralytics (ByteTrack)
    gen = model.track(
        source=0 if args.source == "0" else args.source,
        conf=args.conf,
        classes=[0],
        imgsz=960,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
        save=False,
    )

    for res in gen:
        frame = res.orig_img
        h, w = frame.shape[:2]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 30, (w, h))
            writer.write(frame)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.array([-1]*len(xyxy))

        keep = []
        for (x1, y1, x2, y2), s, tid in zip(xyxy, confs, ids):
            if s < args.conf:
                continue
            w_box = max(1.0, x2 - x1)
            h_box = max(1.0, y2 - y1)
            if h_box < args.min_h:
                continue
            if (h_box / w_box) < args.min_ar:
                continue
            keep.append((int(x1), int(y1), int(x2), int(y2), float(s), int(tid)))

        # Init writer if needed
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        # Draw kept tracks
        for x1, y1, x2, y2, s, tid in keep:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {tid if tid >= 0 else 0} {s:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"Saved tracked output to: {args.save}")


if __name__ == "__main__":
    main()
