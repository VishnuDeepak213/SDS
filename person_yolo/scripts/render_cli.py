import os
import argparse
from typing import Optional, Tuple

from ultralytics import YOLO  # noqa: F401 (kept for type hints)

try:
    # Prefer local package import when running as module
    from person_yolo.web.utils import (
        load_model,
        track_video_collect,
        render_selected_track_video,
    )
except Exception:
    # Fallback to relative import when executed directly
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from person_yolo.web.utils import (
        load_model,
        track_video_collect,
        render_selected_track_video,
    )


def parse_args():
    p = argparse.ArgumentParser("Render selected person or all detections using YOLOv8 + ByteTrack")
    p.add_argument("--source", required=True, help="Path to input video file")
    p.add_argument("--weights", default="", help="Path to YOLOv8 weights (optional)")
    p.add_argument("--conf", type=float, default=0.12, help="Confidence threshold for detection")
    p.add_argument("--imgsz", type=int, default=1152, help="Inference size (multiple of 32 recommended)")
    p.add_argument("--max-det", type=int, default=700, help="Maximum detections per frame")
    p.add_argument("--min-ar", type=float, default=0.0, help="Minimum aspect ratio (h/w) filter")
    p.add_argument("--max-frames", type=int, default=200, help="Number of frames to process in first pass for ID discovery")
    p.add_argument("--strict", action="store_true", help="Strict: only selected person; no fallback frames")
    p.add_argument("--tracker-config", default="", help="Path to ByteTrack config YAML (optional)")
    p.add_argument("--select-id", type=int, default=None, help="Track ID to render (from first pass)")
    p.add_argument("--auto", choices=["largest", "first"], default="largest", help="Auto-select strategy when --select-id is not provided")
    p.add_argument("--all", action="store_true", help="Render all detections instead of a single person")
    p.add_argument("--list-only", action="store_true", help="Only list discovered track IDs; do not render")
    return p.parse_args()


def pick_auto_id(first_boxes: dict, strategy: str) -> Optional[int]:
    if not first_boxes:
        return None
    if strategy == "first":
        return next(iter(first_boxes.keys()))
    # largest bounding box by area
    best_id = None
    best_area = -1.0
    for tid, b in first_boxes.items():
        x1, y1, x2, y2 = map(float, b[:4])
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if area > best_area:
            best_area = area
            best_id = int(tid)
    return best_id


def main():
    args = parse_args()
    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Video not found: {source}")

    # Load model
    weights = args.weights if args.weights and os.path.isfile(args.weights) else None
    model = load_model(weights)

    # Resolve tracker config path
    tracker_cfg = None
    if args.tracker_config and os.path.isfile(args.tracker_config):
        tracker_cfg = args.tracker_config
    else:
        # Default to the tuned ByteTrack config shipped with the web app
        default_cfg = os.path.join(os.path.dirname(__file__), os.pardir, "web", "bytetrack_person.yaml")
        default_cfg = os.path.abspath(default_cfg)
        if os.path.isfile(default_cfg):
            tracker_cfg = default_cfg

    # First pass: collect tracks and reference boxes
    print("[1/2] Tracking (first pass) to collect IDs and thumbnails...")
    summary = track_video_collect(
        model,
        source,
        conf=float(args.conf),
        min_height=0,
        min_aspect_ratio=float(args.min_ar),
        max_det=int(args.max_det),
        imgsz=int(args.imgsz),
        max_frames=int(args.max_frames),
        tracker_cfg=tracker_cfg,
    )
    ids = summary.get("unique_ids", [])
    first_boxes = summary.get("first_boxes", {})
    print(f"Found track IDs: {ids}")

    if args.list_only:
        print("List-only mode: no rendering performed.")
        return

    selected_id: Optional[int]
    selected_id = None if args.all else args.select_id
    if selected_id is None and not args.all:
        selected_id = pick_auto_id(first_boxes, args.auto)
        print(f"Auto-selected ID: {selected_id} (strategy={args.auto})")

    # Prepare output path
    out_dir = os.path.join(os.path.dirname(source), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"all_detections.mp4" if selected_id is None else f"selected_id_{selected_id}.mp4"
    out_path = os.path.join(out_dir, out_name)

    # Second pass: render
    print("[2/2] Rendering video...")
    ref_xyxy: Optional[Tuple[float, float, float, float]] = None
    if selected_id is not None and isinstance(first_boxes, dict):
        ref_xyxy = first_boxes.get(int(selected_id))

    final = render_selected_track_video(
        model,
        source,
        selected_id,
        out_path,
        conf=float(args.conf),
        min_height=0,
        min_aspect_ratio=float(args.min_ar),
        max_det=int(args.max_det),
        imgsz=int(args.imgsz),
        tracker_cfg=tracker_cfg,
        strict=bool(args.strict),
        selected_ref_xyxy=ref_xyxy,
    )
    print(f"Saved: {final}")


if __name__ == "__main__":
    main()
