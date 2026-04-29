import os
import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


def find_best_weights(preferred_runs_root: Optional[str] = None) -> Optional[str]:
    # Find latest best.pt under runs/detect/*/weights
    try:
        repo_root = Path(__file__).resolve().parents[2]
        preferred_root = Path(preferred_runs_root) if preferred_runs_root else repo_root / "runs" / "detect"
        candidates = []
        if preferred_root.is_dir():
            for run_dir in sorted(glob.glob(str(preferred_root / "*")), key=os.path.getmtime, reverse=True):
                best_path = os.path.join(run_dir, "weights", "best.pt")
                if os.path.isfile(best_path):
                    candidates.append(best_path)
        return candidates[0] if candidates else None
    except Exception:
        return None


def load_model(weights: Optional[str] = None) -> YOLO:
    repo_root = Path(__file__).resolve().parents[2]
    if weights and os.path.isfile(weights):
        return YOLO(weights)
    best = find_best_weights()
    if best:
        return YOLO(best)
    packaged_candidates = [
        repo_root / "yolo26n.pt",
        repo_root / "yolov8l.pt",
        Path("yolov8l.pt"),
    ]
    for candidate in packaged_candidates:
        if candidate.is_file():
            return YOLO(str(candidate))
    return YOLO("yolov8n.pt")


def filter_person_boxes(
    result,
    min_conf: float = 0.20,
    min_height: int = 0,
    min_aspect_ratio: float = 0.0,
    max_det: int = 300,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32), None
    xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
    conf = boxes.conf.cpu().numpy()  # (N,)
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf)
    ids = boxes.id.cpu().numpy() if getattr(boxes, 'id', None) is not None else None

    # Person-only: assume single-class model or class 0 for person
    keep = (conf >= min_conf) & ((cls == 0) | (cls == -1))

    if min_height > 0 or min_aspect_ratio > 0.0:
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        ar = np.divide(heights, np.maximum(widths, 1e-6))
        keep = keep & (heights >= float(min_height)) & (ar >= float(min_aspect_ratio))

    xyxy = xyxy[keep]
    conf = conf[keep]
    if ids is not None:
        ids = ids[keep]

    # sort by confidence desc and cap max_det
    if xyxy.shape[0] > 0:
        order = np.argsort(-conf)
        order = order[: int(max_det)]
        xyxy = xyxy[order]
        conf = conf[order]
        if ids is not None:
            ids = ids[order]

    # Return as [x1,y1,x2,y2,conf,(id)]
    if ids is None:
        out = np.concatenate([xyxy, conf[:, None]], axis=1)
        return out, None
    else:
        out = np.concatenate([xyxy, conf[:, None], ids[:, None]], axis=1)
        return out, ids


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    import cv2
    vis = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        # Optional labels: draw ID and confidence if present
        label = None
        if len(b) >= 6:
            try:
                label = f"ID {int(b[5])}"
            except Exception:
                label = None
        if len(b) >= 5:
            try:
                conf_txt = f"{float(b[4]):.2f}"
                label = f"{label} {conf_txt}" if label else conf_txt
            except Exception:
                pass
        if label:
            cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def gaussian_heatmap(h: int, w: int, centers: List[Tuple[int, int]], sigma: int = 32) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    if not centers:
        return heat
    yy, xx = np.mgrid[0:h, 0:w]
    s2 = 2 * sigma * sigma
    for cx, cy in centers:
        heat += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / s2)
    heat = heat / (heat.max() + 1e-6)
    return heat


def analyze_image(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = 0.20,
    min_height: int = 0,
    min_aspect_ratio: float = 0.0,
    max_det: int = 500,
) -> Dict:
    result = model.predict(
        image_bgr[:, :, ::-1],
        imgsz=960,
        verbose=False,
        max_det=max_det,
        classes=[0],
        iou=0.5,
        agnostic_nms=False,
    )[0]
    boxes, _ = filter_person_boxes(result, conf, min_height, min_aspect_ratio, max_det)
    h, w = image_bgr.shape[:2]
    count = boxes.shape[0]
    mpx = (w * h) / 1e6
    density_per_mpx = count / max(mpx, 1e-6)

    centers = [((int(b[0] + b[2]) // 2), (int(b[1] + b[3]) // 2)) for b in boxes]
    heat = gaussian_heatmap(h, w, centers, sigma=max(16, int(min(h, w) * 0.03)))

    vis = draw_boxes(image_bgr, boxes)
    return {
        "count": int(count),
        "density_per_mpx": float(density_per_mpx),
        "boxes": boxes,
        "vis_bgr": vis,
        "heatmap": heat,
    }


def summarize_tracks(track_ids: np.ndarray, boxes: np.ndarray, frame_bgr: np.ndarray) -> Dict[int, np.ndarray]:
    # Create a crop thumbnail per track from first occurrence
    thumbs: Dict[int, np.ndarray] = {}
    if track_ids is None or boxes is None or len(boxes) == 0:
        return thumbs
    for b in boxes:
        if b.shape[0] < 6:
            continue
        tid = int(b[5])
        if tid in thumbs:
            continue
        x1, y1, x2, y2 = map(int, b[:4])
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            thumbs[tid] = crop
    return thumbs


def track_video_collect(
    model: YOLO,
    source: str,
    conf: float = 0.20,
    min_height: int = 0,
    min_aspect_ratio: float = 0.0,
    max_det: int = 500,
    imgsz: int = 960,
    max_frames: Optional[int] = None,
    tracker_cfg: Optional[str] = None,
) -> Dict:
    # Use Ultralytics streaming API for tracking
    counts = []
    densities = []
    unique_ids: set = set()
    sample_thumbs: Dict[int, np.ndarray] = {}
    first_boxes: Dict[int, np.ndarray] = {}
    first_frame_vis = None
    h = w = None

    stream = model.track(
        source=source,
        stream=True,
        tracker=tracker_cfg or "bytetrack.yaml",
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        persist=True,
        classes=[0],
        iou=0.5,
        verbose=False,
    )
    for idx, result in enumerate(stream):
        # result.orig_img is BGR
        frame = result.orig_img.copy()
        if h is None:
            h, w = frame.shape[:2]
        boxes, ids = filter_person_boxes(result, conf, min_height, min_aspect_ratio, max_det)
        if ids is not None:
            unique_ids.update(map(int, ids.tolist()))
        # capture first observed box per track id for later ID remap
        if boxes.shape[0] > 0 and boxes.shape[1] >= 6:
            for b in boxes:
                tid = int(b[5])
                if tid not in first_boxes:
                    first_boxes[tid] = b[:4].copy()

        count = boxes.shape[0]
        counts.append(count)
        mpx = (w * h) / 1e6
        densities.append(count / max(mpx, 1e-6))

        # capture a visualization from the first available frame
        if first_frame_vis is None:
            first_frame_vis = draw_boxes(frame, boxes)
        # gather thumbnails across initial frames until we have a few
        if idx < 100 and len(sample_thumbs) < 16:
            sample_thumbs.update(summarize_tracks(ids, boxes, frame))

        if max_frames is not None and idx + 1 >= max_frames:
            break

    return {
        "counts": counts,
        "densities": densities,
        "unique_ids": sorted(list(unique_ids)),
        "thumbs": sample_thumbs,
        "first_boxes": first_boxes,
        "first_vis_bgr": first_frame_vis,
        "frame_size": (w, h) if w and h else None,
    }


from typing import Optional as _Optional

def iou_xyxy(a, b):
    # a: (4,), b: (N,4) in xyxy
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return inter / union  # (N,)

def render_selected_track_video(
    model: YOLO,
    source: str,
    selected_id: _Optional[int],
    out_path: str,
    conf: float = 0.20,
    min_height: int = 0,
    min_aspect_ratio: float = 0.0,
    max_det: int = 500,
    imgsz: int = 960,
    tracker_cfg: Optional[str] = None,
    strict: bool = True,
    selected_ref_xyxy: _Optional[Tuple[float, float, float, float]] = None,
) -> str:
    import cv2
    # Iterate tracking stream and draw only the selected track id
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    cap.release()

    # Stream tracking and write as we go
    stream = model.track(
        source=source,
        stream=True,
        tracker=tracker_cfg or "bytetrack.yaml",
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        persist=True,
        classes=[0],
        iou=0.5,
        verbose=False,
    )
    remapped_id = None
    remap_done = False
    for result in stream:
        frame = result.orig_img.copy()
        boxes, ids = filter_person_boxes(result, conf, min_height, min_aspect_ratio, max_det)

        # One-time remap: find second-pass ID that matches the first-pass selected box
        if (
            not remap_done
            and selected_id is not None
            and selected_ref_xyxy is not None
            and boxes.shape[0] > 0
        ):
            ious = iou_xyxy(np.array(selected_ref_xyxy, dtype=np.float32), boxes[:, :4].astype(np.float32))
            j = int(np.argmax(ious))
            if ious[j] >= 0.3 and boxes.shape[1] >= 6:  # IoU threshold can be tuned
                remapped_id = int(boxes[j, 5])
                remap_done = True

        target_id = remapped_id if remapped_id is not None else selected_id
        if boxes.shape[0] > 0:
            if target_id is None:
                sel = boxes
            else:
                if boxes.shape[1] >= 6:
                    sel = boxes[boxes[:, 5].astype(int) == int(target_id)]
                    if not strict and sel.shape[0] == 0:
                        sel = boxes  # optional fallback
                else:
                    sel = boxes[:0]
            frame = draw_boxes(frame, sel, color=(0, 255, 255), thickness=3)
        writer.write(frame)

    writer.release()
    return out_path
