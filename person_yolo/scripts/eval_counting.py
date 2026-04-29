import argparse
import os
from typing import List, Tuple, Dict

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def yolo_txt_to_xyxy(txt_path: str, img_w: int, img_h: int) -> np.ndarray:
    boxes = []
    if not os.path.isfile(txt_path):
        return np.zeros((0, 4), dtype=np.float32)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO: class cx cy w h (normalized)
            cls = int(float(parts[0]))
            if cls != 0:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    # boxes shape: (N,4) with xyxy
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)
    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def match_boxes(pred: np.ndarray, gt: np.ndarray, iou_thr: float) -> Tuple[int, List[int], List[int]]:
    # Returns: num_matches, unmatched_pred_idx, unmatched_gt_idx
    if pred.size == 0 or gt.size == 0:
        return 0, list(range(pred.shape[0])), list(range(gt.shape[0]))
    ious = iou_matrix(pred, gt)
    # Hungarian on cost = 1 - IoU
    cost = 1.0 - ious
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = 0
    used_pred = set()
    used_gt = set()
    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= iou_thr:
            matches += 1
            used_pred.add(r)
            used_gt.add(c)
    unmatched_pred = [i for i in range(pred.shape[0]) if i not in used_pred]
    unmatched_gt = [j for j in range(gt.shape[0]) if j not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def valid_pred_boxes(result, conf_thr: float, min_h: int = 0, min_ar: float = 0.0) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf)
    keep = (conf >= conf_thr) & (cls == 0)
    xyxy = xyxy[keep]
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if min_h > 0 or min_ar > 0:
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        ar = np.divide(h, np.maximum(w, 1e-6))
        keep2 = np.ones(xyxy.shape[0], dtype=bool)
        if min_h > 0:
            keep2 &= h >= min_h
        if min_ar > 0:
            keep2 &= ar >= min_ar
        xyxy = xyxy[keep2]
    return xyxy.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Evaluate person detection & counting accuracy")
    ap.add_argument("--weights", required=True, help="Path to YOLO weights (e.g., best.pt)")
    ap.add_argument("--data", required=True, help="data.yaml path")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_match", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument("--min_h", type=int, default=0)
    ap.add_argument("--min_ar", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=0, help="Max images to evaluate (0=all)")
    args = ap.parse_args()

    cfg = load_yaml(args.data)
    base = cfg.get("path", "")
    img_rel = cfg.get(args.split, "")
    if not base or not img_rel:
        raise SystemExit("Invalid data.yaml: must contain 'path' and split entry")
    img_dir = img_rel if os.path.isabs(img_rel) else os.path.join(base, img_rel)
    lbl_dir = img_dir.replace("/images", "/labels").replace("\\images", "\\labels")

    model = YOLO(args.weights)

    # Gather image-label pairs
    images = []
    for root, _, files in os.walk(img_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                ip = os.path.join(root, fn)
                ln = os.path.splitext(fn)[0] + ".txt"
                lp = os.path.join(lbl_dir, ln)
                images.append((ip, lp))
    images.sort()
    if args.max and args.max > 0:
        images = images[: args.max]

    # Metrics accumulators
    abs_errors = []
    abs_pct_errors = []
    total_abs_error = 0
    total_gt = 0
    exact_match = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for ip, lp in images:
        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = yolo_txt_to_xyxy(lp, w, h)
        pred_res = model.predict(img[:, :, ::-1], imgsz=960, conf=args.conf, verbose=False)[0]
        pred = valid_pred_boxes(pred_res, args.conf, args.min_h, args.min_ar)

        gt_count = gt.shape[0]
        pred_count = pred.shape[0]
        # Matching for detection stats
        matches, unp, ung = match_boxes(pred, gt, args.iou_match)
        tp = matches
        fp = len(unp)
        fn = len(ung)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Counting metrics
        err = pred_count - gt_count
        abs_err = abs(err)
        abs_errors.append(abs_err)
        total_abs_error += abs_err
        total_gt += gt_count
        if gt_count > 0:
            abs_pct_errors.append(abs_err / gt_count)
        exact_match += int(pred_count == gt_count)

    n = max(len(images), 1)
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    mape = float(np.mean(abs_pct_errors)) * 100.0 if abs_pct_errors else 0.0
    wape = (total_abs_error / max(total_gt, 1)) * 100.0
    exact_acc = (exact_match / n) * 100.0
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-6)) if (precision + recall) > 0 else 0.0

    print("=== Detection & Counting Metrics ===")
    print(f"Images evaluated: {n}")
    print(f"Detection precision: {precision:.3f}")
    print(f"Detection recall:    {recall:.3f}")
    print(f"F1 score:            {f1:.3f}")
    print("--- Counting (persons/image) ---")
    print(f"MAE:   {mae:.3f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"WAPE:  {wape:.2f}%")
    print(f"Exact count accuracy (images): {exact_acc:.2f}%")


if __name__ == "__main__":
    main()
