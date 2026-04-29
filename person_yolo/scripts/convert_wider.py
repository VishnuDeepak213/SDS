import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2


def read_wider_annotation(anno_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse WiderPerson annotation file.
    Format observed:
    - First line: integer count N
    - Next N lines: "cls x1 y1 x2 y2" (integers), where cls appears to be 1 (person) or 3 (head)
    Returns list of tuples (cls, x1, y1, x2, y2)
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not anno_path.exists():
        return boxes
    try:
        with anno_path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return boxes
        # First line is count; some files may omit it, be defensive
        start_idx = 1
        try:
            _ = int(lines[0])
        except ValueError:
            start_idx = 0
        for ln in lines[start_idx:]:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
            except ValueError:
                continue
            # ensure x1<x2, y1<y2
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((cls, x1, y1, x2, y2))
    except Exception:
        # fall back to empty
        return boxes
    return boxes


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    # Clamp to image bounds
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    # class 0 is 'person'
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_split(ids_file: Path, images_dir: Path, annos_dir: Path, out_images: Path, out_labels: Path) -> int:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    total = 0
    with ids_file.open("r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f.readlines() if ln.strip()]
    for img_id in ids:
        img_name = f"{img_id}.jpg"
        src_img = images_dir / img_name
        src_anno = annos_dir / f"{img_name}.txt"
        if not src_img.exists():
            # try zero-padded id variants if needed
            continue
        # read image to get size
        img = cv2.imread(str(src_img))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_wider_annotation(src_anno)
        # Map cls 1 (person) and 3 (head) to person class 0; skip others
        yolo_lines: List[str] = []
        for cls, x1, y1, x2, y2 in boxes:
            if cls not in (1, 3):
                continue
            line = to_yolo_line(x1, y1, x2, y2, w, h)
            # filter absurd boxes
            parts = line.split()
            bw = float(parts[3])
            bh = float(parts[4])
            if bw <= 0 or bh <= 0:
                continue
            yolo_lines.append(line)
        # Copy image
        dst_img = out_images / img_name
        shutil.copy2(src_img, dst_img)
        # Write label (empty file allowed for negatives)
        dst_lbl = out_labels / (Path(img_name).stem + ".txt")
        with dst_lbl.open("w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines))
        total += 1
    return total


def convert_wider(root: Path, out_root: Path) -> None:
    """
    Convert WiderPerson dataset found at `root` into YOLO format under `out_root`.

    Expected structure at `root`:
    - Images/ (contains <id>.jpg)
    - Annotations/ (contains <id>.jpg.txt)
    - train.txt, val.txt (lists of IDs without extension)
    """
    images_dir = root / "Images"
    annos_dir = root / "Annotations"
    train_ids = root / "train.txt"
    val_ids = root / "val.txt"
    if not images_dir.exists() or not annos_dir.exists() or not train_ids.exists() or not val_ids.exists():
        raise FileNotFoundError("WiderPerson dataset structure invalid. Expect Images/, Annotations/, train.txt, val.txt")

    train_img_out = out_root / "train" / "images"
    train_lbl_out = out_root / "train" / "labels"
    val_img_out = out_root / "val" / "images"
    val_lbl_out = out_root / "val" / "labels"

    print(f"Converting train split from {train_ids} ...")
    n_train = convert_split(train_ids, images_dir, annos_dir, train_img_out, train_lbl_out)
    print(f"Train images converted: {n_train}")

    print(f"Converting val split from {val_ids} ...")
    n_val = convert_split(val_ids, images_dir, annos_dir, val_img_out, val_lbl_out)
    print(f"Val images converted: {n_val}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert WiderPerson to YOLO format (person-only)")
    parser.add_argument("--wider_root", type=str, required=True, help="Path to WiderPerson dataset root (contains Images/, Annotations/, train.txt, val.txt)")
    parser.add_argument("--out_root", type=str, default=str(Path("v:/SSD/person_yolo/datasets/wider_yolo")), help="Output root for YOLO dataset")
    args = parser.parse_args()

    wider_root = Path(args.wider_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    convert_wider(wider_root, out_root)
    print(f"Done. YOLO dataset at: {out_root}")


if __name__ == "__main__":
    main()
