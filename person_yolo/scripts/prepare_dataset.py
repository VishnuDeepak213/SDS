import shutil
from pathlib import Path
from typing import List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def gather_yolo_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not images_dir or not images_dir.exists():
        return pairs
    for img in images_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            lbl = labels_dir / (img.stem + ".txt") if labels_dir else None
            pairs.append((img, lbl if (lbl and lbl.exists()) else None))
    return pairs


def copy_item(src_img: Path, src_lbl: Path | None, dst_img_root: Path, dst_lbl_root: Path) -> None:
    dst_img_root.mkdir(parents=True, exist_ok=True)
    dst_lbl_root.mkdir(parents=True, exist_ok=True)
    # Flatten filename to avoid nested dirs
    dst_img = dst_img_root / src_img.name
    shutil.copy2(src_img, dst_img)
    dst_lbl = dst_lbl_root / (src_img.stem + ".txt")
    if src_lbl and src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        # create empty for negatives
        dst_lbl.write_text("")


def merge_sources(sources: List[Tuple[Path, Path]], negatives: Path | None, out_root: Path, val_ratio: float = 0.1) -> None:
    """
    sources: list of (images_dir, labels_dir) in YOLO layout
    negatives: dir of images without labels (optional)
    Creates `train/images, train/labels, val/images, val/labels` under out_root.
    """
    out_train_img = out_root / "train" / "images"
    out_train_lbl = out_root / "train" / "labels"
    out_val_img = out_root / "val" / "images"
    out_val_lbl = out_root / "val" / "labels"
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)
    out_val_img.mkdir(parents=True, exist_ok=True)
    out_val_lbl.mkdir(parents=True, exist_ok=True)

    # Collect all pairs
    all_pairs: List[Tuple[Path, Path | None]] = []
    for img_dir, lbl_dir in sources:
        all_pairs.extend(gather_yolo_pairs(img_dir, lbl_dir))

    if negatives and negatives.exists():
        for img in negatives.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                all_pairs.append((img, None))

    # Split
    n_total = len(all_pairs)
    n_val = int(n_total * val_ratio)
    # Simple deterministic split: first n_val to val, rest to train
    val_items = all_pairs[:n_val]
    train_items = all_pairs[n_val:]

    for img, lbl in val_items:
        copy_item(img, lbl, out_val_img, out_val_lbl)
    for img, lbl in train_items:
        copy_item(img, lbl, out_train_img, out_train_lbl)

    print(f"Merged total: {n_total} | train: {len(train_items)} | val: {len(val_items)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge YOLO sources (Wider/MOT) plus negatives into a unified dataset")
    parser.add_argument("--wider_yolo", type=str, required=True, help="Path to WiderPerson YOLO root (contains train/ or images/labels)")
    parser.add_argument("--mot_yolo", type=str, default="", help="Path to MOT YOLO root (optional)")
    parser.add_argument("--negatives", type=str, default="", help="Path to negatives images directory (optional)")
    parser.add_argument("--out", type=str, required=True, help="Output root for merged dataset")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def detect_images_labels(root: Path) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        # Support either train/val layout or flat images/labels
        if (root / "train" / "images").exists():
            pairs.append((root / "train" / "images", root / "train" / "labels"))
        if (root / "val" / "images").exists():
            pairs.append((root / "val" / "images", root / "val" / "labels"))
        if not pairs:
            pairs.append((root / "images", root / "labels"))
        return pairs

    sources: List[Tuple[Path, Path]] = []
    wider_root = Path(args.wider_yolo)
    if wider_root.exists():
        sources.extend(detect_images_labels(wider_root))
    else:
        raise FileNotFoundError(f"Wider YOLO root not found: {wider_root}")

    mot_root = Path(args.mot_yolo) if args.mot_yolo else None
    if mot_root and mot_root.exists():
        sources.extend(detect_images_labels(mot_root))
    else:
        if mot_root:
            print(f"[warn] MOT YOLO root not found: {mot_root}. Proceeding without MOT.")

    negatives = Path(args.negatives) if args.negatives else None
    merge_sources(sources, negatives, out_root, args.val_ratio)
    print(f"Merged dataset at: {out_root}")


if __name__ == "__main__":
    main()
import argparse
import os
import shutil
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def gather_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for img in images_dir.rglob("*"):
        if not is_image(img):
            continue
        rel = img.relative_to(images_dir)
        lbl = labels_dir / rel.with_suffix(".txt")
        pairs.append((img, lbl))
    return pairs


def copy_with_structure(src_img: Path, src_lbl: Path, dst_img_root: Path, dst_lbl_root: Path):
    rel = src_img.name  # flatten by filename to avoid deep nesting collisions
    dst_img = dst_img_root / rel
    dst_lbl = dst_lbl_root / Path(rel).with_suffix(".txt")
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    if src_lbl and src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        # create empty label for negatives or missing labels
        dst_lbl.write_text("")


def merge_sources(wider_path: Path, mot_path: Path, negatives_path: Path, out_root: Path, val_ratio: float = 0.1):
    train_images = out_root / "train" / "images"
    train_labels = out_root / "train" / "labels"
    val_images = out_root / "val" / "images"
    val_labels = out_root / "val" / "labels"
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Expect YOLO-formatted sources: each has images/ and labels/
    sources = []
    if wider_path:
        sources.append((wider_path / "images", wider_path / "labels"))
    if mot_path:
        sources.append((mot_path / "images", mot_path / "labels"))

    all_pairs = []
    for img_dir, lbl_dir in sources:
        if not img_dir.exists():
            print(f"Warning: images dir missing: {img_dir}")
            continue
        pairs = gather_pairs(img_dir, lbl_dir)
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * val_ratio)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    for img, lbl in train_pairs:
        copy_with_structure(img, lbl, train_images, train_labels)
    for img, lbl in val_pairs:
        copy_with_structure(img, lbl, val_images, val_labels)

    # Negatives: copy images and create empty labels
    if negatives_path and negatives_path.exists():
        neg_imgs = [p for p in negatives_path.rglob("*") if is_image(p)]
        # put 90% in train and 10% in val
        random.shuffle(neg_imgs)
        n_val_neg = max(1, int(len(neg_imgs) * val_ratio)) if neg_imgs else 0
        val_neg = neg_imgs[:n_val_neg]
        train_neg = neg_imgs[n_val_neg:]
        for img in train_neg:
            copy_with_structure(img, None, train_images, train_labels)
        for img in val_neg:
            copy_with_structure(img, None, val_images, val_labels)
        print(f"Added {len(train_neg)} train negatives and {len(val_neg)} val negatives.")

    print(f"Done. Train images: {len(list(train_images.glob('*')))}, Val images: {len(list(val_images.glob('*')))}")


def main():
    parser = argparse.ArgumentParser(description="Merge WiderPerson+MOT YOLO datasets with negatives.")
    parser.add_argument("--wider_yolo", type=str, required=False, help="Path to WiderPerson in YOLO format (must contain images/ and labels/)")
    parser.add_argument("--mot_yolo", type=str, required=False, help="Path to MOT in YOLO format (must contain images/ and labels/)")
    parser.add_argument("--negatives", type=str, required=False, help="Path to folder containing negative images (no labels)")
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "datasets" / "person_wider_mot"), help="Output root")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    wider = Path(args.wider_yolo) if args.wider_yolo else None
    mot = Path(args.mot_yolo) if args.mot_yolo else None
    negatives = Path(args.negatives) if args.negatives else None
    out_root = Path(args.out)

    merge_sources(wider, mot, negatives, out_root, args.val_ratio)


if __name__ == "__main__":
    main()
