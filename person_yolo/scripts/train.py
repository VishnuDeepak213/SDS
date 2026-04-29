from pathlib import Path
from ultralytics import YOLO


DATA = "v:/SSD/person_yolo/configs/data.yaml"
MODEL = "yolov8l.pt"

# Augmentation and training args aligned to user request
AUG_ARGS = dict(
    mosaic=1.0,
    copy_paste=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)


def main():
    out_name = "yolo_person_wider_mot"
    save_dir = Path("runs/detect") / out_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=300,
        imgsz=960,
        optimizer="AdamW",
        lr0=0.002,
        box=8.0,
        cls=0.3,
        device=0,
        project=str(save_dir.parent),
        name=out_name,
        **AUG_ARGS,
    )
    print("Training complete. Best weights under:", save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
import os
from pathlib import Path
import torch
from ultralytics import YOLO

# Training config matching requirements
EPOCHS = 300
IMGSZ = 960
OPTIMIZER = "AdamW"
LR0 = 0.002
BOX_LOSS = 8.0
CLS_LOSS = 0.3
PROJECT = "runs/detect"
NAME = "yolo_person_wider_mot"
DATA_YAML = str(Path(__file__).resolve().parents[1] / "configs" / "data.yaml")
MODEL = "yolov8l.pt"

# Augmentations
AUG_ARGS = {
    "mosaic": 1.0,       # enable mosaic strongly
    "copy_paste": 0.5,   # synthetic dense crowds
    "fliplr": 0.5,       # horizontal flips
    # Brightness/contrast via HSV V/S
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
}


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(MODEL)
    # Train with specified hyperparameters
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        optimizer=OPTIMIZER,
        lr0=LR0,
        box=BOX_LOSS,
        cls=CLS_LOSS,
        project=PROJECT,
        name=NAME,
        device=device,
        **AUG_ARGS,
    )

    # Confirm best weights path
    best_path = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_path}")


if __name__ == "__main__":
    main()
