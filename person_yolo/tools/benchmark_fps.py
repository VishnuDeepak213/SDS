import argparse
import time
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=str, default="runs/benchmark/fps.txt")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    H = W = args.imgsz
    # create one random BGR image
    img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)

    # warmup
    for _ in range(args.warmup):
        _ = model.predict(img[:, :, ::-1], imgsz=args.imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = model.predict(img[:, :, ::-1], imgsz=args.imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = args.iters / dt if dt > 0 else 0.0

    out = f"FPS: {fps:.2f} | imgsz: {args.imgsz} | iters: {args.iters}\n"
    print(out)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == '__main__':
    main()
