import os
import time
import subprocess
from pathlib import Path
from datetime import datetime


RUN_NAME = "yolo_person_wider_mot_50"
BEST_PATH = Path(f"v:/SSD/runs/detect/{RUN_NAME}/weights/best.pt")
DATA_YAML = Path("v:/SSD/person_yolo/configs/data.yaml")
TECH_REF = Path("v:/SSD/person_yolo/docs/TECHNICAL_REFERENCE.md")


def wait_for_best(path: Path, stable_seconds: int = 60, poll: float = 10.0) -> None:
    """Poll until best.pt exists and its size is stable for stable_seconds."""
    print(f"[watch] Waiting for best.pt at: {path}")
    last_size = None
    stable_start = None
    while True:
        if path.exists():
            size = path.stat().st_size
            if last_size is None or size != last_size:
                last_size = size
                stable_start = time.time()
            else:
                if stable_start and (time.time() - stable_start) >= stable_seconds:
                    print("[watch] best.pt detected and stable.")
                    return
        time.sleep(poll)


def run_eval(weights: Path, data_yaml: Path) -> str:
    """Run eval_counting.py and return stdout text."""
    cmd = [
        str(Path("v:/SSD/.venv/Scripts/python.exe")),
        str(Path("v:/SSD/person_yolo/scripts/eval_counting.py")),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--split", "val",
        "--conf", "0.25",
        "--iou_match", "0.5",
        "--min_h", "0",
        "--min_ar", "0.0",
        "--max", "0",
    ]
    print("[watch] Running evaluator:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[watch] Evaluator failed:")
        print(proc.stderr)
    print("[watch] Evaluator output:")
    print(proc.stdout)
    return proc.stdout


def append_metrics(doc_path: Path, run_name: str, best_path: Path, eval_out: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        "\n\n## Evaluation Metrics (" + ts + ")\n"
        + f"- Run: {run_name}\n"
        + f"- Weights: {best_path}\n\n"
        + "```\n" + eval_out.strip() + "\n```\n"
    )
    with doc_path.open("a", encoding="utf-8") as f:
        f.write(block)
    print(f"[watch] Appended metrics to: {doc_path}")


def main():
    wait_for_best(BEST_PATH)
    out = run_eval(BEST_PATH, DATA_YAML)
    append_metrics(TECH_REF, RUN_NAME, BEST_PATH, out)
    print("[watch] Done.")


if __name__ == "__main__":
    main()
