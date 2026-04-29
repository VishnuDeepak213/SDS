from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
	# Best metrics extracted from training logs.
	labels = [
		"YOLOv8l (50-epoch run, best)",
		"YOLOv8l (100-epoch run, available best)",
	]

	precision = [0.76565, 0.76171]
	recall = [0.65752, 0.58460]
	map50 = [0.75303, 0.69272]
	map50_95 = [0.46524, 0.40896]

	metrics = [
		("Precision", precision, "#1f77b4"),
		("Recall", recall, "#ff7f0e"),
		("mAP@50", map50, "#2ca02c"),
		("mAP@50-95", map50_95, "#d62728"),
	]

	x = np.arange(len(labels))
	width = 0.18

	fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

	offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
	for i, (name, values, color) in enumerate(metrics):
		bars = ax.bar(x + offsets[i], values, width, label=name, color=color, alpha=0.9)
		for bar, value in zip(bars, values):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				value + 0.008,
				f"{value:.3f}",
				ha="center",
				va="bottom",
				fontsize=9,
			)

	ax.set_title("Model Accuracy Comparison", fontsize=18, pad=12)
	ax.set_ylabel("Score (higher is better)", fontsize=12)
	ax.set_xticks(x)
	ax.set_xticklabels(labels, fontsize=10)
	ax.set_ylim(0, 0.9)
	ax.grid(axis="y", linestyle="--", alpha=0.3)
	ax.legend(loc="upper right", frameon=True)

	note = (
		"Source: runs/detect/*/results.csv | "
		"The 100-epoch run currently contains 4 recorded epochs in the available log."
	)
	fig.text(0.01, 0.01, note, fontsize=9, alpha=0.8)

	out_dir = Path("v:/SSD/person_yolo/outputs")
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / "model_accuracy_comparison.png"

	fig.tight_layout(rect=(0, 0.03, 1, 1))
	fig.savefig(out_path, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved: {out_path}")


if __name__ == "__main__":
	main()
