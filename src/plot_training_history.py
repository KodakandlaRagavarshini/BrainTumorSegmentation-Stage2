"""Create a small multi-panel plot from saved training history CSV rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "train_batches",
    "val_batches",
    "validation_tumor_positive_slices",
    "validation_tumor_negative_slices",
    "elapsed_seconds",
]


def _read_history_rows(csv_path: str | Path) -> list[dict[str, float | int]]:
    history_path = Path(csv_path)
    if not history_path.exists():
        raise FileNotFoundError(f"Training history CSV does not exist: {history_path}")
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Training history CSV is missing a header row: {history_path}")
        missing = [field for field in REQUIRED_HISTORY_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Training history CSV is missing required fields: {missing}. "
                f"Expected: {REQUIRED_HISTORY_FIELDS}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Training history CSV contains no rows: {history_path}")
    return rows


def generate_training_plots(csv_path: str | Path, output_path: str | Path, *, overwrite: bool = False) -> Path:
    """Generate a combined PNG with loss, Dice/IoU, and precision/recall curves."""
    source_path = Path(csv_path)
    destination_path = Path(output_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Training history source CSV missing: {source_path}")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Plot output already exists: {destination_path}. "
            "Use overwrite=True or delete the file to replace it explicitly."
        )

    rows = _read_history_rows(source_path)
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    dice = [float(row["dice"]) for row in rows]
    iou = [float(row["iou"]) for row in rows]
    precision = [float(row["precision"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=150, constrained_layout=True)

    axes[0].plot(epochs, train_loss, label="Train loss", color="tab:blue", marker="o")
    axes[0].plot(epochs, val_loss, label="Validation loss", color="tab:orange", marker="s")
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(epochs, dice, label="Dice", color="tab:green", marker="o")
    axes[1].plot(epochs, iou, label="IoU", color="tab:red", marker="s")
    axes[1].set_title("Validation Dice and IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(epochs, precision, label="Precision", color="tab:purple", marker="o")
    axes[2].plot(epochs, recall, label="Recall", color="tab:brown", marker="s")
    axes[2].set_title("Validation precision and recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    fig.suptitle("BraTS segmentation smoke-test learning curves", fontsize=12)
    fig.savefig(destination_path, dpi=150)
    plt.close(fig)
    return destination_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot CSV training history for the BraTS smoke-test experiment.")
    parser.add_argument("--history-csv", type=Path, required=True, help="Path to the training history CSV file.")
    parser.add_argument("--plot-path", type=Path, required=True, help="Output PNG path for the training plots.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing PNG file.")
    args = parser.parse_args(argv)
    generate_training_plots(args.history_csv, args.plot_path, overwrite=args.overwrite)
    print(f"Saved training plots to: {args.plot_path}")


if __name__ == "__main__":
    main()
