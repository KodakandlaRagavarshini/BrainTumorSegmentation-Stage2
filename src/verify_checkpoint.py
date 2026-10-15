"""Load and verify a trusted local checkpoint for the VGG19 U-Net.

This script only accepts a checkpoint that was saved by this project on the local
machine. It intentionally uses pretrained=False to avoid downloading weights, and
it runs a single inference-only forward pass to confirm the checkpoint is valid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from src.model_vgg19_unet import VGG19UNet
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback.
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.model_vgg19_unet import VGG19UNet


def _load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}.")
    required = {
        "model_state_dict",
        "epoch",
        "validation_loss",
        "validation_metrics",
        "training_config",
        "random_seed",
        "data_index_filename",
    }
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint metadata is missing expected keys: {sorted(missing)}")
    return checkpoint


def inspect_checkpoint(checkpoint: dict) -> None:
    print(f"Checkpoint artifact: {checkpoint.get('artifact_label', 'unknown')}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Validation loss: {checkpoint.get('validation_loss')}")
    print(f"Validation metrics: {checkpoint.get('validation_metrics')}")
    print(f"Tumor-positive Dice: {checkpoint.get('tumor_positive_dice')}")
    print(f"Checkpoint selection metric: {checkpoint.get('checkpoint_selection_metric', 'dice')}")
    print(f"Random seed: {checkpoint.get('random_seed')}")
    print(f"Data index file: {checkpoint.get('data_index_filename')}")
    print(f"Training config: {checkpoint.get('training_config')}")
    print(f"Optimizer state present: {'optimizer_state_dict' in checkpoint}")
    print(f"Note: {checkpoint.get('note', '')}")


def verify_checkpoint(checkpoint_path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    inspect_checkpoint(checkpoint)

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model_state_dict.")

    model = VGG19UNet(pretrained=False, freeze_encoder=True).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fixed_input = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    with torch.inference_mode():
        logits = model(fixed_input)

    output_shape = tuple(logits.shape)
    finite_logits = bool(torch.isfinite(logits).all().item())
    print(f"Output shape: {output_shape}")
    print(f"All logits finite: {finite_logits}")

    if output_shape != (1, 1, 256, 256):
        raise ValueError(f"Expected output shape (1, 1, 256, 256), got {output_shape}.")
    if not finite_logits:
        raise ValueError("Model output logits are not finite.")

    print("PASS: checkpoint restored successfully and inference-only forward pass succeeded.")
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "validation_loss": checkpoint.get("validation_loss"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "output_shape": output_shape,
        "finite_logits": finite_logits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a trusted local checkpoint for this project.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a local checkpoint file created by this project.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_checkpoint(args.checkpoint)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, RuntimeError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}")
