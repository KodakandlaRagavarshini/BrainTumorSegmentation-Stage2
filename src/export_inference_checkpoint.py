"""Strip optimizer state from an existing trusted project checkpoint for deployment.

Usage: python -m src.export_inference_checkpoint --input outputs/checkpoints/best_validation_checkpoint.pt
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import torch


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> tuple[Path, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a project training checkpoint with model_state_dict.")
    if output_path.exists():
        raise FileExistsError(f"Output exists; choose another path or move the old file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: checkpoint[key] for key in (
        "model_state_dict", "epoch", "validation_metrics", "artifact_label", "note", "training_config"
    ) if key in checkpoint}
    torch.save(payload, output_path)
    hasher = hashlib.sha256()
    with output_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return output_path, hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/deployment_weights.pt"))
    args = parser.parse_args()
    output_path, digest = export_checkpoint(args.input, args.output)
    print(f"Exported: {output_path.resolve()}")
    print(f"Size: {output_path.stat().st_size / 1024**2:.1f} MiB")
    print(f"SHA-256: {digest}")
    print("Keep this hash for MODEL_SHA256 when hosting weights externally.")


if __name__ == "__main__":
    main()
