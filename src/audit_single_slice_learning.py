"""Deterministic single-positive-slice overfitting diagnostic for the MRI pipeline.

This module intentionally stays read-only and does not modify the existing
training pipeline. It mirrors the project’s real dataset, normalization,
model, optimizer, and loss definitions, but runs independently from the main
training script and only executes when invoked directly.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

from src.dataset_nifti_2d import BraTSNiftiSliceDataset
from src.input_normalization import ImageNetNormalizedDataset
from src.losses import FocalTverskyLoss
from src.metrics import whole_tumor_metrics
from src.model_vgg19_unet import VGG19UNet

DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[1] / "data/processed/slice_index_subset_12train_4val.csv"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data/raw/brats2020"


def set_seed(seed: int = 42) -> None:
    """Set a reproducible seed for the diagnostic, where supported by the runtime."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed_all"):
        torch.mps.manual_seed_all(seed)


def _iter_train_rows(index_path: str | Path) -> list[dict[str, str]]:
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("split") == "train"]
    return rows


def select_single_positive_training_slice(index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Pick the largest tumor-positive training slice with deterministic tie-breaking."""
    rows = _iter_train_rows(index_path)
    positive_rows = [row for row in rows if int(row["tumor_pixel_count"]) > 0]
    if not positive_rows:
        raise ValueError(f"No tumor-positive slice was found in the train split of {index_path}.")

    chosen = sorted(
        positive_rows,
        key=lambda row: (-int(row["tumor_pixel_count"]), row["patient_id"], int(row["slice_index"])),
    )[0]
    chosen["tumor_pixel_count"] = int(chosen["tumor_pixel_count"])
    chosen["slice_index"] = int(chosen["slice_index"])
    return chosen


def resolve_selected_dataset_index(dataset, selected_record: dict[str, Any]) -> int:
    patient_id = selected_record["patient_id"]
    slice_index = int(selected_record["slice_index"])
    for dataset_index, record in enumerate(dataset.records):
        if record["patient_id"] == patient_id and int(record["slice_index"]) == slice_index:
            return dataset_index
    raise ValueError(
        f"Selected patient/slice pair ({patient_id}, {slice_index}) was not found in the dataset records."
    )


def load_selected_single_slice(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and normalize the deterministically selected tumor-positive MRI slice."""
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    dataset = BraTSNiftiSliceDataset(index_path, dataset_root, split="train")
    selected = select_single_positive_training_slice(index_path)
    dataset_index = resolve_selected_dataset_index(dataset, selected)
    image, mask = ImageNetNormalizedDataset(dataset)[dataset_index]
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    if tuple(image.shape) != (1, 3, 256, 256):
        raise ValueError(f"Expected image shape (1, 3, 256, 256), got {tuple(image.shape)}.")
    if tuple(mask.shape) != (1, 1, 256, 256):
        raise ValueError(f"Expected mask shape (1, 1, 256, 256), got {tuple(mask.shape)}.")
    if not torch.isfinite(image).all():
        raise ValueError("Selected MRI slice contains non-finite image values.")
    if not torch.isfinite(mask).all():
        raise ValueError("Selected mask contains non-finite values.")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Selected mask must remain binary after preprocessing.")
    tumor_pixels = int(mask.bool().sum().item())
    if tumor_pixels <= 0:
        raise ValueError("Selected slice does not contain any tumor pixels after preprocessing.")
    if tumor_pixels >= mask.numel():
        raise ValueError("Selected mask contains no background pixels; a binary tumor-positive slice should also include background.")

    return image, mask


def summarize_logit_state(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, Any]:
    """Return finite logit/probability summaries for a single image."""
    if logits.ndim != 4 or target.ndim != 4:
        raise ValueError(f"Expected logits and target of shape (B, 1, H, W), got {tuple(logits.shape)} and {tuple(target.shape)}.")
    if logits.shape != target.shape:
        raise ValueError(f"Expected matching logits and target shape, got {tuple(logits.shape)} and {tuple(target.shape)}.")

    logits = logits.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    probabilities = torch.sigmoid(logits)
    pred_mask = probabilities >= threshold
    target_mask = target.bool()

    tumor_probabilities = probabilities[target_mask]
    background_probabilities = probabilities[~target_mask]
    metrics = whole_tumor_metrics(logits, target, threshold=threshold)
    predicted_tumor_pixels = int(pred_mask.sum().item())
    all_finite = bool(
        torch.isfinite(logits).all().item()
        and torch.isfinite(probabilities).all().item()
        and torch.isfinite(tumor_probabilities).all().item()
        and torch.isfinite(background_probabilities).all().item()
    )
    return {
        "min_logit": float(logits.min().item()),
        "mean_logit": float(logits.mean().item()),
        "max_logit": float(logits.max().item()),
        "mean_probability": float(probabilities.mean().item()),
        "max_probability": float(probabilities.max().item()),
        "mean_probability_in_tumor": float(tumor_probabilities.mean().item()) if tumor_probabilities.numel() > 0 else 0.0,
        "mean_probability_outside_tumor": float(background_probabilities.mean().item()) if background_probabilities.numel() > 0 else 0.0,
        "predicted_tumor_pixels_at_threshold_0_5": predicted_tumor_pixels,
        "dice": float(metrics["dice"]),
        "iou": float(metrics["iou"]),
        "all_finite": all_finite,
    }


def _print_state(label: str, logits: torch.Tensor, target: torch.Tensor, criterion: FocalTverskyLoss) -> dict[str, Any]:
    """Evaluate one snapshot and print a compact summary of the single-slice state."""
    with torch.no_grad():
        logits = logits.detach()
        target = target.detach()
        loss = criterion(logits, target)
        stats = summarize_logit_state(logits, target)

    print(f"\n=== {label} ===")
    print(f"Loss: {loss.item():.6f}")
    print(
        "Logits: min={min_logit:.6f} mean={mean_logit:.6f} max={max_logit:.6f}".format(
            min_logit=stats["min_logit"],
            mean_logit=stats["mean_logit"],
            max_logit=stats["max_logit"],
        )
    )
    print(
        "Probabilities: mean={mean_probability:.6f} max={max_probability:.6f} "
        "tumor_mean={tumor_mean:.6f} bg_mean={bg_mean:.6f}".format(
            mean_probability=stats["mean_probability"],
            max_probability=stats["max_probability"],
            tumor_mean=stats["mean_probability_in_tumor"],
            bg_mean=stats["mean_probability_outside_tumor"],
        )
    )
    print(
        "Predicted tumor pixels (>=0.5): {count} | Dice={dice:.6f} | IoU={iou:.6f}".format(
            count=stats["predicted_tumor_pixels_at_threshold_0_5"],
            dice=stats["dice"],
            iou=stats["iou"],
        )
    )
    print(f"All finite: {stats['all_finite']}")
    return {"loss": float(loss.item()), **stats}


def run_single_slice_learning_diagnostic(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    seed: int = 42,
    learning_rate: float = 1e-4,
    steps: tuple[int, ...] = (1, 5, 10, 20, 30),
) -> dict[str, Any]:
    """Train exactly one tumor-positive MRI slice for 30 optimizer steps and print diagnostics."""
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    index_path = project_root / index_path if not Path(index_path).is_absolute() else Path(index_path)
    dataset_root = project_root / dataset_root if not Path(dataset_root).is_absolute() else Path(dataset_root)

    selected = select_single_positive_training_slice(index_path)
    dataset = BraTSNiftiSliceDataset(index_path, dataset_root, split="train")
    dataset_index = resolve_selected_dataset_index(dataset, selected)
    image, target = ImageNetNormalizedDataset(dataset)[dataset_index]
    image = image.to(device=device, dtype=torch.float32).unsqueeze(0)
    target = target.to(device=device, dtype=torch.float32).unsqueeze(0)

    if tuple(image.shape) != (1, 3, 256, 256):
        raise ValueError(f"Expected image shape (1, 3, 256, 256), got {tuple(image.shape)}.")
    if tuple(target.shape) != (1, 1, 256, 256):
        raise ValueError(f"Expected target shape (1, 1, 256, 256), got {tuple(target.shape)}.")
    if not torch.isfinite(image).all() or not torch.isfinite(target).all():
        raise ValueError("Selected image or target contains NaN/Inf values.")
    if not ((target == 0) | (target == 1)).all():
        raise ValueError("Selected target mask is not binary after preprocessing.")

    tumor_pixels = int(target.bool().sum().item())
    tumor_fraction = 100.0 * (tumor_pixels / target.numel())
    print("Selected single-positive MRI slice diagnostic")
    print(f"Data index: {index_path}")
    print(f"Patient ID: {selected['patient_id']}")
    print(f"Slice index: {selected['slice_index']}")
    print(f"Tumor pixel count: {selected['tumor_pixel_count']}")
    print(f"Processed tumor fraction: {tumor_fraction:.4f}%")
    print(f"Running on device: {device}")
    print(f"Batch size: 1")
    print(f"Learning rate: {learning_rate}")

    model = VGG19UNet(pretrained=True, freeze_encoder=True).to(device=device, dtype=torch.float32)
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Encoder was not frozen as required for the diagnostic.")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("No trainable parameters remained after freezing the encoder.")

    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        foreach=False,
    )

    target_step_set = set(int(step) for step in steps)
    checkpoint: dict[str, Any] = {"initial": None, "final": None}

    model.eval()
    checkpoint["initial"] = _print_state("Initialization", model(image), target, criterion)

    for step in range(1, 31):
        model.train()
        encoder_before = {name: parameter.detach().clone() for name, parameter in model.encoder.named_parameters()}
        decoder_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ("decoder" in name or name.startswith("output_conv"))
        }

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        if logits.shape != target.shape:
            raise ValueError(f"Expected logits shape {tuple(target.shape)}, got {tuple(logits.shape)}.")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("NaN/Inf detected in logits before backward pass.")
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected at step {step}: {loss.item()}")
        loss.backward()

        decoder_grad_norms = {
            name: float(parameter.grad.norm().item()) if parameter.grad is not None else 0.0
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ("decoder" in name or name.startswith("output_conv"))
        }
        if not any(norm > 0.0 for norm in decoder_grad_norms.values()):
            raise RuntimeError(f"No trainable decoder parameter received a non-zero gradient at step {step}.")
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.requires_grad):
            raise FloatingPointError(f"Non-finite gradient encountered at step {step}.")
        if any(parameter.grad is not None for parameter in model.encoder.parameters()):
            raise RuntimeError("Frozen encoder parameters unexpectedly received gradients during the single-slice diagnostic.")

        print(f"\n--- Before optimizer update step {step} ---")
        with torch.no_grad():
            model.eval()
            _print_state(f"Before update {step}", model(image), target, criterion)
        optimizer.step()

        model.eval()
        decoder_after = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ("decoder" in name or name.startswith("output_conv"))
        }
        decoder_changed = any(
            not torch.equal(decoder_before[name], decoder_after[name])
            for name in decoder_before
        )
        if not decoder_changed:
            raise RuntimeError(f"No trainable decoder parameter changed after step {step}.")
        encoder_unchanged = all(
            torch.equal(encoder_before[name], parameter.detach())
            for name, parameter in model.encoder.named_parameters()
        )
        if not encoder_unchanged:
            raise RuntimeError(f"Frozen encoder parameters changed after step {step}.")

        if step in target_step_set:
            checkpoint[f"step_{step}"] = _print_state(f"After update {step}", model(image), target, criterion)

    final_eval = _print_state("Final state", model(image), target, criterion)
    checkpoint["final"] = final_eval
    print("\nFinal summary:")
    initial_loss = checkpoint["initial"]["loss"]
    final_loss = final_eval["loss"]
    initial_predicted = checkpoint["initial"]["predicted_tumor_pixels_at_threshold_0_5"]
    final_predicted = final_eval["predicted_tumor_pixels_at_threshold_0_5"]
    print(f"Initial loss: {initial_loss:.6f}, final loss: {final_loss:.6f}")
    print(f"Initial predicted tumor pixels: {initial_predicted}, final predicted tumor pixels: {final_predicted}")

    if final_loss < initial_loss and final_predicted >= initial_predicted:
        print("PASS: evidence of learning was observed on the selected single MRI slice.")
    else:
        print("FAIL: the selected MRI slice did not show clear learning evidence within 30 steps.")

    return {
        "selected_record": selected,
        "image_shape": tuple(image.shape),
        "mask_shape": tuple(target.shape),
        "initial": checkpoint["initial"],
        "final": final_eval,
        "trainable_decoder_changed": True,
        "encoder_remained_frozen": True,
        "device": str(device),
    }


if __name__ == "__main__":
    try:
        run_single_slice_learning_diagnostic()
    except (FileNotFoundError, OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        print("Single-slice MRI learning diagnostic: FAILURE")
        print(str(error))
        sys.exit(1)
