"""Safe, bounded 2D training and validation pipeline for BraTS MRI slices.

This stage intentionally keeps the scope narrow and memory-friendly:
- train and validation with the verified CSV split files
- ImageNet normalization applied through the existing helper wrapper
- one VGG19 encoder with frozen weights and trainable decoder/head
- Adam optimizer with lr=1e-4
- Focal Tversky loss with alpha=0.7, beta=0.3, gamma=0.75
- validation metrics computed with the project metric helpers
- no automatic full-dataset run; all training is bounded by explicit commands

This is a preliminary training script intended for small smoke-style experiments.
It is not a production training loop or a full dataset trainer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Import the project modules using the same patterns as the earlier smoke-test.
try:
    from src.dataset_nifti_2d import BraTSNiftiSliceDataset
    from src.input_normalization import ImageNetNormalizedDataset
    from src.losses import FocalTverskyLoss, build_loss_criterion
    from src.metrics import whole_tumor_metrics, tumor_detection_diagnostics
    from src.model_vgg19_unet import VGG19UNet
    from src.plot_training_history import generate_training_plots
except ModuleNotFoundError:  # pragma: no cover - support direct script execution.
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dataset_nifti_2d import BraTSNiftiSliceDataset
    from src.input_normalization import ImageNetNormalizedDataset
    from src.losses import FocalTverskyLoss, build_loss_criterion
    from src.metrics import whole_tumor_metrics, tumor_detection_diagnostics
    from src.model_vgg19_unet import VGG19UNet
    from src.plot_training_history import generate_training_plots


def set_seed(seed: int = 42) -> None:
    """Set a reproducible seed for training and test stability.

    Note: Apple MPS kernels can still show small nondeterminism for some ops, so
    exact floating-point outputs may vary slightly across runs even with a fixed seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed_all"):
        torch.mps.manual_seed_all(seed)


def choose_device() -> torch.device:
    """Use MPS when available; otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_train_val_datasets(index_path: str | Path, dataset_root: str | Path):
    """Build the normalized train and validation datasets for bounded experiments."""
    project_root = Path(__file__).resolve().parents[1]
    index_path = Path(index_path) if not isinstance(index_path, Path) else index_path
    dataset_root = Path(dataset_root) if not isinstance(dataset_root, Path) else dataset_root
    if not index_path.is_absolute():
        index_path = project_root / index_path
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    train_dataset = ImageNetNormalizedDataset(BraTSNiftiSliceDataset(index_path, dataset_root, split="train"))
    val_dataset = ImageNetNormalizedDataset(BraTSNiftiSliceDataset(index_path, dataset_root, split="validation"))
    return train_dataset, val_dataset


def build_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0):
    """Build a small, CPU/MPS-friendly DataLoader without multiprocessing."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if num_workers != 0:
        raise ValueError("This bounded smoke-style experiment requires num_workers=0.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _unwrap_dataset_records(dataset) -> list[dict[str, Any]]:
    """Return the underlying CSV-backed dataset records for a wrapped dataset.

    The training pipeline intentionally uses ImageNetNormalizedDataset as an
    opt-in wrapper around BraTSNiftiSliceDataset. Metadata selection must read
    the wrapped dataset's `.records` without bypassing the normalization layer.
    """
    current = dataset
    seen: set[int] = set()
    while not hasattr(current, "records"):
        if not hasattr(current, "dataset"):
            raise AttributeError(
                f"Balanced smoke-test selection requires a dataset with a `.records` attribute or a wrapper exposing `.dataset`; got {type(dataset).__name__}."
            )
        if id(current) in seen:
            raise RuntimeError("Detected a circular dataset wrapper chain while resolving metadata.")
        seen.add(id(current))
        current = current.dataset
    return current.records


def select_balanced_smoke_indices(dataset, split: str, seed: int = 42) -> list[int]:
    """Select one positive and one negative sample for a given split.

    The normalized wrapper keeps the same integer indexing contract as the
    underlying BraTS dataset, so the selected indices remain valid for the wrapped
    dataset while metadata is read from the CSV-backed inner dataset.
    """
    if split not in {"train", "validation"}:
        raise ValueError(f"Balanced smoke-test selection only supports train and validation splits, got {split!r}.")
    records = _unwrap_dataset_records(dataset)
    candidates = list(enumerate(records))
    positive = []
    negative = []
    for dataset_index, record in candidates:
        if record.get("split") != split:
            continue
        tumor_count = int(record.get("tumor_pixel_count", -1))
        if tumor_count > 0:
            positive.append(dataset_index)
        elif tumor_count == 0:
            negative.append(dataset_index)
    if len(positive) == 0:
        raise ValueError(f"Balanced smoke-test requires at least one positive training sample in split {split!r}.")
    if len(negative) == 0:
        raise ValueError(f"Balanced smoke-test requires at least one negative training sample in split {split!r}.")
    rng = random.Random(seed)
    chosen = rng.sample(positive, 1) + rng.sample(negative, 1)
    assert len(set(chosen)) == 2, "Selected balanced split samples must be unique."
    return chosen


def build_balanced_smoke_datasets(train_dataset, val_dataset, seed: int = 42):
    """Return the exact two-sample train/validation subsets for the deterministic smoke test."""
    train_indices = select_balanced_smoke_indices(train_dataset, "train", seed=seed)
    val_indices = select_balanced_smoke_indices(val_dataset, "validation", seed=seed)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    return train_subset, val_subset, train_indices, val_indices


def select_stratified_bounded_indices(dataset, split: str, *, seed: int = 42, patient_count: int = 12, sample_limit: int | None = None) -> list[int]:
    """Select a deterministic bounded sample set from the dataset metadata.

    For training, select exactly one positive and one negative slice from each of
    the first selected patients. For validation, keep the first N records in CSV order
    without enforcing artificial class balance.
    """
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported split {split!r}; expected 'train' or 'validation'.")

    records = _unwrap_dataset_records(dataset)
    if split == "train":
        by_patient: dict[str, list[int]] = {}
        for dataset_index, record in enumerate(records):
            if record.get("split") != "train":
                continue
            by_patient.setdefault(record["patient_id"], []).append(dataset_index)
        if patient_count <= 0:
            raise ValueError(f"patient_count must be positive, got {patient_count}.")
        if len(by_patient) < patient_count:
            raise ValueError(
                f"Training dataset only has {len(by_patient)} patients; cannot select {patient_count} patients."
            )

        patient_ids = sorted(by_patient)
        rng = random.Random(seed)
        rng.shuffle(patient_ids)
        selected: list[int] = []
        for patient_id in patient_ids[:patient_count]:
            patient_indices = by_patient[patient_id]
            positive_indices = [
                index for index in patient_indices if int(records[index].get("tumor_pixel_count", 0)) > 0
            ]
            negative_indices = [
                index for index in patient_indices if int(records[index].get("tumor_pixel_count", 0)) == 0
            ]
            if not positive_indices or not negative_indices:
                raise ValueError(
                    f"Patient {patient_id} does not have both positive and negative training samples. "
                    f"Positive={len(positive_indices)}, Negative={len(negative_indices)}."
                )
            selected.append(rng.choice(positive_indices))
            selected.append(rng.choice(negative_indices))
        if len(selected) != 2 * patient_count:
            raise ValueError(f"Expected {2 * patient_count} training indices, got {len(selected)}.")
        if len(set(selected)) != len(selected):
            raise ValueError("Stratified training selection must not contain duplicated dataset indices.")
        return selected

    validation_indices = [
        dataset_index for dataset_index, record in enumerate(records) if record.get("split") == "validation"
    ]
    if sample_limit is None:
        sample_limit = min(len(validation_indices), 24)
    if sample_limit < 0 or sample_limit > len(validation_indices):
        raise ValueError(
            f"sample_limit must be between 0 and {len(validation_indices)}, got {sample_limit}."
        )
    return validation_indices[:sample_limit]


def build_stratified_bounded_datasets(
    train_dataset,
    val_dataset,
    *,
    seed: int = 42,
    train_patient_count: int = 12,
    validation_sample_limit: int = 24,
):
    """Build a deterministic bounded train/validation subset using metadata only."""
    train_indices = select_stratified_bounded_indices(
        train_dataset,
        "train",
        seed=seed,
        patient_count=train_patient_count,
    )
    val_indices = select_stratified_bounded_indices(
        val_dataset,
        "validation",
        seed=seed,
        sample_limit=validation_sample_limit,
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    return train_subset, val_subset, train_indices, val_indices


def _ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} became NaN or infinity during the bounded experiment.")


def _final_output_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the actual final segmentation output layer used by the model."""
    conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("This model does not expose a final 2D convolution output layer.")
    return conv_layers[-1]


def ensure_finite_decoder_gradients(model: torch.nn.Module) -> None:
    """Fail before optimizer.step if any decoder or output-layer gradient becomes non-finite."""
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("encoder"):
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"decoder gradients became non-finite in parameter {name}.")


def summarize_training_dynamics(logits: torch.Tensor, targets: torch.Tensor, criterion) -> dict[str, dict[str, float | int]]:
    """Aggregate per-slice training dynamics from the current batch before optimizer.step."""
    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(f"Expected 4D logits and targets, got {tuple(logits.shape)} and {tuple(targets.shape)}.")
    if logits.shape != targets.shape:
        raise ValueError(f"Expected logits and targets to match, got {tuple(logits.shape)} and {tuple(targets.shape)}.")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Training targets must be binary 0/1 masks.")

    def empty_group() -> dict[str, float | int]:
        return {
            "num_slices": 0,
            "mean_focal_tversky_loss": 0.0,
            "mean_predicted_tumor_probability": 0.0,
            "mean_probability_inside_ground_truth_tumors": 0.0,
            "mean_probability_outside_ground_truth_tumors": 0.0,
            "max_predicted_probability": 0.0,
            "num_predicted_tumor_positive_slices_at_threshold_0.5": 0,
            "num_predicted_tumor_pixels_at_threshold_0.5": 0,
            "mean_dice_on_tumor_positive_slices": 0.0,
        }

    positive = empty_group()
    negative = empty_group()
    probabilities = torch.sigmoid(logits).to(dtype=torch.float32)

    for sample_index in range(logits.shape[0]):
        sample_logits = logits[sample_index : sample_index + 1]
        sample_target = targets[sample_index : sample_index + 1]
        sample_probability = probabilities[sample_index : sample_index + 1]
        sample_loss = float(criterion(sample_logits, sample_target).detach().item())
        sample_predicted_mask = sample_probability >= 0.5
        metric_group = positive if int(sample_target.sum().item()) > 0 else negative
        metric_group["num_slices"] += 1
        metric_group["mean_focal_tversky_loss"] += sample_loss
        metric_group["mean_predicted_tumor_probability"] += float(sample_probability.mean().item())
        if int(sample_target.sum().item()) > 0:
            inside = sample_probability[sample_target > 0]
            outside = sample_probability[sample_target == 0]
            metric_group["mean_probability_inside_ground_truth_tumors"] += float(inside.mean().item()) if inside.numel() > 0 else 0.0
            metric_group["mean_probability_outside_ground_truth_tumors"] += float(outside.mean().item()) if outside.numel() > 0 else 0.0
            metric_group["mean_dice_on_tumor_positive_slices"] += whole_tumor_metrics(sample_logits, sample_target, threshold=0.5)["dice"]
        else:
            outside = sample_probability[sample_target == 0]
            metric_group["mean_probability_inside_ground_truth_tumors"] += 0.0
            metric_group["mean_probability_outside_ground_truth_tumors"] += float(outside.mean().item()) if outside.numel() > 0 else 0.0
        metric_group["max_predicted_probability"] = max(float(metric_group["max_predicted_probability"]), float(sample_probability.max().item()))
        predicted_pixels = int(sample_predicted_mask.sum().item())
        metric_group["num_predicted_tumor_positive_slices_at_threshold_0.5"] += int(predicted_pixels > 0)
        metric_group["num_predicted_tumor_pixels_at_threshold_0.5"] += predicted_pixels

    for group_name in ("positive", "negative"):
        group = positive if group_name == "positive" else negative
        count = int(group["num_slices"])
        if count == 0:
            continue
        group["mean_focal_tversky_loss"] = float(group["mean_focal_tversky_loss"] / count)
        group["mean_predicted_tumor_probability"] = float(group["mean_predicted_tumor_probability"] / count)
        group["mean_probability_inside_ground_truth_tumors"] = float(group["mean_probability_inside_ground_truth_tumors"] / count)
        group["mean_probability_outside_ground_truth_tumors"] = float(group["mean_probability_outside_ground_truth_tumors"] / count)
        group["mean_dice_on_tumor_positive_slices"] = float(group["mean_dice_on_tumor_positive_slices"] / count) if group_name == "positive" and count > 0 else 0.0

    return {"positive": positive, "negative": negative}


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name.startswith("encoder"):
            continue
        norms[name] = float(parameter.grad.detach().float().norm(2).item())
    return norms


def _epoch_summary_string(epoch: int, summary: dict[str, dict[str, float | int]]) -> str:
    positive = summary["positive"]
    negative = summary["negative"]
    return (
        f"Epoch {epoch} pre-update training dynamics: "
        f"positive_slices={positive['num_slices']} | negative_slices={negative['num_slices']} | "
        f"positive_mean_loss={positive['mean_focal_tversky_loss']:.6f} | negative_mean_loss={negative['mean_focal_tversky_loss']:.6f} | "
        f"positive_mean_prob={positive['mean_predicted_tumor_probability']:.6f} | negative_mean_prob={negative['mean_predicted_tumor_probability']:.6f} | "
        f"positive_mean_prob_inside={positive['mean_probability_inside_ground_truth_tumors']:.6f} | positive_mean_prob_outside={positive['mean_probability_outside_ground_truth_tumors']:.6f} | "
        f"positive_max_prob={positive['max_predicted_probability']:.6f} | positive_predicted_tumor_slices={positive['num_predicted_tumor_positive_slices_at_threshold_0.5']} | "
        f"positive_predicted_tumor_pixels={positive['num_predicted_tumor_pixels_at_threshold_0.5']} | positive_mean_dice={positive['mean_dice_on_tumor_positive_slices']:.6f}"
    )


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, max_batches: int | None = None, diagnose_training_dynamics: bool = False, epoch: int | None = None):
    """Run one training epoch for the current split and return averaged loss."""
    model.train()
    total_loss = 0.0
    seen = 0
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters were found for optimization.")

    epoch_positive = {"num_slices": 0, "loss_sum": 0.0, "prob_sum": 0.0, "inside_sum": 0.0, "outside_sum": 0.0, "max_prob": 0.0, "predicted_positive_slices": 0, "predicted_tumor_pixels": 0, "dice_sum": 0.0}
    epoch_negative = {"num_slices": 0, "loss_sum": 0.0, "prob_sum": 0.0, "inside_sum": 0.0, "outside_sum": 0.0, "max_prob": 0.0, "predicted_positive_slices": 0, "predicted_tumor_pixels": 0, "dice_sum": 0.0}

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
            raise ValueError("Training images or targets contain NaN or infinity.")
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("Training targets must be binary 0/1 masks.")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        _ensure_finite("training logits", logits)
        if logits.shape != targets.shape:
            raise ValueError(f"Expected logits shape {tuple(targets.shape)}, got {tuple(logits.shape)}.")

        if diagnose_training_dynamics:
            batch_summary = summarize_training_dynamics(logits, targets, criterion)
            for group_name, group_summary in batch_summary.items():
                count = int(group_summary["num_slices"])
                if count == 0:
                    continue
                target_group = epoch_positive if group_name == "positive" else epoch_negative
                target_group["num_slices"] += count
                target_group["loss_sum"] += float(group_summary["mean_focal_tversky_loss"]) * count
                target_group["prob_sum"] += float(group_summary["mean_predicted_tumor_probability"]) * count
                target_group["inside_sum"] += float(group_summary["mean_probability_inside_ground_truth_tumors"]) * count
                target_group["outside_sum"] += float(group_summary["mean_probability_outside_ground_truth_tumors"]) * count
                target_group["max_prob"] = max(target_group["max_prob"], float(group_summary["max_predicted_probability"]))
                target_group["predicted_positive_slices"] += int(group_summary["num_predicted_tumor_positive_slices_at_threshold_0.5"])
                target_group["predicted_tumor_pixels"] += int(group_summary["num_predicted_tumor_pixels_at_threshold_0.5"])
                target_group["dice_sum"] += float(group_summary["mean_dice_on_tumor_positive_slices"]) * count

        loss = criterion(logits, targets)
        _ensure_finite("training loss", loss)
        if loss.ndim != 0:
            raise ValueError(f"Training loss must be a scalar, got shape {tuple(loss.shape)}.")
        if not loss.requires_grad:
            raise RuntimeError("Training loss does not require gradients; training cannot proceed.")

        loss.backward()
        ensure_finite_decoder_gradients(model)
        for parameter in trainable_parameters:
            if parameter.grad is not None:
                _ensure_finite(f"gradient for {parameter.shape}", parameter.grad)
        if diagnose_training_dynamics and (batch_index + 1) in {1, 24, 72, 144}:
            gradient_norms = _gradient_norms(model)
            if gradient_norms:
                formatted = ", ".join(f"{name}={norm:.6f}" for name, norm in gradient_norms.items())
                print(f"Epoch {epoch} step {batch_index + 1} decoder gradient norms (pre-update): {formatted}")
        optimizer.step()

        total_loss += float(loss.item()) * images.shape[0]
        seen += images.shape[0]

    if seen == 0:
        raise ValueError("No training batches were evaluated; increase --max-train-batches.")

    if diagnose_training_dynamics:
        final_summary = {"positive": {}, "negative": {}}
        for group_name, group in (("positive", epoch_positive), ("negative", epoch_negative)):
            count = int(group["num_slices"])
            if count == 0:
                final_summary[group_name] = {
                    "num_slices": 0,
                    "mean_focal_tversky_loss": 0.0,
                    "mean_predicted_tumor_probability": 0.0,
                    "mean_probability_inside_ground_truth_tumors": 0.0,
                    "mean_probability_outside_ground_truth_tumors": 0.0,
                    "max_predicted_probability": 0.0,
                    "num_predicted_tumor_positive_slices_at_threshold_0.5": 0,
                    "num_predicted_tumor_pixels_at_threshold_0.5": 0,
                    "mean_dice_on_tumor_positive_slices": 0.0,
                }
                continue
            final_summary[group_name] = {
                "num_slices": count,
                "mean_focal_tversky_loss": float(group["loss_sum"] / count),
                "mean_predicted_tumor_probability": float(group["prob_sum"] / count),
                "mean_probability_inside_ground_truth_tumors": float(group["inside_sum"] / count),
                "mean_probability_outside_ground_truth_tumors": float(group["outside_sum"] / count),
                "max_predicted_probability": float(group["max_prob"]),
                "num_predicted_tumor_positive_slices_at_threshold_0.5": int(group["predicted_positive_slices"]),
                "num_predicted_tumor_pixels_at_threshold_0.5": int(group["predicted_tumor_pixels"]),
                "mean_dice_on_tumor_positive_slices": float(group["dice_sum"] / count) if group_name == "positive" else 0.0,
            }
        print(_epoch_summary_string(epoch or 1, final_summary))

    return total_loss / seen, seen


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    max_batches: int | None = None,
    diagnose_predictions: bool = False,
):
    """Run one validation epoch and optionally accumulate tumor-detection diagnostics."""
    model.eval()
    total_loss = 0.0
    metric_totals = {name: 0.0 for name in ("dice", "iou", "precision", "recall")}
    total_slices = 0
    tumor_positive_slices = 0
    tumor_negative_slices = 0
    tumor_positive_dice_sum = 0.0
    tumor_positive_iou_sum = 0.0
    tumor_positive_recall_sum = 0.0
    tumor_positive_count = 0
    total_predicted_tumor_pixels = 0
    total_ground_truth_tumor_pixels = 0
    true_positive_pixels = 0
    false_positive_pixels = 0
    false_negative_pixels = 0
    pred_positive_slices = 0
    pred_negative_slices = 0
    false_positive_slices = 0
    empty_gt_empty_pred = 0
    threshold_counts = {value: 0 for value in (0.1, 0.3, 0.5)}
    background_threshold_counts = {value: 0 for value in (0.1, 0.3, 0.5)}
    hist_all = torch.zeros(1024, dtype=torch.int64)
    hist_tumor = torch.zeros(1024, dtype=torch.int64)
    hist_background = torch.zeros(1024, dtype=torch.int64)
    total_probability_count = 0
    total_tumor_probability_count = 0
    total_background_probability_count = 0
    diagnostics = None

    def summarize_histogram(histogram: torch.Tensor, total_count: int) -> dict[str, float]:
        if total_count == 0:
            return {
                "min_probability": float("nan"),
                "mean_probability": float("nan"),
                "max_probability": float("nan"),
                "probability_50th_percentile": float("nan"),
                "probability_90th_percentile": float("nan"),
                "probability_99th_percentile": float("nan"),
            }

        values = histogram.float().tolist()
        total = float(total_count)
        nonzero_mask = [value for value in values if value > 0]
        if not nonzero_mask:
            return {
                "min_probability": float("nan"),
                "mean_probability": float("nan"),
                "max_probability": float("nan"),
                "probability_50th_percentile": float("nan"),
                "probability_90th_percentile": float("nan"),
                "probability_99th_percentile": float("nan"),
            }

        def quantile(q: float) -> float:
            target = q * total
            running = 0.0
            for bin_index, count in enumerate(values):
                running += count
                if running >= target:
                    return float((bin_index + 0.5) / 1024.0)
            return 1.0

        mean_probability = float(sum(index * value for index, value in enumerate(values)) / total)
        mean_probability /= 1024.0
        return {
            "min_probability": float(next((index / 1024.0 for index, count in enumerate(values) if count > 0), 0.0)),
            "mean_probability": mean_probability,
            "max_probability": float((max((index for index, count in enumerate(values) if count > 0), default=0) + 0.5) / 1024.0),
            "probability_50th_percentile": quantile(0.50),
            "probability_90th_percentile": quantile(0.90),
            "probability_99th_percentile": quantile(0.99),
        }

    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            _ensure_finite("validation images", images)
            _ensure_finite("validation targets", targets)
            if not ((targets == 0) | (targets == 1)).all():
                raise ValueError("Validation targets must be binary 0/1 masks.")

            logits = model(images)
            _ensure_finite("validation logits", logits)
            if logits.shape != targets.shape:
                raise ValueError(f"Expected logits shape {tuple(targets.shape)}, got {tuple(logits.shape)}.")

            loss = criterion(logits, targets)
            _ensure_finite("validation loss", loss)
            total_loss += float(loss.item()) * images.shape[0]

            if diagnose_predictions:
                probabilities = torch.sigmoid(logits)
                probability_flat = probabilities.reshape(-1).to(dtype=torch.float32).cpu()
                target_flat = targets.reshape(-1).to(dtype=torch.float32).cpu()
                if probability_flat.numel() > 0:
                    hist_all += torch.bincount(
                        (probability_flat * 1024.0).clamp_(0, 1023).to(torch.int64),
                        minlength=1024,
                    )
                    total_probability_count += probability_flat.numel()
                    tumor_mask = target_flat > 0
                    if tumor_mask.any():
                        tumor_probs = probability_flat[tumor_mask]
                        hist_tumor += torch.bincount(
                            (tumor_probs * 1024.0).clamp_(0, 1023).to(torch.int64),
                            minlength=1024,
                        )
                        total_tumor_probability_count += tumor_probs.numel()
                        for threshold_value in (0.1, 0.3, 0.5):
                            threshold_counts[threshold_value] += int((tumor_probs >= threshold_value).sum().item())
                    if (~tumor_mask).any():
                        background_probs = probability_flat[~tumor_mask]
                        hist_background += torch.bincount(
                            (background_probs * 1024.0).clamp_(0, 1023).to(torch.int64),
                            minlength=1024,
                        )
                        total_background_probability_count += background_probs.numel()
                        for threshold_value in (0.1, 0.3, 0.5):
                            background_threshold_counts[threshold_value] += int((background_probs >= threshold_value).sum().item())

            for sample_index in range(images.shape[0]):
                sample_logits = logits[sample_index : sample_index + 1]
                sample_target = targets[sample_index : sample_index + 1]
                metrics = whole_tumor_metrics(sample_logits, sample_target)
                for name in metric_totals:
                    metric_totals[name] += metrics[name]
                total_slices += 1
                pred_mask = torch.sigmoid(sample_logits) >= 0.5
                if sample_target.sum().item() > 0:
                    tumor_positive_slices += 1
                    tumor_positive_dice_sum += metrics["dice"]
                    tumor_positive_iou_sum += metrics["iou"]
                    tumor_positive_recall_sum += metrics["recall"]
                    tumor_positive_count += 1
                else:
                    tumor_negative_slices += 1
                if pred_mask.sum().item() > 0:
                    pred_positive_slices += 1
                else:
                    pred_negative_slices += 1
                if sample_target.sum().item() == 0 and pred_mask.sum().item() > 0:
                    false_positive_slices += 1
                if sample_target.sum().item() == 0 and pred_mask.sum().item() == 0:
                    empty_gt_empty_pred += 1
                total_predicted_tumor_pixels += int(pred_mask.sum().item())
                total_ground_truth_tumor_pixels += int(sample_target.sum().item())
                true_positive_pixels += int(torch.logical_and(pred_mask, sample_target.to(dtype=torch.bool)).sum().item())
                false_positive_pixels += int(torch.logical_and(pred_mask, torch.logical_not(sample_target.to(dtype=torch.bool))).sum().item())
                false_negative_pixels += int(torch.logical_and(torch.logical_not(pred_mask), sample_target.to(dtype=torch.bool)).sum().item())

    if total_slices == 0:
        raise ValueError("No validation slices were evaluated; increase --max-val-batches.")

    metrics = {name: metric_totals[name] / total_slices for name in metric_totals}
    tumor_positive_dice = tumor_positive_dice_sum / tumor_positive_count if tumor_positive_count else 0.0
    result = {
        "loss": total_loss / total_slices,
        "metrics": metrics,
        "num_slices": total_slices,
        "tumor_positive_slices": tumor_positive_slices,
        "tumor_negative_slices": tumor_negative_slices,
        "tumor_positive_dice": tumor_positive_dice,
    }
    if diagnose_predictions:
        diagnostics = {
            "total_validation_slices": total_slices,
            "ground_truth_tumor_positive_slices": tumor_positive_slices,
            "ground_truth_tumor_negative_slices": tumor_negative_slices,
            "predicted_tumor_positive_slices": pred_positive_slices,
            "predicted_tumor_negative_slices": pred_negative_slices,
            "total_predicted_tumor_pixels": total_predicted_tumor_pixels,
            "total_ground_truth_tumor_pixels": total_ground_truth_tumor_pixels,
            "true_positive_pixels": true_positive_pixels,
            "false_positive_pixels": false_positive_pixels,
            "false_negative_pixels": false_negative_pixels,
            "mean_dice_on_tumor_positive_ground_truth_slices": tumor_positive_dice_sum / tumor_positive_count if tumor_positive_count else 0.0,
            "mean_iou_on_tumor_positive_ground_truth_slices": tumor_positive_iou_sum / tumor_positive_count if tumor_positive_count else 0.0,
            "mean_recall_on_tumor_positive_ground_truth_slices": tumor_positive_recall_sum / tumor_positive_count if tumor_positive_count else 0.0,
            "false_positive_slices_among_ground_truth_negative_slices": false_positive_slices,
            "empty_ground_truth_and_empty_prediction_pairs": empty_gt_empty_pred,
            "all_background_baseline_dice": float(tumor_negative_slices / total_slices),
            "all_background_pixel_aggregated_dice": 1.0 if total_ground_truth_tumor_pixels == 0 else 0.0,
            "zero_predicted_tumor_pixels": total_predicted_tumor_pixels == 0,
            "all_probability_summary": summarize_histogram(hist_all, total_probability_count),
            "tumor_probability_summary": summarize_histogram(hist_tumor, total_tumor_probability_count),
            "background_probability_summary": summarize_histogram(hist_background, total_background_probability_count),
        }
        for threshold_value in (0.1, 0.3, 0.5):
            diagnostics[f"tumor_pixels_predicted_at_threshold_{threshold_value}"] = threshold_counts[threshold_value]
            diagnostics[f"background_pixels_predicted_at_threshold_{threshold_value}"] = background_threshold_counts[threshold_value]
        if diagnostics["zero_predicted_tumor_pixels"]:
            print("Warning: zero tumor pixels were predicted across the entire validation subset; the model may be predicting only background.")
        print("Validation diagnosis: total slices=%d | gt positive=%d | gt negative=%d | pred positive=%d | pred negative=%d | pred tumor pixels=%d | gt tumor pixels=%d | tp=%d | fp=%d | fn=%d" % (
            diagnostics["total_validation_slices"],
            diagnostics["ground_truth_tumor_positive_slices"],
            diagnostics["ground_truth_tumor_negative_slices"],
            diagnostics["predicted_tumor_positive_slices"],
            diagnostics["predicted_tumor_negative_slices"],
            diagnostics["total_predicted_tumor_pixels"],
            diagnostics["total_ground_truth_tumor_pixels"],
            diagnostics["true_positive_pixels"],
            diagnostics["false_positive_pixels"],
            diagnostics["false_negative_pixels"],
        ))
        print(
            "Validation diagnosis: mean Dice on tumor-positive ground-truth slices=%0.4f | mean IoU=%0.4f | mean Recall=%0.4f | false-positive slices among negative GT=%d | empty GT/empty pred pairs=%d" % (
                diagnostics["mean_dice_on_tumor_positive_ground_truth_slices"],
                diagnostics["mean_iou_on_tumor_positive_ground_truth_slices"],
                diagnostics["mean_recall_on_tumor_positive_ground_truth_slices"],
                diagnostics["false_positive_slices_among_ground_truth_negative_slices"],
                diagnostics["empty_ground_truth_and_empty_prediction_pairs"],
            )
        )
        print(f"All-background baseline Dice for this validation subset: {diagnostics['all_background_baseline_dice']:.4f}")
        print(f"Pixel-aggregated all-background Dice (different metric): {diagnostics['all_background_pixel_aggregated_dice']:.4f}")
        print(
            "Validation probability stats: all pixels min=%0.4f mean=%0.4f max=%0.4f p50=%0.4f p90=%0.4f p99=%0.4f | "
            "tumor pixels min=%0.4f mean=%0.4f max=%0.4f p50=%0.4f p90=%0.4f p99=%0.4f | "
            "background pixels min=%0.4f mean=%0.4f max=%0.4f p50=%0.4f p90=%0.4f p99=%0.4f" % (
                diagnostics["all_probability_summary"]["min_probability"],
                diagnostics["all_probability_summary"]["mean_probability"],
                diagnostics["all_probability_summary"]["max_probability"],
                diagnostics["all_probability_summary"]["probability_50th_percentile"],
                diagnostics["all_probability_summary"]["probability_90th_percentile"],
                diagnostics["all_probability_summary"]["probability_99th_percentile"],
                diagnostics["tumor_probability_summary"]["min_probability"],
                diagnostics["tumor_probability_summary"]["mean_probability"],
                diagnostics["tumor_probability_summary"]["max_probability"],
                diagnostics["tumor_probability_summary"]["probability_50th_percentile"],
                diagnostics["tumor_probability_summary"]["probability_90th_percentile"],
                diagnostics["tumor_probability_summary"]["probability_99th_percentile"],
                diagnostics["background_probability_summary"]["min_probability"],
                diagnostics["background_probability_summary"]["mean_probability"],
                diagnostics["background_probability_summary"]["max_probability"],
                diagnostics["background_probability_summary"]["probability_50th_percentile"],
                diagnostics["background_probability_summary"]["probability_90th_percentile"],
                diagnostics["background_probability_summary"]["probability_99th_percentile"],
            )
        )
        print(
            "Validation threshold counts: tumor pixels at >=0.1=%d | >=0.3=%d | >=0.5=%d | "
            "background pixels at >=0.1=%d | >=0.3=%d | >=0.5=%d" % (
                diagnostics["tumor_pixels_predicted_at_threshold_0.1"],
                diagnostics["tumor_pixels_predicted_at_threshold_0.3"],
                diagnostics["tumor_pixels_predicted_at_threshold_0.5"],
                diagnostics["background_pixels_predicted_at_threshold_0.1"],
                diagnostics["background_pixels_predicted_at_threshold_0.3"],
                diagnostics["background_pixels_predicted_at_threshold_0.5"],
            )
        )
        print("Note: slice-averaged Dice and tumor-positive-only Dice answer different questions; the latter excludes empty ground-truth slices so it does not inflate with all-background cases.")
        result["diagnostics"] = diagnostics
    return result


def _default_history_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "outputs" / "plots" / "training_history.csv"


def _default_checkpoint_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "outputs" / "checkpoints" / "best_validation_checkpoint.pt"


def checkpoint_selection_score(validation_metrics: dict[str, float]) -> float:
    """Return the scalar used to rank validation checkpoints.

    Tumor-positive Dice is preferred when available; otherwise the global Dice
    score remains the safe fallback for backward compatibility with older runs.
    """
    if not isinstance(validation_metrics, dict):
        return 0.0
    candidate = validation_metrics.get("tumor_positive_dice")
    if candidate is not None:
        return float(candidate)
    if "dice" in validation_metrics:
        return float(validation_metrics["dice"])
    return 0.0


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    validation_loss: float,
    validation_metrics: dict[str, float],
    config: dict[str, Any],
    random_seed: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = False,
) -> None:
    """Save a small checkpoint with model weights and the most relevant experiment metadata.

    This is intentionally opt-in and refuses to overwrite an existing file unless
    the caller explicitly approves it.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {path}. Use overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_label = "development_smoke_test" if config.get("smoke_experiment", False) else "development_experiment"
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "validation_metrics": {name: float(value) for name, value in validation_metrics.items()},
        "model_config": {
            "model_name": model.__class__.__name__,
            "input_channels": 3,
            "output_channels": 1,
            "pretrained": bool(config.get("pretrained", False)),
            "freeze_encoder": bool(config.get("freeze_encoder", False)),
        },
        "training_config": {
            "epochs": int(config.get("epochs", 0)),
            "batch_size": int(config.get("batch_size", 0)),
            "max_train_batches": int(config.get("max_train_batches", 0)),
            "max_val_batches": int(config.get("max_val_batches", 0)),
            "learning_rate": float(config.get("learning_rate", 0.0)),
            "optimizer": "Adam",
            "criterion": config.get("loss_type", "focal_tversky"),
            "device": str(config.get("device", "cpu")),
            "random_seed": int(random_seed),
            "data_index_filename": Path(config.get("index", "unknown.csv")).name,
        },
        "config": config,
        "random_seed": int(random_seed),
        "data_index_filename": Path(config.get("index", "unknown.csv")).name,
        "checkpoint_selection_metric": "tumor_positive_dice",
        "tumor_positive_dice": float(validation_metrics.get("tumor_positive_dice", validation_metrics.get("dice", 0.0))),
        "artifact_label": artifact_label,
        "note": "This checkpoint is a local development/smoke-test artifact and is not a validated scientific model checkpoint.",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def append_history_csv(path: Path, row: dict[str, Any], *, overwrite: bool = False) -> None:
    """Append a single row to a CSV history file for the current run.

    When overwrite=True, the file is rewritten from scratch to make accidental
    replacement explicit rather than silent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "dice",
        "iou",
        "precision",
        "recall",
        "tumor_positive_dice",
        "train_batches",
        "val_batches",
        "validation_tumor_positive_slices",
        "validation_tumor_negative_slices",
        "elapsed_seconds",
    ]
    exists = path.exists()
    if exists and overwrite:
        mode = "w"
    elif exists and not overwrite:
        mode = "a"
    else:
        mode = "w"

    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists or overwrite:
            writer.writeheader()
        writer.writerow({
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "dice": row["dice"],
            "iou": row["iou"],
            "precision": row["precision"],
            "recall": row["recall"],
            "tumor_positive_dice": row.get("tumor_positive_dice", 0.0),
            "train_batches": row["train_batches"],
            "val_batches": row["val_batches"],
            "validation_tumor_positive_slices": row.get("validation_tumor_positive_slices", 0),
            "validation_tumor_negative_slices": row.get("validation_tumor_negative_slices", 0),
            "elapsed_seconds": row["elapsed_seconds"],
        })


def run_sampling_preflight(index_path: str | Path, *, seed: int = 42) -> bool:
    """Run a read-only metadata validation of the stratified bounded selection.

    This exits before model creation, checkpoint writing, or any MRI loading.
    """
    project_root = Path(__file__).resolve().parents[1]
    index_path = Path(index_path)
    if not index_path.is_absolute():
        index_path = project_root / index_path

    train_dataset, val_dataset = build_train_val_datasets(index_path, project_root / "data/raw/brats2020")
    train_indices = select_stratified_bounded_indices(train_dataset, "train", seed=seed, patient_count=12)
    val_indices = select_stratified_bounded_indices(val_dataset, "validation", seed=seed, sample_limit=24)

    train_records = _unwrap_dataset_records(train_dataset)
    val_records = _unwrap_dataset_records(val_dataset)

    train_patient_ids = {train_records[idx]["patient_id"] for idx in train_indices}
    val_patient_ids = {val_records[idx]["patient_id"] for idx in val_indices}
    train_positive = sum(int(train_records[idx].get("tumor_pixel_count", 0)) > 0 for idx in train_indices)
    train_negative = sum(int(train_records[idx].get("tumor_pixel_count", 0)) == 0 for idx in train_indices)
    val_positive = sum(int(val_records[idx].get("tumor_pixel_count", 0)) > 0 for idx in val_indices)
    val_negative = sum(int(val_records[idx].get("tumor_pixel_count", 0)) == 0 for idx in val_indices)

    patient_positive_counts: dict[str, int] = {}
    patient_negative_counts: dict[str, int] = {}
    for idx in train_indices:
        record = train_records[idx]
        patient_id = record["patient_id"]
        if int(record.get("tumor_pixel_count", 0)) > 0:
            patient_positive_counts[patient_id] = patient_positive_counts.get(patient_id, 0) + 1
        else:
            patient_negative_counts[patient_id] = patient_negative_counts.get(patient_id, 0) + 1

    no_overlap = not (train_patient_ids & val_patient_ids)
    no_test = all(train_records[idx]["split"] == "train" for idx in train_indices) and all(val_records[idx]["split"] == "validation" for idx in val_indices)
    no_test_patients = all(record["patient_id"] not in {"test"} for record in train_records + val_records if record["patient_id"] in train_patient_ids | val_patient_ids)
    one_each_per_patient = len(patient_positive_counts) == 12 and len(patient_negative_counts) == 12 and all(count == 1 for count in patient_positive_counts.values()) and all(count == 1 for count in patient_negative_counts.values())
    validation_order_ok = val_indices == list(range(24))
    mapping_ok = all(train_records[idx]["patient_id"] == train_records[idx]["patient_id"] for idx in train_indices)

    pass_checks = {
        "training_sample_count": len(train_indices) == 24,
        "training_patient_count": len(train_patient_ids) == 12,
        "training_positive": train_positive == 12,
        "training_negative": train_negative == 12,
        "validation_sample_count": len(val_indices) == 24,
        "validation_patient_count": len(val_patient_ids) == 4,
        "validation_positive": val_positive == 11,
        "validation_negative": val_negative == 13,
        "one_each_per_patient": one_each_per_patient,
        "no_overlap": no_overlap,
        "no_test": no_test and no_test_patients,
        "validation_order": validation_order_ok,
        "mapping_ok": mapping_ok,
    }

    print(f"Training selected patients: {len(train_patient_ids)}")
    print(f"Validation selected patients: {len(val_patient_ids)}")
    print(f"Training sample counts: positive={train_positive}, negative={train_negative}")
    print(f"Validation sample counts: positive={val_positive}, negative={val_negative}")
    print(f"Train/validation overlap: {no_overlap}")
    print(f"No test patients: {no_test and no_test_patients}")
    print(f"One positive and one negative per training patient: {one_each_per_patient}")
    print(f"Validation first-24 order preserved: {validation_order_ok}")
    print(f"Metadata-to-index mapping check: {mapping_ok}")

    result = all(pass_checks.values())
    print(f"Sampling preflight: {'PASS' if result else 'FAIL'}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments for the bounded training experiment."""
    parser = argparse.ArgumentParser(description="Bounded 2D BraTS training and validation experiment.")
    parser.add_argument("--index", type=Path, required=True, help="Path to the slice index CSV (for example data/processed/slice_index_smoke.csv).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run. Use 1 for the initial smoke experiment.")
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size; keep this at 1 for the first smoke experiment.")
    parser.add_argument("--max-train-batches", type=int, default=2, help="Maximum training batches to run in each epoch; keep this explicit and small.")
    parser.add_argument("--max-val-batches", type=int, default=2, help="Maximum validation batches to run; a small smoke run only.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate for decoder training.")
    parser.add_argument("--save-checkpoint", nargs="?", const="best_validation_checkpoint.pt", default=None, help="Optional checkpoint filename under outputs/checkpoints/.")
    parser.add_argument("--overwrite-checkpoint", action="store_true", help="Allow overwriting an existing checkpoint path.")
    parser.add_argument("--history-csv", type=Path, default=None, help="Optional path to a CSV history log written after each completed epoch.")
    parser.add_argument("--log-history", dest="history_csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional path for a multi-panel PNG training plot. Requires --history-csv.")
    parser.add_argument("--overwrite-history", action="store_true", help="Allow replacing an existing history CSV file.")
    parser.add_argument("--overwrite-plot", action="store_true", help="Allow replacing an existing plot PNG file.")
    parser.add_argument("--balanced-smoke-test", action="store_true", help="Use a deterministic balanced smoke-test subset with one positive and one negative slice in each split.")
    parser.add_argument("--stratified-bounded-sampling", action="store_true", help="Select exactly one positive and one negative training slice from each of the first 12 training patients and keep validation fixed to the first 24 validation records in CSV order.")
    parser.add_argument("--sampling-preflight", action="store_true", help="Read-only metadata validation of the bounded sampling plan without loading VGG19 or starting training.")
    parser.add_argument("--diagnose-predictions", action="store_true", help="Print optional validation diagnostics that flag all-background prediction regimes and summarize tumor-positive slices.")
    parser.add_argument("--diagnose-training-dynamics", action="store_true", help="Log lightweight pre-update training-dynamics diagnostics for positive and empty ground-truth slices, plus decoder gradient norms and output-layer deltas.")
    parser.add_argument("--loss-type", choices=("focal_tversky", "hybrid"), default="focal_tversky", help="Optional training objective: the original focal Tversky loss or the hybrid focal-Tversky+BCEWithLogits experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args(argv)

    if args.stratified_bounded_sampling and args.balanced_smoke_test:
        parser.error("Choose at most one of --stratified-bounded-sampling and --balanced-smoke-test.")
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_train_batches < 1:
        parser.error("--max-train-batches must be a positive integer.")
    if args.max_val_batches < 1:
        parser.error("--max-val-batches must be a positive integer.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be a positive number.")
    if args.index is None or not args.index.exists():
        parser.error(f"Slice index not found: {args.index}")
    if args.plot_path is not None and args.history_csv is None:
        parser.error("--plot-path requires --history-csv so there is a source CSV for the visualization.")
    return args


def run_training_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the bounded training and validation experiment configured by the CLI."""
    project_root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    device = choose_device()

    if args.balanced_smoke_test:
        if not hasattr(args, "batch_size") or args.batch_size is None:
            args.batch_size = 1
        if not hasattr(args, "max_train_batches") or args.max_train_batches is None:
            args.max_train_batches = 2
        if not hasattr(args, "max_val_batches") or args.max_val_batches is None:
            args.max_val_batches = 2
        # Preserve the user-requested epoch count while keeping the bounded smoke-test
        # sampling constraints intact. The balanced smoke subset is repeated across each
        # requested epoch; it is not re-sampled on every epoch.

    train_dataset, val_dataset = build_train_val_datasets(args.index, project_root / "data/raw/brats2020")
    if args.balanced_smoke_test:
        train_subset, val_subset, train_indices, val_indices = build_balanced_smoke_datasets(train_dataset, val_dataset, seed=args.seed)
        train_loader = build_loader(train_subset, batch_size=1, shuffle=False, num_workers=0)
        val_loader = build_loader(val_subset, batch_size=1, shuffle=False, num_workers=0)
        for split_name, record_index_list, selected_dataset in (
            ("train", train_indices, train_dataset),
            ("validation", val_indices, val_dataset),
        ):
            underlying_records = _unwrap_dataset_records(selected_dataset)
            for dataset_index in record_index_list:
                record = underlying_records[dataset_index]
                print(
                    f"Balanced smoke-test {split_name}: patient={record['patient_id']} | "
                    f"slice_index={record['slice_index']} | tumor_pixel_count={record['tumor_pixel_count']} | split={record['split']}"
                )
        train_positive = sum(1 for idx in train_indices if int(_unwrap_dataset_records(train_dataset)[idx]["tumor_pixel_count"]) > 0)
        train_negative = sum(1 for idx in train_indices if int(_unwrap_dataset_records(train_dataset)[idx]["tumor_pixel_count"]) == 0)
        val_positive = sum(1 for idx in val_indices if int(_unwrap_dataset_records(val_dataset)[idx]["tumor_pixel_count"]) > 0)
        val_negative = sum(1 for idx in val_indices if int(_unwrap_dataset_records(val_dataset)[idx]["tumor_pixel_count"]) == 0)
        if train_positive != 1 or train_negative != 1:
            raise ValueError(
                f"Balanced smoke-test train subset must contain exactly one positive and one negative slice, got positive={train_positive}, negative={train_negative}."
            )
        if val_positive != 1 or val_negative != 1:
            raise ValueError(
                f"Balanced smoke-test validation subset must contain exactly one positive and one negative slice, got positive={val_positive}, negative={val_negative}."
            )
        print("Balanced smoke-test subset validated: exactly one positive and one negative slice in each split.")
        print("Warning: balanced smoke-test metrics are developmental and not scientifically representative.")
    elif args.stratified_bounded_sampling:
        train_subset, val_subset, train_indices, val_indices = build_stratified_bounded_datasets(
            train_dataset,
            val_dataset,
            seed=args.seed,
            train_patient_count=12,
            validation_sample_limit=24,
        )
        train_loader = build_loader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_loader = build_loader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        train_records = _unwrap_dataset_records(train_dataset)
        val_records = _unwrap_dataset_records(val_dataset)
        train_patient_ids = sorted({train_records[idx]["patient_id"] for idx in train_indices})
        val_patient_ids = sorted({val_records[idx]["patient_id"] for idx in val_indices})
        train_positive = sum(1 for idx in train_indices if int(train_records[idx].get("tumor_pixel_count", 0)) > 0)
        train_negative = sum(1 for idx in train_indices if int(train_records[idx].get("tumor_pixel_count", 0)) == 0)
        val_positive = sum(1 for idx in val_indices if int(val_records[idx].get("tumor_pixel_count", 0)) > 0)
        val_negative = sum(1 for idx in val_indices if int(val_records[idx].get("tumor_pixel_count", 0)) == 0)

        print(f"Selected training patients: {len(train_patient_ids)}")
        print(f"Selected validation patients: {len(val_patient_ids)}")
        print(f"Selected training samples: tumor_positive={train_positive}, tumor_negative={train_negative}")
        print(f"Selected validation samples: tumor_positive={val_positive}, tumor_negative={val_negative}")
        if train_positive != 12 or train_negative != 12:
            raise ValueError(
                "Stratified bounded-sampling training subset must contain exactly 12 positive and 12 negative samples. "
                f"Got positive={train_positive}, negative={train_negative}."
            )
        if len(train_patient_ids) != 12 or len(val_patient_ids) != 4:
            raise ValueError(
                "Stratified bounded-sampling must keep exactly 12 training patients and 4 validation patients. "
                f"Got train_patients={len(train_patient_ids)}, val_patients={len(val_patient_ids)}."
            )
        if len(val_indices) != 24:
            raise ValueError(f"Validation subset must contain exactly 24 samples in the bounded mode, got {len(val_indices)}.")
        if train_positive != 12 or train_negative != 12:
            raise ValueError(f"Expected 12 positive and 12 negative training samples, got {train_positive}/{train_negative}.")
        print("Stratified bounded-sampling subset validated: 12 train patients with 12 positive and 12 negative samples; 24 validation records preserved in CSV order.")
    else:
        train_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = build_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VGG19UNet(pretrained=True, freeze_encoder=True).to(device=device, dtype=torch.float32)
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Frozen encoder check failed: at least one encoder parameter remains trainable.")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("No trainable parameters remain after freezing the encoder.")

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        foreach=False,
    )
    criterion = build_loss_criterion(
        args.loss_type,
        alpha=0.7,
        beta=0.3,
        gamma=0.75,
        smooth=1.0,
        pos_weight=5.0,
    )

    checkpoint_path = None
    if args.save_checkpoint is not None:
        checkpoint_path = (project_root / "outputs/checkpoints" / args.save_checkpoint).resolve()
        if checkpoint_path.exists() and not args.overwrite_checkpoint:
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}. Use --overwrite-checkpoint to replace it.")

    history_path = None
    if args.history_csv is not None:
        history_path = Path(args.history_csv)
        if not history_path.is_absolute():
            history_path = (project_root / history_path).resolve()
        if history_path.exists() and args.overwrite_history and history_path.is_file():
            history_path.unlink()
        history_path.parent.mkdir(parents=True, exist_ok=True)

    plot_path = None
    if args.plot_path is not None:
        plot_path = Path(args.plot_path)
        if not plot_path.is_absolute():
            plot_path = (project_root / plot_path).resolve()
        if plot_path.exists() and not args.overwrite_plot:
            raise FileExistsError(f"Plot already exists: {plot_path}. Use --overwrite-plot to replace it.")
        plot_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selected device: {device}")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Training batches to evaluate per epoch: {min(args.max_train_batches, len(train_loader))}")
    print(f"Validation batches to evaluate per epoch: {min(args.max_val_batches, len(val_loader))}")
    print(f"Warning: with only a few validation batches from {Path(args.index).name}, these scores are a quick bounded check and are not scientifically meaningful.")

    start_time = time.time()
    history_rows: list[dict[str, Any]] = []
    best_metrics = None
    best_selection_score = float("-inf")
    best_epoch = 0
    best_validation_loss = float("nan")
    best_model_state_dict = None
    best_optimizer_state_dict = None
    checkpoint_config = {
        "index": str(args.index),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "learning_rate": args.learning_rate,
        "pretrained": True,
        "freeze_encoder": True,
        "device": str(device),
        "loss_type": args.loss_type,
        "smoke_experiment": str(args.index).endswith("slice_index_smoke.csv"),
    }

    output_layer = _final_output_layer(model)
    output_weight_before_epoch = output_layer.weight.detach().clone()
    output_bias_before_epoch = output_layer.bias.detach().clone() if output_layer.bias is not None else None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_seen = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
            diagnose_training_dynamics=args.diagnose_training_dynamics,
            epoch=epoch,
        )
        validation_result = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_val_batches,
            diagnose_predictions=args.diagnose_predictions,
        )

        metrics = validation_result["metrics"]
        metrics_with_selection = {**metrics, "tumor_positive_dice": validation_result["tumor_positive_dice"]}
        selection_score = checkpoint_selection_score(metrics_with_selection)
        if best_metrics is None or selection_score > best_selection_score:
            best_metrics = metrics_with_selection.copy()
            best_selection_score = selection_score
            best_epoch = epoch
            best_validation_loss = validation_result["loss"]
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_result['loss']:.6f} | "
            f"dice={metrics['dice']:.4f} | iou={metrics['iou']:.4f} | "
            f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
            f"tumor_positive_dice={validation_result['tumor_positive_dice']:.4f} | "
            f"train_batches={train_seen} | val_batches={validation_result['num_slices']} | "
            f"elapsed={elapsed:.1f}s"
        )
        print(
            f"Validation summary: tumor-positive slices={validation_result['tumor_positive_slices']} | "
            f"tumor-negative slices={validation_result['tumor_negative_slices']}"
        )

        if args.diagnose_training_dynamics:
            output_layer_after = _final_output_layer(model)
            weight_delta = float((output_layer_after.weight.detach() - output_weight_before_epoch).abs().max().item())
            bias_delta = 0.0 if output_layer_after.bias is None or output_bias_before_epoch is None else float((output_layer_after.bias.detach() - output_bias_before_epoch).abs().max().item())
            print(
                f"Epoch {epoch} final output layer change: weight_max_abs_delta={weight_delta:.6f} | "
                f"bias_max_abs_delta={bias_delta:.6f}"
            )
            output_weight_before_epoch = output_layer_after.weight.detach().clone()
            if output_layer_after.bias is not None:
                output_bias_before_epoch = output_layer_after.bias.detach().clone()

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_result["loss"],
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tumor_positive_dice": validation_result["tumor_positive_dice"],
            "train_batches": train_seen,
            "val_batches": validation_result["num_slices"],
            "validation_tumor_positive_slices": validation_result["tumor_positive_slices"],
            "validation_tumor_negative_slices": validation_result["tumor_negative_slices"],
            "elapsed_seconds": elapsed,
        })
        if history_path is not None:
            append_history_csv(history_path, history_rows[-1], overwrite=args.overwrite_history)

    if history_path is not None:
        print(f"Training history CSV saved to: {history_path}")

    if plot_path is not None:
        generate_training_plots(history_path, plot_path, overwrite=args.overwrite_plot)
        print(f"Training metric plot saved to: {plot_path}")

    if checkpoint_path is not None and best_metrics is not None:
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(best_model_state_dict)
        best_optimizer = copy.deepcopy(optimizer)
        if best_optimizer_state_dict is not None:
            best_optimizer.load_state_dict(best_optimizer_state_dict)
        save_checkpoint(
            checkpoint_path,
            best_model,
            best_epoch,
            best_validation_loss,
            best_metrics,
            checkpoint_config,
            args.seed,
            optimizer=best_optimizer,
            overwrite=args.overwrite_checkpoint,
        )
        print(f"Best validation checkpoint saved to: {checkpoint_path}")

    return {
        "device": str(device),
        "epochs": args.epochs,
        "best_metrics": best_metrics,
        "history_rows": history_rows,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "history_path": str(history_path) if history_path is not None else None,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.sampling_preflight:
        ok = run_sampling_preflight(args.index, seed=args.seed)
        if not ok:
            raise SystemExit(1)
        return
    run_training_experiment(args)


if __name__ == "__main__":
    main()
