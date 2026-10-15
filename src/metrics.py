"""Binary whole-tumor segmentation metrics.

The metrics in this module are meant for BraTS-style whole-tumor segmentation:

- Dice coefficient: 2TP / (2TP + FP + FN)
- IoU: TP / (TP + FP + FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)

Each metric is computed per image, then averaged across the batch. This matches
what we want for variable per-slice tumor size and avoids hidden bias from large
images or large batch sizes.

Empty-mask policy
-----------------
This project treats tumors as binary and requires consistent handling for empty
regions:

- prediction empty and target empty -> all metrics = 1
- target contains tumor but prediction is empty -> all metrics = 0
- target empty but prediction has false positives -> Dice, IoU, and Precision = 0,
  and Recall = 1 because no positive target pixels were missed

No smoothing is used; the only time a denominator is zero is when an empty-mask
policy explicitly defines the score.
"""

from __future__ import annotations

import torch


def _validate_inputs(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the shapes, values, and threshold used by the metrics."""
    if not isinstance(pred_logits, torch.Tensor):
        raise TypeError(f"Prediction logits must be a torch.Tensor, got {type(pred_logits).__name__}.")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Target mask must be a torch.Tensor, got {type(target).__name__}.")
    if pred_logits.ndim != 4:
        raise ValueError(f"Prediction logits must have shape (B, 1, H, W), got {tuple(pred_logits.shape)}.")
    if target.ndim != 4:
        raise ValueError(f"Target mask must have shape (B, 1, H, W), got {tuple(target.shape)}.")
    if pred_logits.shape != target.shape:
        raise ValueError(
            "Prediction logits and target mask must have the same shape; "
            f"got {tuple(pred_logits.shape)} and {tuple(target.shape)}."
        )
    if pred_logits.shape[1] != 1 or target.shape[1] != 1:
        raise ValueError(
            "Prediction logits and target mask must have shape (B, 1, H, W); "
            f"got {tuple(pred_logits.shape)} and {tuple(target.shape)}."
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in the range [0, 1], got {threshold}.")
    if not torch.isfinite(pred_logits).all():
        raise ValueError("Prediction logits must contain only finite values.")
    if not torch.isfinite(target).all():
        raise ValueError("Target mask must contain only finite values.")
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Target mask must be binary with values 0 and 1 only.")

    return pred_logits, target


@torch.no_grad()
def whole_tumor_metrics(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute Dice, IoU, Precision, and Recall for binary whole-tumor segmentation.

    Args:
        pred_logits: Raw model logits with shape (B, 1, H, W).
        target: Binary ground-truth mask with shape (B, 1, H, W), values 0 or 1.
        threshold: Threshold applied after sigmoid to convert logits to positives.

    Returns:
        A dictionary with float values for: ``dice``, ``iou``, ``precision``, and
        ``recall``.
    """
    pred_logits, target = _validate_inputs(pred_logits, target, threshold)

    pred_logits = pred_logits.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    pred_mask = torch.sigmoid(pred_logits) >= threshold
    target_mask = target.to(dtype=torch.bool)
    pred_mask = pred_mask.squeeze(1)
    target_mask = target_mask.squeeze(1)

    batch_size = pred_mask.shape[0]
    tp = torch.logical_and(pred_mask, target_mask).sum(dim=(1, 2), dtype=torch.float32)
    fp = torch.logical_and(pred_mask, torch.logical_not(target_mask)).sum(dim=(1, 2), dtype=torch.float32)
    fn = torch.logical_and(torch.logical_not(pred_mask), target_mask).sum(dim=(1, 2), dtype=torch.float32)

    dice = torch.zeros(batch_size, dtype=torch.float32, device=pred_logits.device)
    iou = torch.zeros(batch_size, dtype=torch.float32, device=pred_logits.device)
    precision = torch.zeros(batch_size, dtype=torch.float32, device=pred_logits.device)
    recall = torch.zeros(batch_size, dtype=torch.float32, device=pred_logits.device)

    has_target = target_mask.sum(dim=(1, 2)) > 0
    has_pred = pred_mask.sum(dim=(1, 2)) > 0

    # Both empty: all metrics are 1.
    both_empty = (~has_target) & (~has_pred)
    dice[both_empty] = 1.0
    iou[both_empty] = 1.0
    precision[both_empty] = 1.0
    recall[both_empty] = 1.0

    # Target is present but prediction is empty: all metrics are 0.
    target_only = has_target & (~has_pred)
    dice[target_only] = 0.0
    iou[target_only] = 0.0
    precision[target_only] = 0.0
    recall[target_only] = 0.0

    # Ground truth is empty but prediction contains false positives.
    pred_only = (~has_target) & has_pred
    dice[pred_only] = 0.0
    iou[pred_only] = 0.0
    precision[pred_only] = 0.0
    recall[pred_only] = 1.0

    # Normal case: both target and prediction have at least one positive pixel.
    normal = has_target & has_pred
    denominator_dice = 2.0 * tp[normal] + fp[normal] + fn[normal]
    dice[normal] = (2.0 * tp[normal]) / denominator_dice
    iou[normal] = tp[normal] / (tp[normal] + fp[normal] + fn[normal])
    precision[normal] = tp[normal] / (tp[normal] + fp[normal])
    recall[normal] = tp[normal] / (tp[normal] + fn[normal])

    metrics = {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }
    return metrics


def _summarize_probability_distribution(probabilities: torch.Tensor) -> dict[str, float]:
    """Return compact summary stats from probability values without retaining all samples."""
    flat = probabilities.reshape(-1).to(dtype=torch.float32)
    if flat.numel() == 0:
        return {
            "min_probability": float("nan"),
            "mean_probability": float("nan"),
            "max_probability": float("nan"),
            "probability_50th_percentile": float("nan"),
            "probability_90th_percentile": float("nan"),
            "probability_99th_percentile": float("nan"),
        }

    values = flat.cpu()
    total = values.numel()
    bins = 1024
    hist = torch.bincount(
        (values * bins).clamp_(0, bins - 1).to(torch.int64),
        minlength=bins,
    ).to(dtype=torch.float32)

    def percentile(quantile: float) -> float:
        target = max(1.0, quantile * total)
        running = 0.0
        for bin_index, count in enumerate(hist.tolist()):
            running += count
            if running >= target:
                return float((bin_index + 0.5) / bins)
        return 1.0

    return {
        "min_probability": float(values.min().item()),
        "mean_probability": float(values.mean().item()),
        "max_probability": float(values.max().item()),
        "probability_50th_percentile": percentile(0.50),
        "probability_90th_percentile": percentile(0.90),
        "probability_99th_percentile": percentile(0.99),
    }


@torch.no_grad()
def tumor_detection_diagnostics(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float | int | bool]:
    """Compute validation diagnostics for whole-tumor detection without storing full prediction arrays.

    This helper is intentionally lightweight: it only accumulates scalar counts and
    mean values for tumor-positive slices while using the same sigmoid/threshold and
    empty-mask convention as whole_tumor_metrics.

    The all-background baseline is defined using the project’s slice-averaged Dice
    convention, not a pixel-aggregated Dice. A separate pixel-aggregated value is
    reported under a different key to avoid mixing the two metrics.
    """
    pred_logits, target = _validate_inputs(pred_logits, target, threshold)
    pred_logits = pred_logits.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    probabilities = torch.sigmoid(pred_logits)
    pred_mask = probabilities >= threshold
    target_mask = target.to(dtype=torch.bool)
    pred_mask = pred_mask.squeeze(1)
    target_mask = target_mask.squeeze(1)
    probabilities = probabilities.squeeze(1)

    total_slices = int(pred_logits.shape[0])
    gt_positive_slices = int((target_mask.sum(dim=(1, 2)) > 0).sum().item())
    gt_negative_slices = total_slices - gt_positive_slices
    pred_positive_slices = int((pred_mask.sum(dim=(1, 2)) > 0).sum().item())
    pred_negative_slices = total_slices - pred_positive_slices
    total_predicted_tumor_pixels = int(pred_mask.sum().item())
    total_ground_truth_tumor_pixels = int(target_mask.sum().item())
    true_positive_pixels = int(torch.logical_and(pred_mask, target_mask).sum().item())
    false_positive_pixels = int(torch.logical_and(pred_mask, torch.logical_not(target_mask)).sum().item())
    false_negative_pixels = int(torch.logical_and(torch.logical_not(pred_mask), target_mask).sum().item())
    false_positive_slices = int(
        ((target_mask.sum(dim=(1, 2)) == 0) & (pred_mask.sum(dim=(1, 2)) > 0)).sum().item()
    )
    empty_gt_empty_pred = int(
        ((target_mask.sum(dim=(1, 2)) == 0) & (pred_mask.sum(dim=(1, 2)) == 0)).sum().item()
    )

    positive_idx = torch.nonzero(target_mask.sum(dim=(1, 2)) > 0, as_tuple=False).flatten()
    if positive_idx.numel() == 0:
        mean_positive_dice = 0.0
        mean_positive_iou = 0.0
        mean_positive_recall = 0.0
    else:
        positive_metrics = []
        for sample_index in positive_idx.tolist():
            sample_logits = pred_logits[sample_index : sample_index + 1]
            sample_target = target[sample_index : sample_index + 1]
            positive_metrics.append(whole_tumor_metrics(sample_logits, sample_target, threshold=threshold))
        mean_positive_dice = float(sum(item["dice"] for item in positive_metrics) / len(positive_metrics))
        mean_positive_iou = float(sum(item["iou"] for item in positive_metrics) / len(positive_metrics))
        mean_positive_recall = float(sum(item["recall"] for item in positive_metrics) / len(positive_metrics))

    if total_slices == 0:
        all_background_baseline_dice = 0.0
    else:
        all_background_baseline_dice = float((target_mask.sum(dim=(1, 2)) == 0).float().mean().item())

    if total_ground_truth_tumor_pixels == 0:
        pixel_aggregated_all_background_dice = 1.0
    else:
        pixel_aggregated_all_background_dice = 0.0

    tumor_probabilities = probabilities[target_mask]
    outside_probabilities = probabilities[~target_mask]
    threshold_counts = {
        f"tumor_pixels_predicted_at_threshold_{threshold_value}": int((tumor_probabilities >= threshold_value).sum().item())
        for threshold_value in (0.1, 0.3, 0.5)
    }
    threshold_counts.update({
        f"background_pixels_predicted_at_threshold_{threshold_value}": int((outside_probabilities >= threshold_value).sum().item())
        for threshold_value in (0.1, 0.3, 0.5)
    })

    zero_predicted_tumor_pixels = total_predicted_tumor_pixels == 0

    return {
        "total_validation_slices": total_slices,
        "ground_truth_tumor_positive_slices": gt_positive_slices,
        "ground_truth_tumor_negative_slices": gt_negative_slices,
        "predicted_tumor_positive_slices": pred_positive_slices,
        "predicted_tumor_negative_slices": pred_negative_slices,
        "total_predicted_tumor_pixels": total_predicted_tumor_pixels,
        "total_ground_truth_tumor_pixels": total_ground_truth_tumor_pixels,
        "true_positive_pixels": true_positive_pixels,
        "false_positive_pixels": false_positive_pixels,
        "false_negative_pixels": false_negative_pixels,
        "mean_dice_on_tumor_positive_ground_truth_slices": mean_positive_dice,
        "mean_iou_on_tumor_positive_ground_truth_slices": mean_positive_iou,
        "mean_recall_on_tumor_positive_ground_truth_slices": mean_positive_recall,
        "false_positive_slices_among_ground_truth_negative_slices": false_positive_slices,
        "empty_ground_truth_and_empty_prediction_pairs": empty_gt_empty_pred,
        "all_background_baseline_dice": all_background_baseline_dice,
        "all_background_pixel_aggregated_dice": float(pixel_aggregated_all_background_dice),
        "zero_predicted_tumor_pixels": zero_predicted_tumor_pixels,
        "all_probability_summary": _summarize_probability_distribution(probabilities),
        "tumor_probability_summary": _summarize_probability_distribution(tumor_probabilities),
        "background_probability_summary": _summarize_probability_distribution(outside_probabilities),
        **threshold_counts,
    }


def main() -> None:
    """Demonstrate the metric behavior on small synthetic examples."""
    examples = [
        (
            "Perfect overlap",
            torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32),
        ),
        (
            "Partial overlap",
            torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32),
        ),
        (
            "No overlap",
            torch.tensor([[[[-10.0, -10.0], [-10.0, -10.0]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32),
        ),
        (
            "Both masks empty",
            torch.tensor([[[[-10.0, -10.0], [-10.0, -10.0]]]], dtype=torch.float32),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32),
        ),
    ]

    for name, logits, target in examples:
        result = whole_tumor_metrics(logits, target)
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()
