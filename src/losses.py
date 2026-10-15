"""Differentiable Focal Tversky Loss for binary whole-tumor segmentation."""

import math
from numbers import Real

import torch
from torch import nn


class FocalTverskyLoss(nn.Module):
    """Consume raw logits and binary targets of matching shape (B, 1, H, W).

    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = mean_over_images((1 - TI) ** gamma)

    alpha weights false positives; beta weights false negatives. Their defaults
    are 0.7 and 0.3. Each must lie in [0, 1], and at least one must be positive;
    they need not sum to 1. gamma is a positive exponent, defaulting to 0.75.
    For errors in (0, 1), gamma < 1 raises the loss relative to a linear error;
    gamma > 1 lowers it. The exponent is gamma itself, never 1/gamma.

    Positive smoothing (default 1.0) is an implementation choice for stability,
    especially with empty tumor masks. No prediction thresholding is used.
    Float16/bfloat16 inputs are computed in float32 to avoid reduction overflow;
    float64 inputs retain float64 precision. The returned loss is a scalar.
    Smoothing must fit the calculation dtype's normal positive range, and gamma
    must fit its finite range, so conversion cannot silently become zero/infinity.
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma), ("smooth", smooth)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError("alpha and beta must each be in [0, 1].")
        if alpha + beta == 0:
            raise ValueError("At least one of alpha and beta must be positive.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("logits and targets must both be PyTorch tensors.")
        if logits.ndim != 4 or targets.ndim != 4:
            raise ValueError("Expected logits and targets with shape (B, 1, H, W).")
        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have identical shapes; got {tuple(logits.shape)} and {tuple(targets.shape)}.")
        if logits.shape[1] != 1 or any(size == 0 for size in logits.shape):
            raise ValueError("Expected nonempty tensors with shape (B, 1, H, W).")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating-point tensors containing raw model scores.")
        if targets.is_complex():
            raise TypeError("targets must contain real binary values 0 and 1.")
        if logits.device != targets.device:
            raise ValueError("logits and targets must be on the same device.")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must contain only finite values; NaNs and infinities are invalid.")
        if not torch.isfinite(targets).all():
            raise ValueError("targets must contain only finite values; NaNs and infinities are invalid.")
        # Validate BEFORE dtype conversion, so rounding cannot hide bad labels.
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("targets must contain only binary values 0 and 1.")

        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        precision = torch.finfo(dtype)
        if not precision.tiny <= self.smooth <= precision.max:
            raise ValueError(f"smooth must fit the normal positive range of calculation dtype {dtype}.")
        if self.gamma > precision.max:
            raise ValueError(f"gamma must fit the finite range of calculation dtype {dtype}.")
        # Sigmoid is applied exactly once, inside this loss. Casting preserves
        # gradients; bool/integer masks are also safely converted for arithmetic.
        probabilities = torch.sigmoid(logits.to(dtype=dtype))
        targets = targets.to(dtype=dtype)
        dimensions = (1, 2, 3)  # Channel, height, width; never the batch axis.
        # TP: predicted tumor probability at real tumor pixels.
        true_positives = (probabilities * targets).sum(dim=dimensions)
        # FP: predicted tumor probability at background pixels.
        false_positives = (probabilities * (1 - targets)).sum(dim=dimensions)
        # FN: missing tumor probability at real tumor pixels.
        false_negatives = ((1 - probabilities) * targets).sum(dim=dimensions)
        tversky = (true_positives + self.smooth) / (
            true_positives
            + self.alpha * false_positives
            + self.beta * false_negatives
            + self.smooth
        )
        error = (1 - tversky).clamp(min=0, max=1)

        # Fractional powers have an infinite derivative at zero. For numerically
        # perfect predictions, keep the exact zero loss and choose a finite zero
        # gradient. Replace zero bases BEFORE pow so backward never computes 0
        # to a negative power. Positive errors still use the exact requested rule.
        positive_error = error > 0
        safe_base = torch.where(positive_error, error, torch.ones_like(error))
        per_image_loss = torch.where(positive_error, safe_base.pow(self.gamma), torch.zeros_like(error))
        return per_image_loss.mean()


class HybridFocalTverskyBceLoss(nn.Module):
    """Optional hybrid objective: 0.5 * FocalTversky + 0.5 * BCEWithLogits.

    This is a diagnostic/experimental objective for background-collapse prevention.
    It preserves the original Focal Tversky defaults and uses raw logits for the
    BCE term, not sigmoid probabilities.
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0, pos_weight=5.0):
        super().__init__()
        for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma), ("smooth", smooth), ("pos_weight", pos_weight)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError("alpha and beta must each be in [0, 1].")
        if alpha + beta == 0:
            raise ValueError("At least one of alpha and beta must be positive.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive.")
        self.focal = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.pos_weight = float(pos_weight)

    def forward(self, logits, targets):
        if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("logits and targets must both be PyTorch tensors.")
        if logits.ndim != 4 or targets.ndim != 4:
            raise ValueError("Expected logits and targets with shape (B, 1, H, W).")
        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have identical shapes; got {tuple(logits.shape)} and {tuple(targets.shape)}.")
        if logits.shape[1] != 1 or any(size == 0 for size in logits.shape):
            raise ValueError("Expected nonempty tensors with shape (B, 1, H, W).")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating-point tensors containing raw model scores.")
        if targets.is_complex():
            raise TypeError("targets must contain real binary values 0 and 1.")
        if logits.device != targets.device:
            raise ValueError("logits and targets must be on the same device.")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must contain only finite values; NaNs and infinities are invalid.")
        if not torch.isfinite(targets).all():
            raise ValueError("targets must contain only finite values; NaNs and infinities are invalid.")
        if not ((targets == 0) | (targets == 1)).all():
            raise ValueError("targets must contain only binary values 0 and 1.")

        working_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        working_logits = logits.to(dtype=working_dtype)
        working_targets = targets.to(device=working_logits.device, dtype=working_dtype)
        focal = self.focal(working_logits, working_targets)
        pos_weight = torch.tensor(self.pos_weight, device=working_logits.device, dtype=working_dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            working_logits,
            working_targets,
            pos_weight=pos_weight,
            reduction="mean",
        )
        return 0.5 * focal + 0.5 * bce


def build_loss_criterion(loss_type: str, *, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0, pos_weight=5.0):
    """Build the requested training criterion while keeping the default Focal Tversky path untouched."""
    normalized = (loss_type or "focal_tversky").strip().lower()
    if normalized == "focal_tversky":
        return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
    if normalized == "hybrid":
        return HybridFocalTverskyBceLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth, pos_weight=pos_weight)
    raise ValueError("loss_type must be one of {'focal_tversky', 'hybrid'}.")


def main():
    # Two tiny CPU examples: one mixed tumor mask and one empty tumor mask.
    logits = torch.tensor(
        [[[[2.0, -2.0], [0.5, -0.5]]], [[[-2.0, -2.0], [-2.0, -2.0]]]],
        dtype=torch.float32, requires_grad=True,
    )
    targets = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    criterion = FocalTverskyLoss()
    loss = criterion(logits, targets)
    print(f"alpha={criterion.alpha}, beta={criterion.beta}, gamma={criterion.gamma}, smooth={criterion.smooth}")
    print(f"Logits and target shapes: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}, shape: {tuple(loss.shape)}, dtype: {loss.dtype}")
    # This checks gradients of synthetic logits only; there is no model training.
    loss.backward()
    print(f"Gradient shape: {tuple(logits.grad.shape)}")
    print(f"All gradients finite: {torch.isfinite(logits.grad).all().item()}")


if __name__ == "__main__":
    main()
