"""Read-only audit of the project Focal Tversky loss on synthetic CPU tensors.

This script intentionally does not modify model weights or train the network.
It exercises the exact project loss implementation with raw logits and prints
loss/gradient diagnostics for empty, small-tumor, and large-tumor masks.
"""

from __future__ import annotations

import torch

from src.losses import FocalTverskyLoss

ALPHA = 0.7
BETA = 0.3
GAMMA = 0.75
SMOOTH = 1.0


def build_masks():
    empty = torch.zeros(1, 1, 6, 6, dtype=torch.float32)
    small_tumor = torch.zeros_like(empty)
    small_tumor[:, :, 2:4, 2:4] = 1.0
    large_tumor = torch.zeros_like(empty)
    large_tumor[:, :, 1:5, 1:5] = 1.0
    return {"empty": empty, "small": small_tumor, "large": large_tumor}


def gradient_summary(grad: torch.Tensor, target: torch.Tensor):
    tumor_grad = grad[target == 1]
    background_grad = grad[target == 0]

    def summarize(values: torch.Tensor):
        if values.numel() == 0:
            return {"count": 0, "min": None, "max": None, "mean_abs": None, "finite": True}
        return {
            "count": int(values.numel()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean_abs": float(values.abs().mean().item()),
            "finite": bool(torch.isfinite(values).all().item()),
        }

    return {
        "tumor": summarize(tumor_grad),
        "background": summarize(background_grad),
        "all": summarize(grad),
    }


def fmt_value(value):
    return "n/a" if value is None else f"{value:.6e}"


def evaluate_case(mask_name: str, target: torch.Tensor, logit_value: float):
    criterion = FocalTverskyLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=SMOOTH)
    logits = torch.full_like(target, logit_value, dtype=torch.float32, requires_grad=True)
    probabilities = torch.sigmoid(logits)
    probability_stats = {
        "min": float(probabilities.min().item()),
        "mean": float(probabilities.mean().item()),
        "max": float(probabilities.max().item()),
    }
    loss = criterion(logits, target)
    loss.backward()
    grad = logits.grad
    stats = gradient_summary(grad, target)

    step_lr = 0.1
    logits_step = logits.detach().clone().requires_grad_(True)
    step_loss = criterion(logits_step, target)
    step_loss.backward()
    updated_logits = logits_step - step_lr * logits_step.grad
    updated_loss = criterion(updated_logits, target)
    step_reduced = bool(updated_loss.item() < step_loss.item())

    return {
        "mask": mask_name,
        "logit": float(logit_value),
        "sigmoid_probability": probability_stats,
        "loss": float(loss.item()),
        "step_reduced": step_reduced,
        "updated_loss": float(updated_loss.item()),
        "grad": stats,
    }


def pass_fail(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def summarize_batch_behavior():
    empty = torch.zeros(1, 1, 6, 6, dtype=torch.float32)
    positive = torch.zeros_like(empty)
    positive[:, :, 2:5, 2:5] = 1.0
    criterion = FocalTverskyLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=SMOOTH)
    bg_logits = torch.full_like(empty, -10.0, dtype=torch.float32)
    positive_loss = criterion(bg_logits, positive)
    empty_loss = criterion(bg_logits, empty)
    print("\nBatch-size-1 / per-image averaging audit:")
    print(f"  all-background logits on empty mask: loss={empty_loss.item():.6f}")
    print(f"  all-background logits on positive mask: loss={positive_loss.item():.6f}")
    print("  Interpretation: with batch size 1, each slice loss is averaged exactly once.")
    print("  Empty slices can dominate the mean if validation contains many empty masks, even when the model predicts no tumor.")


def main():
    print("Focal Tversky audit for the project implementation")
    print(f"Hyperparameters: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, smooth={SMOOTH}")
    print("Inputs are raw logits. Sigmoid is applied inside the loss implementation.")
    print("\nSynthetic mask experiments (constant logits per image):")

    masks = build_masks()
    results = []
    for mask_name, target in masks.items():
        for logit_value in (-10.0, -5.0, 0.0, 5.0):
            result = evaluate_case(mask_name, target, logit_value)
            results.append(result)
            print(
                f"\nMask={mask_name:5s}  logit={logit_value:>4.1f}  "
                f"sigmoid min={result['sigmoid_probability']['min']:.6f} "
                f"mean={result['sigmoid_probability']['mean']:.6f} "
                f"max={result['sigmoid_probability']['max']:.6f}  "
                f"loss={result['loss']:.6f}"
            )
            tumor = result["grad"]["tumor"]
            background = result["grad"]["background"]
            print(
                "  tumor grad: count={count:>3d} min={min} max={max} mean_abs={mean_abs} finite={finite}".format(
                    count=tumor["count"],
                    min=fmt_value(tumor["min"]),
                    max=fmt_value(tumor["max"]),
                    mean_abs=fmt_value(tumor["mean_abs"]),
                    finite=tumor["finite"],
                )
            )
            print(
                "  bg grad:    count={count:>3d} min={min} max={max} mean_abs={mean_abs} finite={finite}".format(
                    count=background["count"],
                    min=fmt_value(background["min"]),
                    max=fmt_value(background["max"]),
                    mean_abs=fmt_value(background["mean_abs"]),
                    finite=background["finite"],
                )
            )
            print(f"  step reduced loss: {result['step_reduced']} -> updated_loss={result['updated_loss']:.6f}")

    summarize_batch_behavior()

    positive_mask = masks["small"]
    empty_mask = masks["empty"]
    criterion = FocalTverskyLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=SMOOTH)
    empty_all_background = criterion(torch.full_like(empty_mask, -10.0, dtype=torch.float32, requires_grad=False), empty_mask)
    positive_all_background = criterion(torch.full_like(positive_mask, -10.0, dtype=torch.float32, requires_grad=False), positive_mask)

    print("\nAll-background prediction comparison:")
    print(f"  empty mask loss:          {empty_all_background.item():.6f}")
    print(f"  tumor-positive mask loss: {positive_all_background.item():.6f}")

    checks = []
    checks.append(
        pass_fail(
            "Loss is finite for all synthetic experiments.",
            all(torch.isfinite(torch.tensor(r["loss"], dtype=torch.float64)) for r in results),
        )
    )
    checks.append(
        pass_fail(
            "Finite gradient values across all synthetic experiments.",
            all(
                all(
                    stats["finite"]
                    for stats in (r["grad"]["tumor"], r["grad"]["background"], r["grad"]["all"])
                )
                for r in results
            ),
        )
    )
    checks.append(
        pass_fail(
            "All background predictions are lower loss on an empty mask than a tumor-positive mask.",
            empty_all_background.item() < positive_all_background.item(),
        )
    )
    checks.append(
        pass_fail(
            "Tumor-positive gradients are negative and background gradients are positive for zero-logit start on a tumor-positive mask.",
            any(
                (
                    (evaluate_case("small", masks["small"], 0.0)["grad"]["tumor"]["min"] is not None)
                    and (evaluate_case("small", masks["small"], 0.0)["grad"]["tumor"]["min"] < 0.0)
                    and (evaluate_case("small", masks["small"], 0.0)["grad"]["background"]["max"] > 0.0)
                )
                for _ in [0]
            ),
        )
    )
    checks.append(
        pass_fail(
            "A direct one-step gradient update can reduce a tumor-positive loss at zero logits.",
            any(r["step_reduced"] for r in results if r["mask"] == "small" and r["logit"] == 0.0),
        )
    )

    print("\nAudit summary:")
    if all(checks):
        print("PASS: the project Focal Tversky implementation is mathematically consistent for these synthetic CPU checks.")
    else:
        print("FAIL: at least one mathematical consistency check failed. Review the exact loss/gradient behavior before changing the model or training setup.")

    print("\nObserved findings vs. possible limitations:")
    print("- The code path in src/losses.py applies sigmoid inside the loss and uses a per-image mean across the batch axis.")
    print("- For large-magnitude logits, sigmoid saturation can make gradients very small even when the loss is non-zero.")
    print("- This does not necessarily indicate a formula bug; it is a known optimization limitation of saturated logits and tiny learning rates.")
    print("- Empty validation slices can still dominate an average if the subset contains many empty masks, because each slice contributes equally to the mean by design.")
    print("- The training stall in the VGG19 U-Net is consistent with a model that becomes background-dominant under frozen features and tiny updates, not necessarily with an incorrect Tversky formula.")
    print("\nThis script is read-only and intentionally leaves src/losses.py unchanged.")


if __name__ == "__main__":
    main()
