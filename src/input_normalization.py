"""ImageNet normalization helpers for BraTS FLAIR images.

This module intentionally keeps the original MRI dataset behavior unchanged and
adds an opt-in wrapper that applies the standard VGG19 ImageNet normalization to
already-normalized MRI intensities in the range [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from src.dataset_nifti_2d import BraTSNiftiSliceDataset
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback.
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dataset_nifti_2d import BraTSNiftiSliceDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet_image(image: torch.Tensor) -> torch.Tensor:
    """Apply standard ImageNet channel-wise normalization to a 3-channel image.

    Args:
        image: Tensor with shape (3, H, W) and float32 dtype. Values must already
            be in the range [0, 1], since the MRI dataset normalizes each slice to
            that range before this step.

    Returns:
        A new float32 tensor with the same spatial dimensions and the same device.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image tensor shape (3, H, W), got {tuple(image.shape)}.")
    if image.dtype != torch.float32:
        raise TypeError(f"Expected float32 image tensor, got {image.dtype}.")
    if not torch.isfinite(image).all():
        raise ValueError("Image tensor contains non-finite values.")
    if torch.any(image < 0.0) or torch.any(image > 1.0):
        raise ValueError("Image values must be in [0, 1] before ImageNet normalization. Do not normalize twice.")

    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


class ImageNetNormalizedDataset(Dataset):
    """Wrap an existing dataset and apply ImageNet normalization to images only.

    The wrapped dataset keeps every mask and extra metadata intact. This is an
    opt-in layer and does not change the original dataset or default training data.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]

        if isinstance(sample, dict):
            sample_copy = dict(sample)
            if "image" not in sample_copy:
                raise KeyError("Dataset item dictionary must contain an 'image' field.")
            sample_copy["image"] = normalize_imagenet_image(sample_copy["image"])
            return sample_copy

        if isinstance(sample, (tuple, list)):
            if len(sample) < 2:
                raise ValueError("Expected a dataset item with at least (image, mask).")
            image, *rest = sample
            normalized_image = normalize_imagenet_image(image)
            return (normalized_image, *rest)

        raise TypeError(
            "Unsupported dataset item type. Expected a tuple/list or dict with an 'image' field."
        )


def main() -> None:
    """Demonstrate the normalization on the smoke-test split without training."""
    project_root = Path(__file__).resolve().parents[1]
    index_path = project_root / "data/processed/slice_index_smoke.csv"
    dataset = BraTSNiftiSliceDataset(index_path, project_root / "data/raw/brats2020", split="train")
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    original_images, original_masks = next(iter(loader))

    normalized_dataset = ImageNetNormalizedDataset(dataset)
    normalized_loader = DataLoader(normalized_dataset, batch_size=1, num_workers=0, shuffle=False)
    normalized_images, normalized_masks = next(iter(normalized_loader))

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    expected = (original_images - mean) / std

    print(f"Original image shape: {tuple(original_images.shape)}")
    print(f"Original image pixel range: [{original_images.min().item():.4f}, {original_images.max().item():.4f}]")
    print(f"Normalized image shape: {tuple(normalized_images.shape)}")
    print(f"Normalized image pixel range: [{normalized_images.min().item():.4f}, {normalized_images.max().item():.4f}]")
    print(f"Mask shape: {tuple(original_masks.shape)}")
    print(f"Mask unique values: {torch.unique(original_masks).tolist()}")
    print(f"All normalized image values are finite: {torch.isfinite(normalized_images).all().item()}")
    print(f"Mask unchanged: {bool(torch.equal(original_masks, normalized_masks))}")
    print(f"Normalized channels match the expected formula: {bool(torch.allclose(normalized_images, expected))}")


if __name__ == "__main__":
    main()
