"""Load preprocessed FLAIR slices and binary masks one sample at a time."""

from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BrainTumorDataset2D(Dataset):
    """Return (image, mask) tensors from a directory of smoke-test NPZ files.

    Images have three identical FLAIR channels; masks have one channel.
    Both tensors use float32 and stay on the CPU for DataLoader batching.
    """

    def __init__(self, sample_dir: str | Path | None = None):
        if sample_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            sample_dir = project_root / "data/processed/smoke_test"
        self.sample_dir = Path(sample_dir)
        if not self.sample_dir.is_dir():
            raise ValueError(f"Sample directory does not exist: {self.sample_dir}")

        # Store only filenames here. Sorting gives a reproducible sample order;
        # the actual arrays are not read until __getitem__ is called.
        self.sample_paths = sorted(
            path for path in self.sample_dir.glob("*.npz") if path.is_file()
        )
        if not self.sample_paths:
            raise ValueError(f"No .npz samples found in: {self.sample_dir}")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_path = self.sample_paths[index]
        try:
            archive = np.load(sample_path, allow_pickle=False)
            if not hasattr(archive, "files"):
                raise ValueError("Expected an NPZ archive containing image and mask arrays.")
            # The context manager closes the archive after this one pair is read.
            with archive as sample:
                missing = {"image", "mask"} - set(sample.files)
                if missing:
                    raise ValueError(f"Missing arrays: {sorted(missing)}")
                image = sample["image"]
                mask = sample["mask"]
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            raise ValueError(f"Could not load sample {sample_path}: {error}") from error

        # Validate before converting dtypes so rounding cannot hide bad values.
        if image.shape != (256, 256):
            raise ValueError(f"{sample_path}: image shape {image.shape}; expected (256, 256).")
        if mask.shape != (256, 256):
            raise ValueError(f"{sample_path}: mask shape {mask.shape}; expected (256, 256).")
        # NumPy's dtype kinds b/u/i/f represent bool and real numeric types.
        if image.dtype.kind not in "buif" or not np.isfinite(image).all():
            raise ValueError(f"{sample_path}: image must contain finite real numeric values.")
        if np.any(image < 0) or np.any(image > 1):
            raise ValueError(f"{sample_path}: image values must be within [0, 1].")
        if mask.dtype.kind not in "buif" or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{sample_path}: mask values must be exactly 0 or 1.")

        # Add a channel dimension: (256, 256) -> (1, 256, 256).
        # Repeat grayscale FLAIR three times for the planned encoder input.
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_tensor = image_tensor.repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


def main():
    dataset = BrainTumorDataset2D()
    # A small batch and no worker processes keep this demonstration lightweight
    # and straightforward to run on macOS with limited RAM.
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    images, masks = next(iter(loader))
    print(f"Number of samples: {len(dataset)}")
    print(f"First batch image shape: {tuple(images.shape)}")
    print(f"First batch mask shape: {tuple(masks.shape)}")
    print(f"Image tensor dtype: {images.dtype}")
    print(f"Mask tensor dtype: {masks.dtype}")
    print(f"Unique mask values: {torch.unique(masks).tolist()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f"Error: {error}")
