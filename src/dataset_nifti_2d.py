"""Read indexed BraTS axial slices lazily, with per-slice normalization."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")


def normalize_flair_slice(image):
    """Clip nonzero brain pixels at their 1st/99th percentiles and scale to [0, 1].

    This uses PER-SLICE statistics. The earlier preprocessing smoke test used
    per-volume statistics; the percentile choices and background handling match.
    Empty or constant slices safely become zeros.
    """
    image = np.array(image, dtype=np.float32, copy=True)
    if not np.isfinite(image).all():
        raise ValueError("FLAIR slice must contain finite float32 intensities.")
    background = image == 0
    brain_pixels = image[~background]
    if brain_pixels.size == 0:
        return image
    low, high = np.percentile(brain_pixels, [1, 99])
    if high <= low:
        image.fill(0)
        return image
    np.clip(image, low, high, out=image)
    image -= float(low)
    image /= float(high - low)
    # Float32 rounding can slightly exceed the endpoints; clamp those values.
    np.clip(image, 0, 1, out=image)
    image[background] = 0
    return image


class BraTSNiftiSliceDataset(Dataset):
    """Return (image, mask) tensors for exactly one split of a slice-index CSV.

    Initialization stores CSV metadata only. Each access opens two NIfTI headers
    and reads their matching 2D slices, without caching any full volume. Returned
    tensors stay on the CPU: image (3, 256, 256), mask (1, 256, 256), both float32.
    """

    def __init__(self, index_path: str | Path, dataset_root: str | Path, split: str):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
        self.index_path = Path(index_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        if not self.dataset_root.is_dir():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")

        # No nib.load() calls or MRI arrays here: only small CSV metadata.
        with self.index_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.columns = reader.fieldnames or []
            required = {"patient_id", "split", "flair_path", "seg_path", "slice_index"}
            missing = required - set(self.columns)
            if missing:
                raise ValueError(f"Slice index is missing columns: {sorted(missing)}")
            rows = list(reader)

        self.split_counts = Counter(row["split"] for row in rows)
        unexpected = set(self.split_counts) - set(SPLITS)
        if unexpected:
            raise ValueError(f"Slice index contains unexpected split labels: {list(unexpected)}")
        self.records = []
        assignments = {}
        patient_paths = {}
        seen = set()
        for line_number, row in enumerate(rows, start=2):
            if any(not row.get(column) for column in required):
                raise ValueError(f"Slice index line {line_number}: empty required value.")
            patient_id = row["patient_id"]
            if patient_id in assignments and assignments[patient_id] != row["split"]:
                raise ValueError(f"Patient {patient_id} appears in multiple splits.")
            assignments[patient_id] = row["split"]
            paths = (row["flair_path"], row["seg_path"])
            if patient_id in patient_paths and patient_paths[patient_id] != paths:
                raise ValueError(f"Patient {patient_id} has inconsistent MRI/mask paths.")
            patient_paths[patient_id] = paths
            for relative_path in paths:
                path = Path(relative_path)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Expected a dataset-relative path: {relative_path!r}")
            try:
                slice_index = int(row["slice_index"])
            except ValueError as error:
                raise ValueError(f"Line {line_number}: slice_index must be an integer.") from error
            if slice_index < 0:
                raise ValueError(f"Line {line_number}: slice_index must be nonnegative.")
            row["slice_index"] = slice_index
            key = (patient_id, slice_index)
            if key in seen:
                raise ValueError(f"Duplicate patient/slice record: {key}")
            seen.add(key)
            if row["split"] == split:
                self.records.append(row)
        if not self.records:
            raise ValueError(f"No slice records found for split {split!r}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index {index} out of range for {len(self)} samples.")
        record = self.records[index]
        slice_index = record["slice_index"]
        description = f"{record['patient_id']}, axial slice {slice_index}"
        flair_path = self.dataset_root / record["flair_path"]
        seg_path = self.dataset_root / record["seg_path"]
        for path in (flair_path, seg_path):
            if not path.is_file():
                raise FileNotFoundError(f"{description}: NIfTI file not found: {path}")
        try:
            # Read-only memory mapping where supported. Slice reads close their
            # file handles; these local header/proxy objects are never cached.
            flair = nib.load(flair_path, mmap="r", keep_file_open=False)
            segmentation = nib.load(seg_path, mmap="r", keep_file_open=False)
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            raise ValueError(f"{description}: cannot load NIfTI files: {error}") from error
        if flair.shape != segmentation.shape:
            raise ValueError(f"{description}: volume shapes differ: {flair.shape} vs {segmentation.shape}.")
        if len(flair.shape) != 3 or any(size == 0 for size in flair.shape):
            raise ValueError(f"{description}: expected nonempty three-dimensional volumes.")
        if slice_index >= flair.shape[2]:
            raise IndexError(f"{description}: slice_index out of range for volume shape {flair.shape}.")
        if not np.allclose(flair.affine, segmentation.affine, rtol=0, atol=1e-5):
            raise ValueError(f"{description}: FLAIR and mask spatial coordinates differ.")

        # Convert only the selected 2D arrays, never a complete 3D data proxy.
        try:
            image = np.asanyarray(flair.dataobj[:, :, slice_index])
            mask = np.asanyarray(segmentation.dataobj[:, :, slice_index])
        except (OSError, ValueError) as error:
            raise ValueError(f"{description}: cannot read axial slices: {error}") from error
        if image.shape != mask.shape:
            raise ValueError(f"{description}: MRI and mask slice dimensions differ.")
        if np.iscomplexobj(image) or not np.isfinite(image).all():
            raise ValueError(f"{description}: FLAIR must contain finite real values.")
        if not np.isfinite(mask).all():
            raise ValueError(f"{description}: segmentation must contain finite values.")
        labels = np.unique(mask)
        if not np.isin(labels, [0, 1, 2, 4]).all():
            raise ValueError(f"{description}: unexpected segmentation labels {labels.tolist()}; expected 0, 1, 2, 4.")

        image = normalize_flair_slice(image)
        mask = (mask > 0).astype(np.uint8)  # Labels 1, 2 AND 4 are whole tumor.
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        np.clip(image, 0, 1, out=image)
        if not np.isfinite(image).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{description}: invalid resized image or nonbinary mask.")

        # Add channels without changing orientation. No augmentation or ImageNet
        # mean/std normalization is applied at this stage.
        image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=project_root / "data/processed/slice_index_smoke.csv")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    dataset = BraTSNiftiSliceDataset(args.index, project_root / "data/raw/brats2020", args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    images, masks = next(iter(loader))
    print(f"CSV columns: {dataset.columns}")
    print(f"CSV split labels and slice counts: {dict(dataset.split_counts)}")
    print(f"Selected split: {dataset.split}")
    print(f"Dataset length: {len(dataset)}")
    print(f"First batch image shape: {tuple(images.shape)}, dtype: {images.dtype}")
    print(f"First batch mask shape: {tuple(masks.shape)}, dtype: {masks.dtype}")
    print(f"Image pixel range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"Mask pixel range: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
    print(f"Unique mask values: {torch.unique(masks).tolist()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError) as error:
        sys.exit(f"Error: {error}")
