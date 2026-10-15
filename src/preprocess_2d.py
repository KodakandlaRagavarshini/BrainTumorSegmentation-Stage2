"""Preprocess up to eight axial slices from the first training patient."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import cv2
import matplotlib

# Set the backend before importing pyplot so no GUI window is required.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def read_training_patient(manifest_path):
    """Inspect the manifest and keep the first training row in CSV order."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        print(f"CSV columns: {columns}")
        required = {"patient_id", "split", "flair_path", "seg_path"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        first_training_patient = None
        split_counts = Counter()
        for row in reader:
            split_counts[row["split"]] += 1
            # 'train' is the training label in the existing project manifest.
            if row["split"] == "train" and first_training_patient is None:
                first_training_patient = row
    print(f"CSV split labels and counts: {dict(split_counts)}")
    if first_training_patient is None:
        raise ValueError("No patient with split='train' was found in the CSV.")
    return first_training_patient


def dataset_path(dataset_root, relative_path):
    """Join a manifest path to the dataset link without altering that link."""
    path = Path(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a dataset-relative file path: {relative_path!r}")
    full_path = dataset_root / path
    if not full_path.is_file():
        raise ValueError(f"Required NIfTI file not found: {full_path}")
    return full_path


def normalization_bounds(flair):
    """Find one pair of robust intensity bounds for all this patient's slices."""
    if not np.isfinite(flair).all():
        raise ValueError("FLAIR contains NaN or infinite intensities.")
    # Ignore the large zero background when calculating the brain percentiles.
    brain_voxels = flair[flair != 0]
    if brain_voxels.size == 0:
        print("Normalization: empty brain; images will safely contain zeros.")
        return None
    low, high = np.percentile(brain_voxels, [1, 99])
    if high <= low:
        print("Normalization: constant percentile range; images will contain zeros.")
        return None
    print(f"Nonzero brain intensity percentiles (1%, 99%): {low:.4f}, {high:.4f}")
    return float(low), float(high)


def normalize_slice(flair_slice, bounds):
    """Clip and scale a copy of one slice, leaving zero background at zero."""
    image = np.array(flair_slice, dtype=np.float32, copy=True)
    if bounds is None:
        image.fill(0)
        return image
    background = image == 0
    low, high = bounds
    np.clip(image, low, high, out=image)
    image -= low
    image /= high - low
    # Float32 rounding can slightly exceed 1 when percentile bounds are fractional.
    np.clip(image, 0, 1, out=image)
    image[background] = 0
    return image


def evenly_spaced(indices, limit):
    """Choose up to limit indices, including both ends when possible."""
    count = min(len(indices), limit)
    if count == 0:
        return []
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    return [int(indices[position]) for position in positions]


def save_visualization(image, mask, patient_id, slice_index, path, overwrite):
    """Save an MRI, binary mask, and transparent red tumor overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    try:
        # Transpose both arrays consistently for display only. Saved arrays keep
        # the same axis order as the source slices; no anatomical reorientation.
        display_image = image.T
        display_mask = mask.T
        axes[0].imshow(display_image, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[0].set_title("Normalized FLAIR")
        axes[1].imshow(display_mask, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[1].set_title("Binary whole-tumor mask")
        axes[2].imshow(display_image, cmap="gray", vmin=0, vmax=1, origin="lower")
        overlay = np.zeros((*display_mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1
        overlay[..., 3] = display_mask * 0.4
        axes[2].imshow(overlay, origin="lower")
        axes[2].set_title("Red tumor overlay")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{patient_id} — axial slice {slice_index}")
        fig.tight_layout()
        # Exclusive creation also prevents accidental overwrite after the check.
        with path.open("wb" if overwrite else "xb") as handle:
            fig.savefig(handle, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly allow replacing this smoke test's existing NPZ and PNG outputs.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data/raw/brats2020"
    manifest_path = project_root / "data/processed/patient_splits.csv"
    sample_dir = project_root / "data/processed/smoke_test"
    plot_path = project_root / "outputs/plots/preprocessing_smoke_test.png"

    patient = read_training_patient(manifest_path)
    patient_id = patient["patient_id"]
    print(f"Selected training patient: {patient_id}")
    flair_path = dataset_path(dataset_root, patient["flair_path"])
    seg_path = dataset_path(dataset_root, patient["seg_path"])

    # Load only this patient's two volumes. Float32 avoids float64 MRI storage;
    # retain the mask's original dtype until its labels have been validated.
    flair_nifti = nib.load(flair_path)
    seg_nifti = nib.load(seg_path)
    print(f"Original FLAIR dimensions: {flair_nifti.shape}")
    print(f"Original segmentation dimensions: {seg_nifti.shape}")
    if flair_nifti.shape != seg_nifti.shape:
        raise ValueError("FLAIR and segmentation volume shapes do not match.")
    if len(flair_nifti.shape) != 3 or any(size == 0 for size in flair_nifti.shape):
        raise ValueError("Expected nonempty three-dimensional NIfTI volumes.")
    if not np.allclose(flair_nifti.affine, seg_nifti.affine, rtol=0, atol=1e-5):
        raise ValueError("FLAIR and segmentation spatial coordinates do not match.")

    flair = flair_nifti.get_fdata(dtype=np.float32, caching="unchanged")
    segmentation = np.asanyarray(seg_nifti.dataobj)
    labels = np.unique(segmentation)
    print(f"Unique segmentation labels: {labels.tolist()}")
    if not np.isin(labels, [0, 1, 2, 4]).all():
        raise ValueError(f"Unexpected segmentation labels: {labels.tolist()}; expected 0, 1, 2, 4.")
    segmentation = np.asarray(segmentation, dtype=np.uint8)
    bounds = normalization_bounds(flair)

    # Axis 2 is axial for these BraTS volumes. Count tumor pixels one slice at
    # a time, avoiding a second full-volume binary mask allocation.
    tumor_counts = np.array([
        np.count_nonzero(segmentation[:, :, index] > 0)
        for index in range(segmentation.shape[2])
    ])
    tumor_slices = evenly_spaced(np.flatnonzero(tumor_counts > 0), 6)
    background_slices = evenly_spaced(np.flatnonzero(tumor_counts == 0), 2)
    if not tumor_slices:
        raise ValueError("No tumor-positive axial slice is available for the smoke test visualization.")
    print(f"Selected tumor-positive slice indices: {tumor_slices}")
    print(f"Selected tumor-free slice indices: {background_slices}")
    selected_slices = sorted(tumor_slices + background_slices)
    sample_paths = {
        index: sample_dir / f"{patient_id}_slice_{index:03d}.npz"
        for index in selected_slices
    }

    # Check ALL planned output files before saving any of them.
    existing = [
        path for path in [*sample_paths.values(), plot_path]
        if path.exists() or path.is_symlink()
    ]
    if existing and not args.overwrite:
        raise ValueError(
            "Outputs already exist; nothing was overwritten:\n"
            + "\n".join(str(path) for path in existing)
            + "\nRerun with --overwrite only if you intend to replace these outputs."
        )

    # Keep only the tiny selected samples in memory, validating every pair before
    # writing. MRI uses bilinear interpolation; masks use nearest-neighbor only.
    samples = []
    for index in selected_slices:
        image = normalize_slice(flair[:, :, index], bounds)
        mask = (segmentation[:, :, index] > 0).astype(np.uint8)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        np.clip(image, 0, 1, out=image)
        if image.shape != (256, 256) or image.dtype != np.float32:
            raise ValueError(f"Slice {index}: expected a float32 image of shape (256, 256).")
        if mask.shape != (256, 256) or mask.dtype != np.uint8:
            raise ValueError(f"Slice {index}: expected a uint8 mask of shape (256, 256).")
        if not np.isfinite(image).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Slice {index}: invalid image intensities or nonbinary resized mask.")
        samples.append((index, image, mask))

    # Pick the selected sample with the most tumor for a clear visualization.
    plot_sample = max(samples, key=lambda sample: np.count_nonzero(sample[2]))
    if not np.any(plot_sample[2]):
        raise ValueError("No selected mask retains tumor after resizing.")
    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    for index, image, mask in samples:
        with sample_paths[index].open("wb" if args.overwrite else "xb") as handle:
            np.savez_compressed(handle, image=image, mask=mask)
        print(
            f"Saved {sample_paths[index].name}: "
            f"image {image.shape} {image.dtype}, range [{image.min():.4f}, {image.max():.4f}]; "
            f"mask {mask.shape} {mask.dtype}, values {np.unique(mask).tolist()}"
        )
    index, image, mask = plot_sample
    save_visualization(image, mask, patient_id, index, plot_path, args.overwrite)
    print(f"Saved samples: {len(samples)} ({len(tumor_slices)} tumor-positive, {len(background_slices)} tumor-free)")
    print(f"Saved visualization: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, nib.filebasedimages.ImageFileError, cv2.error) as error:
        sys.exit(f"Error: {error}")
