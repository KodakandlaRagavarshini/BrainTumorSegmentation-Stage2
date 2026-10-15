"""Visualize the axial slice with the most tumor in BraTS patient 082."""

from pathlib import Path

import matplotlib

# Select a noninteractive backend before importing pyplot (no GUI needed).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    # Resolve paths from this file so running from another directory also works.
    project_root = Path(__file__).resolve().parents[1]
    patient_dir = (
        project_root
        / "data/raw/brats2020/BraTS2020_TrainingData"
        / "MICCAI_BraTS2020_TrainingData/BraTS20_Training_082"
    )
    output_path = (
        project_root / "outputs/plots/brats2020_patient082_visualization.png"
    )

    # NiBabel loads headers first; its data proxies let us read slices on demand.
    # These reads never modify the original dataset files or its symbolic link.
    flair = nib.load(patient_dir / "BraTS20_Training_082_flair.nii")
    segmentation = nib.load(patient_dir / "BraTS20_Training_082_seg.nii")
    print(f"FLAIR image dimensions: {flair.shape}")
    print(f"Segmentation mask dimensions: {segmentation.shape}")
    if flair.shape != segmentation.shape:
        raise ValueError("FLAIR and segmentation dimensions must match.")
    if len(flair.shape) != 3:
        raise ValueError("Expected three-dimensional BraTS volumes.")

    # Scan one mask slice at a time to keep memory usage low. Every nonzero
    # segmentation label belongs to the binary whole-tumor mask.
    labels = set()
    tumor_counts = np.zeros(segmentation.shape[2], dtype=np.int64)
    for index in range(segmentation.shape[2]):
        mask_slice = np.asarray(segmentation.dataobj[:, :, index])
        labels.update(np.unique(mask_slice).tolist())
        tumor_counts[index] = np.count_nonzero(mask_slice)
    print(f"Unique segmentation label values: {sorted(labels)}")

    # The third dimension is the axial slice axis for these BraTS files.
    # argmax chooses the first slice when several slices share the maximum.
    slice_index = int(np.argmax(tumor_counts))
    print(f"Selected slice index: {slice_index}")
    print(f"Number of tumor pixels in the selected slice: {tumor_counts[slice_index]}")

    # Read only the selected FLAIR slice instead of the entire MRI volume.
    flair_slice = np.asarray(flair.dataobj[:, :, slice_index], dtype=np.float32)
    tumor_mask = np.asarray(segmentation.dataobj[:, :, slice_index]) != 0

    # Transpose both slices consistently to display the axial plane upright.
    display_flair = flair_slice.T
    display_mask = tumor_mask.T
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_flair, cmap="gray", origin="lower")
    axes[0].set_title("Original FLAIR MRI")
    axes[1].imshow(display_mask, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("Binary whole-tumor mask")
    axes[2].imshow(display_flair, cmap="gray", origin="lower")

    # An RGBA overlay gives tumor pixels red color with 40% opacity;
    # background pixels have zero opacity, leaving the MRI visible.
    overlay = np.zeros((*display_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = display_mask * 0.4
    axes[2].imshow(overlay, origin="lower")
    axes[2].set_title("FLAIR with tumor overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"BraTS 2020 — Patient 082 — Axial slice {slice_index}")
    fig.tight_layout()

    # Create the output folder if needed, save the PNG, and release the figure.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
