"""Streamlit research demo for the project's trained 2D BraTS VGG19 U-Net."""
from __future__ import annotations

from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from src.deploy_inference import (
    create_overlay, find_weights, load_inference_model, predict_slice, preprocess_flair
)

MAX_UPLOAD_BYTES = 125 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 45_000_000

st.set_page_config(page_title="Brain Tumor Segmentation | Research Demo", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Segmentation")
st.caption("VGG19 encoder + 2D U-Net | BraTS FLAIR | Research prototype")
st.warning("For research and education only. This tool is not a medical device and must not be used for diagnosis or treatment.")

with st.sidebar:
    st.header("Segmentation settings")
    threshold = st.slider("Tumor probability threshold", 0.05, 0.95, 0.50, 0.05)
    st.markdown("**Expected input:** BraTS-style FLAIR MRI. NIfTI volumes are preferred; ordinary images are a lower-fidelity demonstration input.")
    st.markdown("Training input geometry: **256 × 256**. The model outputs **whole-tumor** vs. background (not separate tumor subregions).")

uploaded = st.file_uploader("Upload an MRI slice or FLAIR NIfTI volume", type=["nii", "gz", "png", "jpg", "jpeg", "tif", "tiff"])


def read_uploaded_image(file) -> tuple[np.ndarray, str]:
    """Extract one 2D axial image and discard temporary uploaded data."""
    if file.size > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds 125 MB. Use a smaller NIfTI file or a single MRI slice.")
    filename = file.name.lower()
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        with tempfile.TemporaryDirectory(prefix="bt-upload-") as directory:
            path = Path(directory) / ("volume" + suffix)
            path.write_bytes(file.getvalue())
            volume = nib.load(str(path), mmap=False)
            shape = volume.shape
            if len(shape) != 3 or min(shape) < 16:
                raise ValueError("Upload a 3D NIfTI volume with at least 16 pixels along each axis.")
            index = st.slider("Axial slice number (axis 2)", min_value=0, max_value=shape[2] - 1, value=shape[2] // 2)
            result = np.asarray(volume.dataobj[:, :, index], dtype=np.float32)
        return result, f"NIfTI axial slice {index} (axis 2)"
    if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        with Image.open(file) as image:
            image = ImageOps.exif_transpose(image)
            grayscale = np.array(image.convert("L"), dtype=np.float32)
        return grayscale, "Grayscale 2D image (intensity is quantized; FLAIR-only input recommended)"
    raise ValueError("Supported files: .nii, .nii.gz, .png, .jpg, .jpeg, .tif, .tiff")


@st.cache_resource(show_spinner="Loading trained VGG19 U-Net weights…")
def cached_model(weights_path: str):
    return load_inference_model(Path(weights_path))


if uploaded is None:
    st.info("Upload a BraTS FLAIR MRI scan to begin. The app only predicts when trained weights are available.")
    st.stop()

try:
    image, description = read_uploaded_image(uploaded)
    normalized, model_input = preprocess_flair(image)
except (ValueError, OSError, UnidentifiedImageError, nib.filebasedimages.ImageFileError) as error:
    st.error(f"Unable to process the uploaded MRI: {error}")
    st.stop()

st.caption(description)

try:
    weights_path = find_weights()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Trained model not configured: {error}")
    st.info("Export your Phase 5 checkpoint with `python -m src.export_inference_checkpoint --input outputs/checkpoints/best_validation_checkpoint.pt`, then provide `models/deployment_weights.pt` locally or configure MODEL_URL and MODEL_SHA256 on Streamlit Cloud.")
    st.image(normalized, caption="Input MRI — inference unavailable", clamp=True)
    st.stop()

try:
    model, metadata = cached_model(str(weights_path))
    prob, binary = predict_slice(model, model_input, threshold=threshold)
except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
    st.error(f"Model could not be loaded or used for inference: {error}")
    st.stop()

artifact = str(metadata.get("artifact_label") or "")
if "smoke" in artifact or "development" in artifact:
    st.warning("This checkpoint is labeled as a development experiment/smoke test. Its predictions are not evidence of validated accuracy.")

col1, col2, col3 = st.columns(3)
with col1:
    st.image(normalized, caption="Preprocessed FLAIR MRI", clamp=True, use_container_width=True)
with col2:
    st.image(binary * 255, caption="Predicted binary tumor mask", clamp=True, use_container_width=True)
with col3:
    st.image(create_overlay(normalized, binary), caption="Red tumor overlay", use_container_width=True)

pixel_count = int(binary.sum())
st.metric("Predicted tumor-region pixels", f"{pixel_count:,} / {binary.size:,}")
st.caption("Pixel counts refer to the 256 × 256 resized image; they are not a physical tumor volume estimate.")
if pixel_count == 0:
    st.info("No pixels exceeded the selected probability threshold. This does not rule out a tumor; the model may be undertrained.")

if metadata.get("epoch") is not None:
    st.caption(f"Checkpoint epoch: {metadata['epoch']}")
with st.expander("Model details and interpretation"):
    st.write("The output is a pixelwise sigmoid probability for whole-tumor segmentation. It is not a classification of tumor type, prognosis, or a substitute for radiology review.")
    st.write("The training pipeline uses slice-wise 1st/99th percentile clipping, 256×256 bilinear resizing, grayscale repetition to three channels and ImageNet normalization. PNG/JPEG inputs cannot reproduce the raw NIfTI intensities exactly.")
    metrics = metadata.get("validation_metrics")
    if isinstance(metrics, dict):
        st.write("Checkpoint validation metrics (from the local training run, not recomputed here):", metrics)
