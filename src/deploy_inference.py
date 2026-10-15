"""Inference helpers that reproduce the project's 2D BraTS FLAIR pipeline.

Research/demo use only; these predictions are not medical diagnoses.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

from src.dataset_nifti_2d import normalize_flair_slice
from src.input_normalization import normalize_imagenet_image
from src.model_vgg19_unet import VGG19UNet

IMAGE_SIZE = 256
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "deployment_weights.pt"


def preprocess_flair(image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Match train_2d.py: per-slice percentile clip, resize, ImageNet norm."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Please provide a single, two-dimensional grayscale FLAIR MRI slice.")
    if image.size == 0 or image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError("Image must have height and width of at least 16 pixels.")
    if np.iscomplexobj(image) or not np.isfinite(image).all():
        raise ValueError("MRI image intensities must be real and finite.")

    normalized = normalize_flair_slice(image)
    resized = cv2.resize(normalized, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0, 1).astype(np.float32)
    three_channel = torch.from_numpy(resized).unsqueeze(0).repeat(3, 1, 1)
    model_input = normalize_imagenet_image(three_channel).unsqueeze(0)
    return resized, model_input


def create_overlay(normalized_image: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    if normalized_image.shape != binary_mask.shape:
        raise ValueError("The prediction mask and MRI slice must have the same size.")
    grayscale = np.clip(normalized_image * 255, 0, 255).astype(np.uint8)
    base = np.repeat(grayscale[..., np.newaxis], 3, axis=2)
    result = base.astype(np.float32)
    affected = binary_mask.astype(bool)
    if affected.any():
        red = np.array([255, 45, 60], dtype=np.float32)
        result[affected] = (1 - alpha) * result[affected] + alpha * red
    return np.clip(result, 0, 255).astype(np.uint8)


def predict_slice(model: VGG19UNet, model_input: torch.Tensor, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    model.eval()
    with torch.inference_mode():
        logits = model(model_input.to(device="cpu", dtype=torch.float32))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    if probability.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(probability).all():
        raise RuntimeError("Model output is invalid.")
    mask = (probability >= threshold).astype(np.uint8)
    return probability, mask


def download_verified_weights(url: str, sha256: str, *, directory: Path | None = None) -> Path:
    """Download HTTPS model weights once after a required SHA-256 integrity check."""
    if urlparse(url).scheme.lower() != "https":
        raise ValueError("MODEL_URL must use HTTPS.")
    sha256 = sha256.lower().strip()
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValueError("Set MODEL_SHA256 to the 64-character SHA-256 hash of your weights file.")

    directory = Path(directory) if directory else Path(tempfile.gettempdir()) / "brain-tumor-model-cache"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{sha256}.pt"
    if destination.is_file() and _file_sha256(destination) == sha256:
        return destination
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        hasher = hashlib.sha256()
        byte_count = 0
        request = Request(url, headers={"User-Agent": "BrainTumorResearchDemo/1.0"})
        with urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            if urlparse(response.url).scheme.lower() != "https":
                raise ValueError("MODEL_URL redirected to an insecure URL.")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Remote weights exceed the 512 MB download limit.")
                hasher.update(chunk)
                handle.write(chunk)
        if hasher.hexdigest() != sha256:
            raise ValueError("Downloaded checkpoint failed SHA-256 integrity verification.")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_weights() -> Path:
    custom_path = os.environ.get("MODEL_PATH", "").strip()
    path = Path(custom_path).expanduser() if custom_path else DEFAULT_WEIGHTS_PATH
    if path.is_file():
        return path
    url = os.environ.get("MODEL_URL", "").strip()
    if url:
        return download_verified_weights(url, os.environ.get("MODEL_SHA256", ""))
    raise FileNotFoundError(
        f"No model checkpoint found at {path}. Export the trained weights with "
        "'python -m src.export_inference_checkpoint --input ...' or configure MODEL_URL and MODEL_SHA256."
    )


def load_inference_model(weights_path: Path) -> tuple[VGG19UNet, dict]:
    """Restore trusted project weights using PyTorch's safer tensor-only loader."""
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is not a PyTorch state-dictionary or project checkpoint.")
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = {key: payload.get(key) for key in ("epoch", "validation_metrics", "artifact_label", "note", "training_config")}
    else:
        state = payload
        metadata = {}
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint does not contain a compatible tensor state dictionary.")
    model = VGG19UNet(pretrained=False, freeze_encoder=True).to("cpu").eval()
    model.load_state_dict(state, strict=True)
    return model, metadata
