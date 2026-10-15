"""Audit VGG19 loading offline; --download explicitly permits official weights.

Default mode never downloads. If no checkpoint is cached, architecture/freezing
checks run with random weights and pretrained verification is reported PENDING.
If a checkpoint exists, its official hash and every encoder weight/bias are checked.

Only --download may call the official weights API's downloader. Verification is
CPU-only and inference-only. Reference tensor hashes are retained instead of a
second VGG19 model or a second full checkpoint. No BraTS files are read or changed.
"""

import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import VGG19_Weights

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.model_vgg19_unet import VGG19UNet


def checkpoint_path(weights):
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name


def check_checkpoint_hash(path, weights):
    """Confirm cached bytes match the SHA256 prefix in Torchvision's official URL."""
    expected_prefix = Path(urlparse(weights.url).path).stem.rsplit("-", 1)[-1]
    if len(expected_prefix) < 8 or any(character not in "0123456789abcdef" for character in expected_prefix):
        raise ValueError("Official checkpoint URL must contain a SHA256 hash prefix.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if not digest.hexdigest().startswith(expected_prefix):
        raise ValueError(f"Checkpoint checksum does not match official weights: {path}. No file was modified.")


def tensor_signature(tensor):
    """Capture shape, dtype, and exact tensor bytes without storing another copy."""
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return tuple(tensor.shape), tensor.dtype, digest


def reference_signatures(weights, allow_download):
    """Use the official weights API, with cache-only access unless consent is given."""
    path = checkpoint_path(weights)
    if not path.is_file() and not allow_download:
        return None
    if path.is_file():
        # Torchvision checks hashes during downloads; explicitly check a cached
        # file too, before deserializing it.
        check_checkpoint_hash(path, weights)
    # The official API handles both current and older checkpoint file formats.
    # main() blocks its download helper offline, even if the cache disappears.
    state = weights.get_state_dict(progress=allow_download, check_hash=True, weights_only=True)
    check_checkpoint_hash(path, weights)
    signatures = {
        name: tensor_signature(tensor)
        for name, tensor in state.items() if name.startswith("features.")
    }
    if not signatures:
        raise ValueError("Official checkpoint contains no VGG19 feature tensors.")
    # Release the large checkpoint, including its unused classifier, BEFORE
    # constructing the segmentation model. Only small hashes survive this call.
    del state
    return signatures


def verify_encoder_tensors(model, expected):
    """Check every encoder tensor against its original Torchvision feature key."""
    matched = set()
    for stage in model.encoder:
        # Sequential slicing preserves Torchvision's original layer numbers.
        for layer_number, layer in stage.named_children():
            for name, parameter in layer.named_parameters(recurse=False):
                key = f"features.{layer_number}.{name}"
                if key in matched:
                    raise ValueError(f"Encoder feature was included more than once: {key}")
                if key not in expected or tensor_signature(parameter) != expected[key]:
                    raise ValueError(f"Encoder tensor does not match pretrained weights: {key}")
                matched.add(key)
    if matched != set(expected):
        raise ValueError(f"Pretrained feature tensors missing from encoder: {sorted(set(expected) - matched)}")
    return len(matched)


def verify_model_contract(model):
    """Verify five stages, freezing, trainable decoder/head, and full-size raw logits."""
    if len(model.encoder) != 5:
        raise ValueError("Expected five VGG19 convolutional encoder stages.")
    convolution_counts = []
    stage_channels = []
    for stage in model.encoder:
        convolutions = [layer for layer in stage if isinstance(layer, nn.Conv2d)]
        convolution_counts.append(len(convolutions))
        stage_channels.append(convolutions[-1].out_channels if convolutions else None)
    if convolution_counts != [2, 2, 4, 4, 4] or stage_channels != [64, 128, 256, 512, 512]:
        raise ValueError("Encoder does not match the five official VGG19 convolutional stages.")
    if sum(isinstance(layer, nn.MaxPool2d) for layer in model.encoder.modules()) != 4:
        raise ValueError("Expected four encoder pools; the final VGG19 pool is intentionally omitted.")
    if any(isinstance(layer, nn.Linear) for layer in model.modules()):
        raise ValueError("VGG19 classification layers must not be retained.")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise ValueError("An encoder parameter is not frozen.")
    if not all(parameter.requires_grad for parameter in model.decoder.parameters()):
        raise ValueError("Decoder parameters must remain trainable.")
    if not all(parameter.requires_grad for parameter in model.output_conv.parameters()):
        raise ValueError("The output head must remain trainable.")
    if model.output_conv.out_channels != 1 or model.output_conv.kernel_size != (1, 1):
        raise ValueError("Expected a one-channel 1x1 output convolution.")

    # Capture the head's output to check forward returns these exact raw logits,
    # rather than a sigmoid-transformed tensor. No weights or inputs are changed.
    head_outputs = []
    handle = model.output_conv.register_forward_hook(
        lambda module, inputs, output: head_outputs.append(output.detach().clone())
    )
    try:
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 256, 256, dtype=torch.float32))
    finally:
        handle.remove()
    if tuple(output.shape) != (1, 1, 256, 256) or not torch.isfinite(output).all():
        raise ValueError("Model must return finite logits of shape (1, 1, 256, 256).")
    if len(head_outputs) != 1 or not torch.equal(output, head_outputs[0]):
        raise ValueError("Forward must return the raw output-convolution logits unchanged.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true",
        help="Explicitly permit downloading official VGG19 weights if they are missing.",
    )
    args = parser.parse_args(argv)
    weights = VGG19_Weights.DEFAULT
    transforms = weights.transforms()
    print(f"Official weights: {weights}")
    print(f"Checkpoint: {checkpoint_path(weights)}")
    print(f"Cached checkpoint present: {checkpoint_path(weights).is_file()}")
    print(f"Downloads explicitly enabled: {args.download}")
    print(f"ImageNet RGB mean: {transforms.mean}; std: {transforms.std}")
    print(f"Classification transform: resize {transforms.resize_size}, center crop {transforms.crop_size}")
    print("BraTS FLAIR is grayscale repeated across three channels; ImageNet weights transfer from a different image domain.")
    print("No ImageNet normalization or classification crop is applied; Dataset changes are deferred.")

    # Offline mode also blocks the builder's download helper. Thus a cache file
    # disappearing between checks causes an error instead of an accidental fetch.
    guard = nullcontext() if args.download else patch(
        "torch.hub.download_url_to_file",
        side_effect=PermissionError("Network download blocked: explicit --download is required."),
    )
    with guard:
        expected = reference_signatures(weights, allow_download=args.download)
        model = VGG19UNet(pretrained=expected is not None, freeze_encoder=True).cpu()
        matched = verify_encoder_tensors(model, expected) if expected is not None else None
        verify_model_contract(model)
    print("Architecture, frozen encoder, trainable decoder/head, and raw logits: PASS")
    if matched is None:
        print("Pretrained encoder verification: PENDING (no cached checkpoint; random weights used for architecture checks)")
        print("No weights were downloaded. Rerun with --download only when you approve the official weights download.")
    else:
        print(f"Pretrained encoder verification: PASS ({matched} weight/bias tensors match the official checkpoint)")
    print("No training was performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Pretrained encoder verification: FAIL\n{error}")
