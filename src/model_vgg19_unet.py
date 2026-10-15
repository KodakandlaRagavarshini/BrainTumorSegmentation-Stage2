"""A small VGG19 encoder/U-Net decoder for binary whole-tumor logits.

Architectural choices compared with the original U-Net (arxiv.org/abs/1505.04597):
* Torchvision's five VGG19 convolutional stages replace the original encoder.
* VGG19's final max-pool is omitted: the deepest feature is at 1/16 resolution.
* Four decoder blocks use 128, 64, 32, 16 channels to limit memory use.
* Bilinear upsampling replaces learned up-convolutions; padded convolutions
  preserve spatial size, so skip features need no cropping.

Inputs are the Dataset's grayscale BraTS FLAIR slices repeated across three
channels, currently in [0, 1]. ImageNet pretrained weights come from RGB natural
images: this is transfer between different image domains, not MRI pretraining.

Official VGG19 RGB normalization uses mean [0.485, 0.456, 0.406] and standard
deviation [0.229, 0.224, 0.225], applied as (channel - mean) / std. Torchvision's
classification transform also resizes to 256 and center-crops to 224. We do not
apply that transform to segmentation image/mask pairs: aligned 256x256 geometry
and consistent input normalization will be handled in a separate stage.
Source: docs.pytorch.org/vision/0.29/models/generated/torchvision.models.vgg19.html

No input normalization or sigmoid is applied here. Pretrained weights are
accepted only from a local cache; this module does not approve downloads.
"""

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


def _vgg19_features(pretrained):
    """Retain only convolutional features, with no classification layers."""
    if pretrained:
        # Use the official weights enum, rather than the old pretrained API.
        # The builder reuses this cached file. A missing file is an error, never
        # permission to download: downloads require separate explicit approval.
        weights = VGG19_Weights.DEFAULT
        filename = Path(urlparse(weights.url).path).name
        cached_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not cached_path.is_file():
            raise PermissionError(
                f"VGG19 pretrained weights are not cached at {cached_path}. "
                "A weights download requires explicit user approval and is disabled "
                "in this stage. Use pretrained=False."
            )
        return vgg19(weights=weights, progress=False).features

    # The meta device creates the architecture without allocating weight storage
    # for VGG19's large unused classifier. Allocate only the feature weights on
    # CPU, then initialize them as Torchvision initializes its VGG convolutions.
    with torch.device("meta"):
        features = vgg19(weights=None).features
    features.to_empty(device="cpu")
    for layer in features.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return features


class DecoderBlock(nn.Module):
    """Upsample, concatenate an aligned encoder skip, then apply two convolutions."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature, skip):
        if feature.ndim != 4 or skip.ndim != 4:
            raise ValueError("Decoder features and skips must be four-dimensional tensors.")
        if feature.shape[0] != skip.shape[0]:
            raise ValueError("Decoder feature and skip batch sizes do not match.")
        if feature.shape[1] != self.in_channels or skip.shape[1] != self.skip_channels:
            raise ValueError(
                f"Decoder channel mismatch: expected {self.in_channels} feature and "
                f"{self.skip_channels} skip channels; got {feature.shape[1]} and {skip.shape[1]}."
            )
        expected_size = (feature.shape[2] * 2, feature.shape[3] * 2)
        if tuple(skip.shape[2:]) != expected_size:
            raise ValueError(f"Skip spatial mismatch: expected {expected_size}, got {tuple(skip.shape[2:])}.")
        feature = F.interpolate(feature, size=expected_size, mode="bilinear", align_corners=False)
        return self.convolutions(torch.cat((feature, skip), dim=1))


class VGG19UNet(nn.Module):
    """Five encoder stages, four skip connections, and one-channel raw logits.

    Height and width must be positive multiples of 16. For 256x256 inputs the
    encoder channels/resolutions are 64/256, 128/128, 256/64, 512/32, 512/16.
    pretrained=False uses random weights and never accesses the weights cache.
    freeze_encoder=True freezes only the encoder, leaving decoder/head trainable.
    """

    def __init__(self, pretrained: bool = False, freeze_encoder: bool = False):
        super().__init__()
        features = _vgg19_features(pretrained)
        # Feature indices follow Torchvision's non-BatchNorm VGG19 architecture.
        # Stages 2-5 begin with pooling, followed by their convolution/ReLU layers.
        # Index 36 is the unused fifth pool; all 16 encoder convolutions are kept.
        stage_ranges = ((0, 4), (4, 9), (9, 18), (18, 27), (27, 36))
        self.encoder = nn.ModuleList([features[start:end] for start, end in stage_ranges])
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 512, 128),  # 16 -> 32 pixels; skip from stage 4.
            DecoderBlock(128, 256, 64),   # 32 -> 64 pixels; skip from stage 3.
            DecoderBlock(64, 128, 32),    # 64 -> 128 pixels; skip from stage 2.
            DecoderBlock(32, 64, 16),     # 128 -> 256 pixels; skip from stage 1.
        ])
        self.output_conv = nn.Conv2d(16, 1, kernel_size=1)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, images):
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
            raise ValueError("Expected input shape (B, 3, H, W) with a nonempty batch.")
        if not images.is_floating_point():
            raise TypeError("MRI inputs must be floating-point tensors.")
        height, width = images.shape[2:]
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError("Input height and width must be multiples of 16, at least 16.")

        # Save each encoder feature for its matching decoder skip connection.
        skips = []
        feature = images
        for index, (stage, channels) in enumerate(zip(self.encoder, (64, 128, 256, 512, 512), strict=True)):
            feature = stage(feature)
            expected = (images.shape[0], channels, height // (2 ** index), width // (2 ** index))
            if tuple(feature.shape) != expected:
                raise ValueError(f"Encoder stage {index + 1}: expected {expected}, got {tuple(feature.shape)}.")
            skips.append(feature)

        # The deepest feature is the bottleneck. Every shallower feature is used
        # exactly once, in reverse order; strict zip prevents skipping a feature.
        feature = skips[-1]
        for block, skip in zip(self.decoder, reversed(skips[:-1]), strict=True):
            feature = block(feature, skip)
        return self.output_conv(feature)  # Raw logits: no sigmoid or threshold.


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VGG19UNet(pretrained=False, freeze_encoder=True).to(device).eval()
    images = torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device)
    # Inference only: no gradient graph, loss, optimizer, or weight update.
    with torch.no_grad():
        logits = model(images)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Output dtype: {logits.dtype}")
    print(f"Device: {logits.device}")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    main()
