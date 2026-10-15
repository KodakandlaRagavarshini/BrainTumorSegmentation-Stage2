"""Run exactly one real, tumor-positive training step when executed directly."""

import hashlib
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

# Support running this file directly without installing the project as a package.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.dataset_nifti_2d import BraTSNiftiSliceDataset
from src.losses import FocalTverskyLoss
from src.model_vgg19_unet import VGG19UNet


def select_tumor_positive_slice(dataset):
    """Choose the first positive training record in CSV order, using metadata only."""
    if dataset.split != "train":
        raise ValueError("The training smoke test requires the train split.")
    for index, record in enumerate(dataset.records):
        if record["split"] != "train":
            raise ValueError("Training Dataset contains a record from another split.")
        try:
            count = int(record["tumor_pixel_count"])
            flag = int(record["has_tumor"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Slice metadata needs integer tumor_pixel_count and has_tumor values.") from error
        if count < 0 or flag not in (0, 1) or flag != int(count > 0):
            raise ValueError("Slice metadata has inconsistent tumor counts and flags.")
        if count > 0:
            return index, record
    raise ValueError("No tumor-positive training slice is present in the index.")


def _check_parameter_configuration(model):
    """Check the existing model's encoder/decoder API and return trainable weights."""
    encoder_parameters = list(model.encoder.parameters())
    decoder_parameters = list(model.decoder.parameters())
    if not encoder_parameters or any(parameter.requires_grad for parameter in encoder_parameters):
        raise ValueError("All encoder parameters must be frozen.")
    if not decoder_parameters or any(not parameter.requires_grad for parameter in decoder_parameters):
        raise ValueError("All decoder parameters must remain trainable.")
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def build_optimizer(model):
    """Give Adam only trainable decoder/head parameters, never frozen encoder weights."""
    parameters = _check_parameter_configuration(model)
    # Disabling foreach avoids its extra temporary parameter-sized tensor lists.
    return torch.optim.Adam(list(parameters.values()), lr=1e-4, foreach=False)


def choose_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _encoder_fingerprint(encoder):
    """Hash frozen weights one tensor at a time instead of cloning the full encoder."""
    digest = hashlib.sha256()
    for name, parameter in encoder.named_parameters():
        digest.update(name.encode("utf-8"))
        # Only a single CPU tensor copy is needed at a time for MPS parameters.
        # A byte view avoids creating another full bytes copy of that tensor.
        digest.update(memoryview(parameter.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def run_one_training_step(model, images, targets, criterion, optimizer, spatial_size=(256, 256)):
    """Perform one forward/backward/update and return scalar verification results.

    The default size matches real BraTS inputs. Tests can use a tiny model and
    smaller synthetic images by setting spatial_size. No output files are saved.
    """
    if tuple(images.shape) != (1, 3, *spatial_size):
        raise ValueError(f"Expected image shape {(1, 3, *spatial_size)}, got {tuple(images.shape)}.")
    if tuple(targets.shape) != (1, 1, *spatial_size):
        raise ValueError(f"Expected mask shape {(1, 1, *spatial_size)}, got {tuple(targets.shape)}.")
    if images.dtype != torch.float32 or targets.dtype != torch.float32:
        raise TypeError("Images and masks must be float32; mixed precision is disabled.")
    if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
        raise ValueError("Images and masks must contain only finite values.")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("Image intensities must be within [0, 1].")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Masks must contain only binary values 0 and 1.")
    if not (targets > 0).any():
        raise ValueError("The selected training slice must retain tumor pixels after resizing.")

    trainable = _check_parameter_configuration(model)
    all_parameters = list(model.parameters())
    if images.device != targets.device or any(parameter.device != images.device for parameter in all_parameters):
        raise ValueError("Images, masks, and model must be on the same selected device.")
    if any(parameter.dtype != torch.float32 for parameter in all_parameters):
        raise TypeError("Model parameters must be float32.")
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected_ids = {id(parameter) for parameter in trainable.values()}
    if len(optimizer_parameters) != len(expected_ids) or {id(parameter) for parameter in optimizer_parameters} != expected_ids:
        raise ValueError("Optimizer must contain exactly the model's trainable parameters.")

    # Dataset inputs do not need gradients. Detaching makes that explicit: with
    # frozen encoder weights, PyTorch builds no graph for the encoder operations.
    # Its skip features still exist because the trainable decoder needs them.
    images, targets = images.detach(), targets.detach()
    encoder_before = _encoder_fingerprint(model.encoder)
    # This copies only the small decoder/head, not the large frozen encoder.
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    decoder_ids = {id(parameter) for parameter in model.decoder.parameters()}
    encoder_graph_checks = []

    def check_encoder_output(module, inputs, output):
        detached = not output.requires_grad and output.grad_fn is None
        encoder_graph_checks.append(detached)
        if not detached:
            raise RuntimeError("Frozen encoder retained an unnecessary autograd graph.")

    # VGG19's encoder is a ModuleList, so hook its stages individually.
    stages = list(model.encoder.children()) or [model.encoder]
    handles = [stage.register_forward_hook(check_encoder_output) for stage in stages]
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if tuple(logits.shape) != tuple(targets.shape):
            raise ValueError(f"Expected output shape {tuple(targets.shape)}, got {tuple(logits.shape)}.")
        if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
            raise ValueError("Model output must be finite float32 logits.")
        if not encoder_graph_checks or not all(encoder_graph_checks):
            raise RuntimeError("Frozen encoder outputs were not verified.")

        # The existing loss applies sigmoid internally; pass raw logits once.
        loss = criterion(logits, targets)
        if loss.ndim != 0 or not torch.isfinite(loss) or not loss.requires_grad:
            raise RuntimeError("Loss must be a finite scalar connected to trainable parameters.")
        loss.backward()
        for name, parameter in trainable.items():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Nonfinite gradient in trainable parameter {name}.")
        gradients_ok = any(parameter.grad is not None for parameter in model.decoder.parameters())
        if not gradients_ok:
            raise RuntimeError("No trainable decoder parameter received a gradient.")
        if any(parameter.grad is not None for parameter in model.encoder.parameters()):
            raise RuntimeError("Frozen encoder parameters unexpectedly received gradients.")
        optimizer.step()  # Exactly ONE optimizer update; no epoch/batch loop.
    finally:
        for handle in handles:
            handle.remove()

    trainable_changed = any(not torch.equal(before[name], parameter.detach()) for name, parameter in trainable.items())
    decoder_changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in trainable.items() if id(parameter) in decoder_ids
    )
    encoder_frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in model.encoder.parameters())
    encoder_unchanged = _encoder_fingerprint(model.encoder) == encoder_before
    if not trainable_changed or not decoder_changed:
        raise RuntimeError("No trainable decoder parameter changed after the optimizer step.")
    if not encoder_frozen or not encoder_unchanged:
        raise RuntimeError("Frozen encoder parameters changed or are no longer frozen.")
    return {
        "initial_loss": loss.item(), "output_shape": tuple(logits.shape),
        "gradients_ok": gradients_ok, "decoder_changed": decoder_changed,
        "trainable_changed": trainable_changed, "encoder_frozen": encoder_frozen,
        "encoder_unchanged": encoder_unchanged,
        "encoder_graph_free": all(encoder_graph_checks), "passed": True,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    device = choose_device()
    print(f"Device: {device}", flush=True)
    dataset = BraTSNiftiSliceDataset(
        root / "data/processed/slice_index_smoke.csv", root / "data/raw/brats2020", "train",
    )
    selected_index, record = select_tumor_positive_slice(dataset)
    print(f"Patient ID: {record['patient_id']}")
    print(f"Slice index: {record['slice_index']}")
    # A one-element Subset prevents even a second indexed MRI slice being loaded.
    loader = DataLoader(Subset(dataset, [selected_index]), batch_size=1, num_workers=0, shuffle=False)
    images, targets = next(iter(loader))
    images = images.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    torch.manual_seed(42)
    model = VGG19UNet(pretrained=False, freeze_encoder=True).to(device=device, dtype=torch.float32)
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75).to(device)
    optimizer = build_optimizer(model)
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Target shape: {tuple(targets.shape)}", flush=True)
    result = run_one_training_step(model, images, targets, criterion, optimizer)
    print(f"Output shape: {result['output_shape']}")
    print(f"Initial loss: {result['initial_loss']:.6f}")
    print(f"Gradients successfully computed: {result['gradients_ok']}")
    print(f"Decoder parameters changed: {result['decoder_changed']}")
    print(f"Encoder parameters remained frozen: {result['encoder_frozen']}")
    print(f"Encoder parameters unchanged: {result['encoder_unchanged']}")
    print(f"Encoder outputs have no autograd graph: {result['encoder_graph_free']}")
    print("One-step training test: PASS")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, IndexError, TypeError) as error:
        # Report the failure on the chosen device; never retry training on CPU.
        sys.exit(f"One-step training test: FAIL\n{error}")
