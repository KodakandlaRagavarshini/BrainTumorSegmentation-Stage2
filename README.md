# Brain Tumor Segmentation — Stage 2

PyTorch implementation of a **2D BraTS FLAIR whole-tumor segmentation** model with a VGG19 encoder, a U-Net decoder, and transfer learning. Intended for research/educational development, not medical diagnosis.

## Code in this package

- `src/model_vgg19_unet.py` — VGG19-U-Net architecture.
- `src/dataset_nifti_2d.py` — BraTS NIfTI FLAIR reader and training preprocessing.
- `src/input_normalization.py` — ImageNet normalization.
- `src/train_2d.py` — bounded training/validation experiments and checkpoint save.
- `src/metrics.py` and `src/losses.py` — evaluation and loss functions.
- `src/export_inference_checkpoint.py` — turn a saved Phase 5 checkpoint into smaller deployable weights.
- `src/deploy_inference.py` — matching preprocessing, checkpoint loading and prediction utilities.
- `app.py` — Streamlit deployment UI.
- `tests/` — project and inference tests.

**Start with [`README_DEPLOY.md`](README_DEPLOY.md) for step-by-step Mac and Streamlit Community Cloud deployment instructions.**

This code-only package does not include MRI data or trained weights. Predictions require a trained checkpoint obtained from your own project. The original data path used for training is `data/raw/brats2020/`, and saved checkpoints default to `outputs/checkpoints/` when enabled. For the original paper, see your uploaded research PDF; do not confuse published paper metrics with your own results.
