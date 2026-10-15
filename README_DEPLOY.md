# Stage 2 Deployment: Brain Tumor Segmentation (VGG19 U-Net)

This package preserves the original `src/` training pipeline. The added Streamlit app runs **CPU inference** on a trained 2D BraTS FLAIR checkpoint. It displays the MRI, predicted whole-tumor mask and overlay. It cannot predict until weights from Phase 5 are provided.

**Research prototype only. Not for clinical diagnosis or treatment.**

## 1. Copy this code into your Mac's existing VS Code project

If you want to keep all your existing data, copy `app.py`, `requirements.txt`, `.streamlit/config.toml`, `src/deploy_inference.py`, and `src/export_inference_checkpoint.py` into your original project. Do not overwrite trained weights or dataset files. The included `src/` copies are from your uploaded code-only ZIP.

## 2. Find and verify the real checkpoint

In VS Code's terminal at project root:

```bash
find outputs/checkpoints -maxdepth 2 -type f \( -name '*.pt' -o -name '*.pth' \) -exec ls -lh {} \;
python -m src.verify_checkpoint --checkpoint outputs/checkpoints/best_validation_checkpoint.pt
```

If your checkpoint has a different name, substitute its path. `train_2d.py` only writes one if training was started with `--save-checkpoint`; if absent, you must save a checkpoint from training before deployment.

**Important:** the original training defaults are tiny bounded smoke runs (1 epoch, 2 train batches, 2 val batches). Verify that the checkpoint is from meaningful training and that validation predictions/metrics have actually been inspected. The exported checkpoint may still say "development_experiment" and that is not independently validated.

## 3. Export inference-only weights (remove optimizer tensors)

```bash
python -m src.export_inference_checkpoint --input outputs/checkpoints/best_validation_checkpoint.pt
ls -lh models/deployment_weights.pt
shasum -a 256 models/deployment_weights.pt
```

No trained data is included in this ZIP. The exporter saves a smaller state-dict-based checkpoint at `models/deployment_weights.pt`. The SHA-256 printed by the exporter is used to verify external downloads.

## 4. Local demo

Activate the project's virtual environment:

```bash
source .venv/bin/activate
python -m pip install streamlit pillow nibabel opencv-python-headless
python -m streamlit run app.py
```

If necessary, install the matching torch/torchvision versions from `requirements.txt` in a **separate clean environment** to avoid changing your working training setup. Streamlit's terminal prints the local app URL, usually `http://localhost:8501`. Upload a BraTS FLAIR `.nii.gz` volume, choose the axial slice with the slider, and inspect the three displays. PNG/JPEG is supported for illustration but intensity and geometry may not reproduce the training distribution.

## 5. Prepare GitHub repository

Publish only your deployment code. **Never upload the patient dataset or virtual environment.** `.gitignore` excludes `data/`, `outputs/`, model weights and `secrets.toml` by default. Use the web GitHub repository creation flow, or from VS Code terminal:

```bash
git init
git add app.py requirements.txt README_DEPLOY.md .gitignore .streamlit/config.toml src/
git commit -m "Add Stage 2 Streamlit inference app"
```

To push, create a repository on GitHub, then follow the `git remote add origin ...` and `git push -u origin main` commands shown by GitHub (branch name depends on your setup).

## 6. Easiest option for tomorrow: include weights with Git if under 100 MB

For this specific VGG19 U-Net the inference-only state dict is usually around **83 MiB**, while the full training checkpoint may be much larger because it contains optimizer state. Check the actual size after export. GitHub warns above 50 MB and rejects files above 100 MB. If the exported weights are **under 100 MB**, you can add the single file to your GitHub repository from the terminal (Git command line, not the GitHub website's file uploader):

```bash
git add -f models/deployment_weights.pt
git commit -m "Add inference-only model weights for Stage 2 demo"
git push
```

The app will find `models/deployment_weights.pt` automatically on Streamlit Cloud. **In that case, you do not need Hugging Face or any Streamlit model secrets.** Commit only public, non-sensitive weights that you have rights to redistribute. Larger weights are better hosted externally.

## 7. Alternative: host model weights outside GitHub

The trained state dict may exceed GitHub's normal 100 MB per-file limit. Create a **public Hugging Face model repository** and upload `models/deployment_weights.pt` (not MRI data). A raw model URL has this structure:

`https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_REPO/resolve/main/deployment_weights.pt`

In your Streamlit Cloud deployment **Advanced settings → Secrets**, enter, with your actual URL and SHA-256 from step 3:

```toml
MODEL_URL = "https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_REPO/resolve/main/deployment_weights.pt"
MODEL_SHA256 = "REPLACE_WITH_REAL_64_CHARACTER_HASH"
```

Do not commit actual credentials or private datasets to GitHub. If you want private model hosting, it requires an access-controlled solution rather than this public URL mechanism.

For local testing with hosted weights you can also set environment variables:

```bash
MODEL_URL="https://huggingface.co/.../resolve/main/deployment_weights.pt" MODEL_SHA256="YOUR_HASH" python -m streamlit run app.py
```

The app downloads over HTTPS only and validates the SHA-256 before loading with `torch.load(..., weights_only=True)`.

## 8. Deploy Streamlit Community Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app**, select the GitHub repo/branch and set entrypoint to `app.py`.
3. Choose Python **3.12** in Advanced settings. If you hosted the weights on Hugging Face, add the `MODEL_URL` and `MODEL_SHA256` secrets above. If the model weights are committed directly in `models/`, leave these unset. Then Deploy.
4. Wait for dependencies and weights to load. Test with a publicly shareable, non-sensitive BraTS FLAIR sample.
5. Share the `*.streamlit.app` URL during Stage 2 review.

If cloud memory is insufficient for VGG19 CPU inference, use the **local Streamlit URL for the review** or deploy on a service with more memory. Your M2 device has MPS; the cloud app uses CPU and therefore will be slower.

## Interpretation limitations

- Supports **2D whole tumor** segmentation, not 3D reconstruction, tumor subtype prediction, or separate BraTS labels 1/2/4.
- The image is resized to 256x256, and the displayed pixel count is not a clinical volume.
- The uploaded code includes a 2025 paper reference, but no trained checkpoint or reproducible final accuracy results; do not claim the paper's published performance as your model's results.
- Valid test metrics must be computed on an independent patient-level test set, not copied from the paper.
