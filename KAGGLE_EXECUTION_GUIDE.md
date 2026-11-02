# Kaggle High-Speed GPU Training & Replication Guide

Follow these steps to train all deep learning models on a free Kaggle GPU in under 5 minutes, and then resume locally.

---

## Step 1: Open Kaggle & Start a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Click the **+ Create** button in the top left and select **New Notebook**.
3. On the right-hand sidebar under **Notebook options**:
   * **ACCELERATOR:** Change to **GPU T4 x 2** (or **GPU P100**).
   * **INTERNET:** Toggle to **On** (required to download the repo and dataset).

---

## Step 2: Copy & Run the Kaggle Master Script

Paste the following script into the first code cell of your Kaggle notebook and press **Shift + Enter** (or click the Run button):

```python
import os
import subprocess

print("=== 1. CLONING THESIS REPOSITORY ===")
!git clone https://github.com/mohaimin2103001/thesis.git /kaggle/working/thesis
%cd /kaggle/working/thesis

print("\n=== 2. INSTALLING DEPENDENCIES ===")
!pip install -q pyarrow scikit-image shapely pyyaml timm albumentations segmentation-models-pytorch onnx onnxruntime

print("\n=== 3. EXECUTING KAGGLE MASTER GPU TRAINING PIPELINE ===")
# Trains:
# 1. Image-Only Baseline (ResNet-34)
# 2. Wake-Only Baseline (Masked SAR)
# 3. Geometry-Only Baseline (MLP on Signed Curvatures)
# 4. Deep Visual + Geometry Fusion Model
# 5. Physics-Guided Fusion Model (ROT ~ v * kappa)
!python kaggle/KAGGLE_MASTER_TRAINING.py

print("\n=== 4. PACKAGING TRAINED CHECKPOINTS ===")
!zip -r /kaggle/working/thesis_results.zip checkpoints/ data/processed/ configs/

print("\n" + "=" * 65)
print("SUCCESS! All 5 research models trained and checkpoints saved.")
print("Download '/kaggle/working/thesis_results.zip' from Output on the right!")
print("=" * 65)
```

---

## Step 3: Download `thesis_results.zip`
1. When execution finishes, look at the right-side panel under **Output** (`/kaggle/working/`).
2. Refresh the output directory if needed.
3. Click the three dots (`...`) next to `thesis_results.zip` and click **Download**.

---

## Step 4: Resume Here on Your Local PC
1. Extract `thesis_results.zip` on your computer.
2. Copy the extracted folders (`models/`, `outputs/`, `data/processed/`) into your workspace:
   * Copy `models/*` into `e:\thesis\models\`
   * Copy `outputs/*` into `e:\thesis\outputs\`
3. Come back to this chat and say **"I have pasted the files"** or **"resume"**!
   * We will immediately verify the model checkpoints and show you all the generated performance curves, confusion matrices, and Grad-CAM visualizations.
