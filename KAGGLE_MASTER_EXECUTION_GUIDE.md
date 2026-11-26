# Kaggle Master GPU Execution Guide: Complete End-to-End Pipeline

This guide gives you the exact, zero-error steps to run the entire thesis pipeline on a free Kaggle GPU (**Wake-Net scSE + Radon Transform -> MMDEC Full Ship Harvest & Auto-Masking -> ConvNeXt-Tiny + 7x7 Lee Filter + Geometry MLP Soft-Voting Ensemble -> Binary + 3-Class Directional + Signed ROT**).

---

## Step 1: Open Kaggle & Configure GPU + Storage
1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Click **+ Create** -> **New Notebook**.
3. On the right sidebar under **Notebook options**:
   * **ACCELERATOR:** Select **GPU T4 x 2** (or **GPU P100**).
   * **INTERNET:** Toggle to **On** (required to pull GitHub + Zenodo datasets).

---

## Step 2: Run the Complete One-Click Master Cell

Paste the code below into a single Kaggle notebook cell and click **Run** (or press **Shift + Enter**):

> **Optional Harvest Size:** By default, `--max_straight 4000 --max_turning -1` harvests up to **4,000 straight ships + ALL qualified turning ships**. Change `--max_straight -1` if you want **every single qualified straight ship** in MMDEC!

```python
import os
import shutil

print("=" * 80)
print("=== 1. CLONING MASTER THESIS REPOSITORY ===")
print("=" * 80)
if os.path.exists('/kaggle/working/thesis/.git'):
    !git -C /kaggle/working/thesis fetch origin && !git -C /kaggle/working/thesis reset --hard origin/main
else:
    !git clone https://github.com/mohaimin2103001/thesis.git /kaggle/working/thesis
%cd /kaggle/working/thesis

print("\n=== 2. INSTALLING DEPENDENCIES ===")
!pip install -q pyarrow scikit-image shapely pyyaml timm albumentations segmentation-models-pytorch onnx onnxruntime pypdf tqdm

import torch
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 80)
print("=== 3. DOWNLOADING UEIKAP + MMDEC DATASETS IN KAGGLE CLOUD ===")
print("=" * 80)
!mkdir -p data/raw/mmdec data/processed data/interim checkpoints outputs tables figures logs

import subprocess
def run_stage(cmd):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    subprocess.run(cmd, env=env, check=True)

run_stage(["python", "-u", "-m", "src.data.kaggle_download_helper"])

if not os.path.exists('data/processed/wake_segmentation_kaggle/images') or len(os.listdir('data/processed/wake_segmentation_kaggle/images')) == 0:
    run_stage(["python", "-u", "-m", "src.data.ueikap_parser"])
    run_stage(["python", "-u", "-m", "src.data.build_master_dataset"])
    run_stage(["python", "-u", "-m", "src.wake.package_segmentation_data"])

print("\n" + "=" * 80)
print("=== 4. STAGE 1: TRAINING HYBRID WAKE-NET (ResNet-50 + scSE + Radon) ===")
print("=" * 80)
if not os.path.exists('checkpoints/wake_net/wake_net_best.pt'):
    run_stage(["python", "-u", "-m", "src.wake.train_wake_net", "--epochs", "35", "--batch_size", "8", "--lr", "2e-4"])
else:
    print("Using existing checkpoints/wake_net/wake_net_best.pt already trained!")

print("\n" + "=" * 80)
print("=== 5. STAGE 2: MMDEC HARVESTER + WAKE-NET AUTO-MASKING + RADON ===")
print("=" * 80)
# Set --max_straight -1 if you want EVERY single straight ship in MMDEC
run_stage(["python", "-u", "-m", "src.data.mmdec_extract_and_mask", "--min_sog", "3.5", "--min_turn_rot", "1.0", "--max_straight", "4000", "--max_turning", "-1", "--batch_size", "32"])

print("\n" + "=" * 80)
print("=== 6. STAGE 3: TRAINING DUAL-BRANCH CONVNEXT + LEE + GEOMETRY ENSEMBLE ===")
print("=" * 80)
run_stage(["python", "-u", "kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py", "--epochs", "35", "--batch_size", "24", "--lr", "2e-4", "--img_size", "320", "--gradcam_per_class", "-1"])

print("\n" + "=" * 80)
print("=== 7. PACKAGING WEIGHTS, METRICS & GRAD-CAM VISUALS ===")
print("=" * 80)
!zip -r /kaggle/working/master_thesis_results_3k.zip checkpoints/ logs/ tables/ figures/ outputs/ data/processed/*.parquet data/processed/*.json data/processed/*.csv

print("\nSUCCESS! Download '/kaggle/working/master_thesis_results_3k.zip' from the Output panel on the right.")
```

---

## Step 3: Download & Sync Back Locally
1. Once finished (~15–20 minutes total including the 10.5 GB cloud download, cropping, Wake-Net, and ConvNeXt training), open **Output** in the Kaggle sidebar.
2. Download `master_thesis_results_3k.zip`.
3. Extract it inside `e:\thesis`:
   ```powershell
   Expand-Archive -Path .\master_thesis_results_3k.zip -DestinationPath .\ -Force
   ```
4. Read `checkpoints/convnext_ensemble/master_evaluation_metrics.json` and `logs/wake_net_evaluation.json` for your final thesis numbers.
