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

# A. UEIKAP Benchmark (~100 MB)
if not os.path.exists('data/raw/UEIKAP_SAR.zip'):
    !curl -L -C - "https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1" -o data/raw/UEIKAP_SAR.zip
    !unzip -q -o data/raw/UEIKAP_SAR.zip -d data/raw/UEIKAP_SAR

raw_dir = 'data/raw'
tgt_dir = os.path.join(raw_dir, 'UEIKAP_SAR')
for root, dirs, files in os.walk(raw_dir):
    if 'images' in dirs and root != tgt_dir:
        for d in dirs:
            src_p = os.path.join(root, d)
            dst_p = os.path.join(tgt_dir, d)
            if not os.path.exists(dst_p):
                shutil.move(src_p, dst_p)

!python -m src.data.ueikap_parser
!python -m src.data.build_master_dataset
!python -m src.wake.package_segmentation_data

# B. MMDEC AIS (430 MB) + Corners (5 MB) + SAR Chunks Zip (10.5 GB - streamed directly!)
mmdec_dir = 'data/raw/mmdec'
if not os.path.exists(f'{mmdec_dir}/Dataset_AIS_POS.parquet'):
    !curl -L -C - "https://zenodo.org/api/records/17491518/files/Dataset_AIS_POS.parquet/content" -o {mmdec_dir}/Dataset_AIS_POS.parquet

if not os.path.exists(f'{mmdec_dir}/Dataset_S1_chunks_corners.zip'):
    !curl -L -C - "https://zenodo.org/api/records/17491518/files/Dataset_S1_chunks_corners.zip/content" -o {mmdec_dir}/Dataset_S1_chunks_corners.zip

if not os.path.exists(f'{mmdec_dir}/Dataset_S1_chunks.zip'):
    print("Downloading MMDEC Dataset_S1_chunks.zip (~10.5 GB, streamed from zip without unzipping)...")
    !curl -L -C - "https://zenodo.org/api/records/17491518/files/Dataset_S1_chunks.zip/content" -o {mmdec_dir}/Dataset_S1_chunks.zip

print("\n" + "=" * 80)
print("=== 4. STAGE 1: TRAINING HYBRID WAKE-NET (ResNet-50 + scSE + Radon) ===")
print("=" * 80)
!python -m src.wake.train_wake_net --epochs 35 --batch_size 8 --lr 3e-4

print("\n" + "=" * 80)
print("=== 5. STAGE 2: MMDEC HARVESTER + WAKE-NET AUTO-MASKING + RADON ===")
print("=" * 80)
# Set --max_straight -1 if you want EVERY single straight ship in MMDEC
!python -m src.data.mmdec_extract_and_mask --min_sog 3.5 --min_turn_rot 1.5 --max_straight 4000 --max_turning -1 --batch_size 32

print("\n" + "=" * 80)
print("=== 6. STAGE 3: TRAINING DUAL-BRANCH CONVNEXT + LEE + GEOMETRY ENSEMBLE ===")
print("=" * 80)
!python kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py --epochs 25 --batch_size 32 --lr 2e-4 --gradcam_per_class -1

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
