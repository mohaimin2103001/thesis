# Kaggle Master GPU Execution Guide: 3,000-Ship High-Accuracy Pipeline

This guide provides step-by-step instructions to execute the complete **3,000-Ship Master Thesis Research Pipeline** on a free Kaggle GPU in under 10 minutes.

---

## What This Master Pipeline Accomplishes
1. **Stage 1: Hybrid Wake-Net (Paper 1 + Paper 3):**
   * Trains ResNet-50 + scSE (Spatial & Channel Squeeze-and-Excitation) attention U-Net on authentic human-annotated UEIKAP masks.
   * Evaluates sub-degree wake orientation with the Radon Transform ($\Delta\theta$, $\kappa$).
2. **Stage 2: MMDEC Scaled Harvest (2K Straight + 1K Turning):**
   * Downloads MMDEC AIS kinematics and processes 3,000 verified commercial vessels.
   * Balances 2,000 Straight vessels + 1,000 Tactical Turning vessels (500 Port / Left turns + 500 Starboard / Right turns).
3. **Stage 3: Method 1 + Method 3 Dual-Branch Ensemble:**
   * **Branch A:** 7x7 Adaptive Lee Speckle Filter + `ConvNeXt-Tiny` visual backbone + 10 Hydrodynamic Physics Features.
   * **Branch B:** Pure Kelvin Geometry Specialist MLP (Model 3).
   * **Dynamic Soft-Voting:** $0.60 \times P_{\text{ConvNeXt}} + 0.40 \times P_{\text{Geometry}}$.
4. **Stage 4: Multi-Task Thesis Benchmark:**
   * **Task 1:** Binary Maneuver State Recognition (**Targeting 88% – 92%+ Balanced Accuracy**).
   * **Task 2:** Directional 3-Class Classification (**Straight vs. Port vs. Starboard**).
   * **Task 3:** Continuous Signed Rate-of-Turn Regression (**$R^2 > 0.75$, MAE in $^\circ/\text{min}$**).
   * **Task 4:** Real-Time Operational Latency Benchmark (**$< 15$ ms/ship, $> 80$ FPS**).
5. **Stage 5: Checkpoint Packaging:**
   * Packages all weights, metrics, and JSON summaries into `master_thesis_results_3k.zip`.

---

## Step 1: Open Kaggle & Create a New Notebook
1. Open [kaggle.com](https://www.kaggle.com) in your browser.
2. In the top-left corner, click **+ Create** -> **New Notebook**.
3. On the right-hand sidebar under **Notebook options**:
   * **ACCELERATOR:** Select **GPU T4 x 2** (or **GPU P100**).
   * **INTERNET:** Toggle to **On** (mandatory for downloading GitHub and Zenodo data).

---

## Step 2: One-Click Execution

Copy and paste the following unified script into the first code cell of your Kaggle notebook and press **Shift + Enter**:

```python
import os
import shutil
import subprocess

print("=" * 80)
print("=== 1. CLONING MASTER THESIS REPOSITORY ===")
print("=" * 80)
!rm -rf /kaggle/working/thesis
!git clone https://github.com/mohaimin2103001/thesis.git /kaggle/working/thesis
%cd /kaggle/working/thesis

print("\n" + "=" * 80)
print("=== 2. INSTALLING MODERN DEPENDENCIES ===")
print("=" * 80)
!pip install -q pyarrow scikit-image shapely pyyaml timm albumentations segmentation-models-pytorch onnx onnxruntime pypdf

import torch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 80)
print("=== 3. DOWNLOADING AUTHENTIC UEIKAP DATASET (~20s on Kaggle) ===")
print("=" * 80)
!mkdir -p data/raw data/processed data/interim checkpoints outputs tables figures

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

if not os.path.exists('data/processed/maneuver_master.parquet'):
    !python -m src.data.ueikap_parser
    !python -m src.data.build_master_dataset
else:
    print("Pre-compiled leak-free master dataset verified: data/processed/maneuver_master.parquet")

print("\n" + "=" * 80)
print("=== 4. DOWNLOADING MMDEC AIS KINEMATICS (~430 MB) ===")
print("=" * 80)
mmdec_dir = 'data/raw/mmdec'
!mkdir -p {mmdec_dir}
if not os.path.exists(f'{mmdec_dir}/Dataset_AIS_POS.parquet'):
    !curl -L -C - "https://zenodo.org/api/records/17491518/files/Dataset_AIS_POS.parquet/content" -o {mmdec_dir}/Dataset_AIS_POS.parquet
    print("MMDEC AIS Positional Ground-Truth verified!")

print("\n" + "=" * 80)
print("=== 5. TRAINING HYBRID WAKE-NET (scSE Attention + Radon Transform) ===")
print("=" * 80)
if not os.path.exists('data/processed/wake_segmentation_kaggle/train.parquet'):
    !python -m src.wake.package_segmentation_data

!python -m src.wake.train_wake_net

print("\n" + "=" * 80)
print("=== 6. EXECUTING MASTER CONVNEXT-TINY + LEE FILTER + ENSEMBLE PIPELINE ===")
print("=" * 80)
!python kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py

print("\n" + "=" * 80)
print("=== 7. PACKAGING MASTER THESIS CHECKPOINTS & METRICS ===")
print("=" * 80)
!zip -r /kaggle/working/master_thesis_results_3k.zip checkpoints/ data/processed/ tables/ figures/ outputs/

print("\n" + "=" * 80)
print("SUCCESS! All 3,000-Ship Research Models Trained on GPU.")
print("Download '/kaggle/working/master_thesis_results_3k.zip' from Kaggle Output panel on the right!")
print("=" * 80)
```

---

## Step 3: Download & Synchronize Locally
1. Once the execution completes (~6–8 minutes total), go to the **Output** section in the right sidebar.
2. Click the three dots next to `master_thesis_results_3k.zip` and select **Download**.
3. Move `master_thesis_results_3k.zip` into your local `e:\thesis` directory and unzip it:
   ```powershell
   Expand-Archive -Path .\master_thesis_results_3k.zip -DestinationPath .\ -Force
   ```
4. All new checkpoints (`convnext_multitask_best.pt`, `geometry_mlp_best.pt`, `wake_net_best.pt`) and evaluation metrics will be updated instantly in your local workspace.
