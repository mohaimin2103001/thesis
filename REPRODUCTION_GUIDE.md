# Complete Reproduction & Deployment Workflow Guideline
## How to Replicate This Research From Scratch on Any Fresh PC (Windows, Linux, macOS, Cloud GPU)

This guide provides an exhaustive, step-by-step, copy-paste protocol to reproduce the entire **Satellite SAR Ship Wake Analysis for Maritime Maneuver Recognition** system on any computer from zero.

---

## Table of Contents
1. [System Prerequisites & Hardware Sizing](#1-system-prerequisites--hardware-sizing)
2. [Step 1: Clone Repository & Environment Setup](#2-step-1-clone-repository--environment-setup)
3. [Step 2: Authentic Dataset Acquisition & Integrity Check](#3-step-2-authentic-dataset-acquisition--integrity-check)
4. [Step 3: Ground-Truth Kinematics Derivation (AIS Telemetry)](#4-step-3-ground-truth-kinematics-derivation-ais-telemetry)
5. [Step 4: Exploratory Data Analysis & Radar Dynamic Range Audit](#5-step-4-exploratory-data-analysis--radar-dynamic-range-audit)
6. [Step 5: Train Deep Convolutional Backbones](#6-step-5-train-deep-convolutional-backbones)
7. [Step 6: Dual-Architecture Ensemble & 4-Fold TTA Benchmark](#7-step-6-dual-architecture-ensemble--4-fold-tta-benchmark)
8. [Step 7: Epistemic Uncertainty Quantification (MC-Dropout)](#8-step-7-epistemic-uncertainty-quantification-mc-dropout)
9. [Step 8: Grad-CAM Hydrodynamic Explainability](#9-step-8-grad-cam-hydrodynamic-explainability)
10. [Step 9: Meteo-Marine Operational Domain Shift Stress-Testing](#10-step-9-meteo-marine-operational-domain-shift-stress-testing)
11. [Step 10: Qualitative Hydrodynamic Failure Analysis](#11-step-10-qualitative-hydrodynamic-failure-analysis)
12. [One-Command Master Automated Pipeline Execution](#12-one-command-master-automated-pipeline-execution)
13. [Output Verification Checklist & Benchmarks](#13-output-verification-checklist--benchmarks)
14. [Cross-Platform Troubleshooting & FAQs](#14-cross-platform-troubleshooting--faqs)

---

## 1. System Prerequisites & Hardware Sizing

### Hardware Requirements:
* **Storage:** Minimum **10 GB** free disk space (to store the 1.46 GB ZIP archive, extracted images, checkpoints, and figures).
* **RAM:** Minimum **8 GB** system RAM (16 GB recommended).
* **Compute:** 
  * **Option A (GPU - Recommended):** NVIDIA GeForce RTX (3060, 3070, 4060, 4080, A100, etc.) with $\ge 4\text{ GB}$ VRAM.
  * **Option B (CPU Mode):** Multi-core CPU (Intel Core i5/i7/i9 or AMD Ryzen). Training takes ~10–15 minutes per model on modern CPUs with multi-threading.
  * **Option C (Apple Silicon):** M1/M2/M3/M4 chips supported via PyTorch Metal Performance Shaders (`mps`).

### Software Prerequisites:
* **Operating System:** Windows 10/11, Ubuntu 20.04/22.04/24.04 LTS, or macOS 12+.
* **Python:** Python **3.10**, **3.11**, or **3.12** (Python 3.10–3.12 recommended for widest pre-compiled wheel compatibility).
* **Git:** Installed on system PATH (`git --version`).
* **Curl / PowerShell / Wget:** Standard command-line downloading utility.

---

## 2. Step 1: Clone Repository & Environment Setup

Open your terminal (PowerShell on Windows, or Bash on Linux/macOS) and execute:

### 2.1 Clone Repository
```bash
git clone https://github.com/mohaimin2103001/thesis.git
cd thesis
```

### 2.2 Create & Activate Python Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```
* **On Linux / macOS (Bash):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 2.3 Upgrade Pip & Install Dependencies
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

*(Optional - If running on NVIDIA GPU, verify PyTorch CUDA acceleration):*
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Step 2: Authentic Dataset Acquisition & Integrity Check

The research utilizes the official **UEIKAP** European Space Agency Sentinel-1 SAR benchmark. Due to size constraints, raw data is excluded from git and must be downloaded to `data/raw/`.

### 3.1 Create Data Directories
* **On Windows (PowerShell):**
  ```powershell
  New-Item -ItemType Directory -Force -Path "data\raw", "data\processed", "models", "outputs\figures", "outputs\metrics", "outputs\gradcam"
  ```
* **On Linux / macOS:**
  ```bash
  mkdir -p data/raw data/processed models outputs/figures outputs/metrics outputs/gradcam
  ```

### 3.2 Download the Official Archive (`UEIKAP_SAR.zip` - 1.46 GB)

You can download using either the automated Python script, `curl`, or direct browser download:

#### Method A: Using Built-In Python Script (Recommended)
```bash
python -m src.download_ueikap
```

#### Method B: Using `curl` (Resumable Single-Stream)
```bash
# Windows / Linux / macOS:
curl -L -C - "https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1" -o data/raw/UEIKAP_SAR.zip
```

#### Method C: Manual Browser Download
Download directly from:  
👉 **[https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1](https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1)**  
Save the file as `data/raw/UEIKAP_SAR.zip`.

### 3.3 Verify Integrity via MD5 Checksum
The published official Zenodo MD5 checksum is: **`356148359a9ae6b7ae3d3cec2fb61304`**.

* **On Windows (PowerShell):**
  ```powershell
  Get-FileHash -Algorithm MD5 data\raw\UEIKAP_SAR.zip
  ```
* **On Linux:**
  ```bash
  md5sum data/raw/UEIKAP_SAR.zip
  ```
* **On macOS:**
  ```bash
  md5 data/raw/UEIKAP_SAR.zip
  ```
Ensure the output matches `356148359a9ae6b7ae3d3cec2fb61304`.

### 3.4 Extract Archive Contents
* **On Windows (PowerShell):**
  ```powershell
  Expand-Archive -Path data\raw\UEIKAP_SAR.zip -DestinationPath data\raw\UEIKAP_SAR -Force
  ```
* **On Linux / macOS:**
  ```bash
  unzip -q data/raw/UEIKAP_SAR.zip -d data/raw/UEIKAP_SAR
  ```

Verify that `data/raw/UEIKAP_SAR/images/` contains 515 chips across `train/`, `val/`, and `test/`, and `data/raw/UEIKAP_SAR/AIS_interpolated_SAR_UEIKAP/` contains 522 AIS CSV files.

---

## 4. Step 3: Ground-Truth Kinematics Derivation (AIS Telemetry)

Parse raw AIS transponder broadcasts, synchronize vessel kinematics with satellite acquisition timestamps, compute Rate of Turn ($ROT = \frac{d\psi}{dt}$) and circular drift ($\Delta\theta$), prune ambiguous deadband cases, and create the master index:

```bash
python -m src.label_generator
```

* **Output Artifact:** [`data/processed/dataset_index.csv`](data/processed/dataset_index.csv)
* **Expected Result:** 464 clean samples indexed (359 Straight, 105 Turning). Exact splits: 277 Train, 93 Validation, 94 Test.

---

## 5. Step 4: Exploratory Data Analysis & Radar Dynamic Range Audit

Audit raw radar dynamic range compression vs. engineered polarimetric CLAHE enhancement:

```bash
python -m src.eda_audit
```

* **Output Artifact:** [`outputs/figures/preprocessing_comparison.png`](outputs/figures/preprocessing_comparison.png)
* **Expected Verification:** Confirms raw mean backscatter of ~2.5 DN is successfully lifted to ~75.3 DN across the full 8-bit dynamic range.

---

## 6. Step 5: Train Deep Convolutional Backbones

Train all models from scratch with automatic mixed precision (AMP) and class-weighted Focal Loss ($\gamma = 2.0$, $\alpha = [0.25, 0.75]$):

### 6.1 Model 1: ResNet-18 Binary Classifier (Baseline High-Specificity)
```bash
python -m src.train --arch resnet18 --mode binary --epochs 15
```
* **Saved Checkpoint:** `models/best_resnet18_binary_model.pt`
* **Saved Curves:** `outputs/figures/resnet18_binary_training_curves.png`
* **Saved Metrics:** `outputs/metrics/resnet18_binary_test_metrics.json`
* **Benchmark Test Acc:** **56.38%** (Macro F1: **0.4731**)

### 6.2 Model 2: ResNet-34 Polarimetric Model (High-Sensitivity)
```bash
python -m src.train --arch resnet34 --mode binary --epochs 15
```
* **Saved Checkpoint:** `models/best_resnet34_binary_model.pt`
* **Saved Curves:** `outputs/figures/resnet34_binary_training_curves.png`
* **Saved Metrics:** `outputs/metrics/resnet34_binary_test_metrics.json`
* **Validation F1 Peak:** **0.6773**

### 6.3 Model 3: ResNet-18 Directional 3-Class Model (Straight vs. Port vs. Starboard)
```bash
python -m src.train --arch resnet18 --mode 3class --epochs 15
```
* **Saved Checkpoint:** `models/best_resnet18_3class_model.pt`
* **Saved Curves:** `outputs/figures/resnet18_3class_training_curves.png`
* **Saved Metrics:** `outputs/metrics/resnet18_3class_test_metrics.json`
* **Benchmark Test Acc:** **73.40%** (Macro F1: **0.3184**)

---

## 7. Step 6: Dual-Architecture Ensemble & 4-Fold TTA Benchmark

Combine the high-specificity ResNet-18 ($w=0.45$) with the high-sensitivity ResNet-34 ($w=0.55$) using 4-fold geometric Test-Time Augmentation (TTA: original, horizontal flip, vertical flip, transpose):

```bash
python -m src.ensemble_eval
```

* **Output Artifacts:** 
  * Confusion Matrix: [`outputs/figures/ensemble_confusion_matrix.png`](outputs/figures/ensemble_confusion_matrix.png)
  * Metrics JSON: [`outputs/metrics/dual_ensemble_test_metrics.json`](outputs/metrics/dual_ensemble_test_metrics.json)
* **Benchmark Performance:**
  * **Test Accuracy:** **65.96% – 77.66%** (+9.58% to +22.34% over baseline)
  * **Specificity (True Negative Rate):** **73.42% – 84.80%**
  * **Macro F1-Score:** **0.4919 – 0.6141**

---

## 8. Step 7: Epistemic Uncertainty Quantification (MC-Dropout)

Execute $N=30$ stochastic forward passes with dropout ($p=0.3$) kept active during inference to evaluate predictive entropy $\mathcal{H}(y|\mathbf{x})$:

```bash
python -m src.uncertainty --samples 30
```

* **Output Artifacts:**
  * Selective Rejection Curve: [`outputs/figures/resnet18_uncertainty_rejection_curve.png`](outputs/figures/resnet18_uncertainty_rejection_curve.png)
  * Calibration Metrics: [`outputs/metrics/resnet18_uncertainty_results.json`](outputs/metrics/resnet18_uncertainty_results.json)
* **Expected Verification:** Misclassified samples exhibit higher entropy than correct predictions, proving that uncertain cases can be safely rejected for human operator triage.

---

## 9. Step 8: Grad-CAM Hydrodynamic Explainability

Compute gradients of the predicted class score with respect to `layer4` convolutional activation maps to visualize model attention:

```bash
python -m src.explainability
```

* **Output Artifacts:** 20 high-resolution visual attribution heatmaps saved in [`outputs/gradcam/`](outputs/gradcam/).
* **Expected Verification:** Visual proof that activations localize along the theoretical $19.47^\circ$ Kelvin arms and curved rudder vortices rather than background speckle noise.

---

## 10. Step 9: Meteo-Marine Operational Domain Shift Stress-Testing

Stress-test model resilience under severe simulated ocean conditions (rough sea Rayleigh speckle clutter, calm sea specular backscatter collapse, and wide-swath spatial resolution downsampling) across 5 severity levels:

```bash
python -m src.domain_shift
```

* **Output Artifacts:**
  * Domain Shift Curves: [`outputs/figures/resnet18_domain_shift_curves.png`](outputs/figures/resnet18_domain_shift_curves.png)
  * Degradation Metrics: [`outputs/metrics/resnet18_domain_shift_results.json`](outputs/metrics/resnet18_domain_shift_results.json)

---

## 11. Step 10: Qualitative Hydrodynamic Failure Analysis

Classify all test set errors into oceanographic physical failure mechanisms:

```bash
python -m src.failure_analysis
```

* **Output Artifacts:**
  * Visual Error Gallery: [`outputs/figures/resnet18_failure_gallery.png`](outputs/figures/resnet18_failure_gallery.png)
  * Systematic JSON Taxonomy: [`outputs/metrics/resnet18_failure_analysis.json`](outputs/metrics/resnet18_failure_analysis.json)
* **Failure Taxonomy Breakdown:**
  1. *Wake Dissipation / Ambiguous Temporal Latency* (~90.9%)
  2. *Sub-Critical Froude Number ($U < 6$ kts; Kelvin wake collapsed)* (~5.5%)
  3. *Pure Drift without Rudder Deflection* (~3.6%)

---

## 12. One-Command Master Automated Pipeline Execution

If you prefer to execute the entire end-to-end research workflow automatically in one command:

```bash
python -m src.pipeline_runner
```

This sequentially runs:
1. `label_generator`
2. `train` (ResNet-18 Binary)
3. `train` (ResNet-34 Polarimetric)
4. `train` (ResNet-18 3-Class)
5. `ensemble_eval`
6. `uncertainty`
7. `explainability`
8. `domain_shift`
9. `failure_analysis`

All figures, metrics, and models will be populated automatically.

---

## 13. Output Verification Checklist & Benchmarks

Once replication finishes, verify that the following files are present:

### 13.1 Generated Plots & Visualizations (`outputs/figures/`):
- [x] `ensemble_confusion_matrix.png` — Dual ensemble confusion matrix
- [x] `preprocessing_comparison.png` — Raw vs CLAHE enhanced SAR chips
- [x] `resnet18_binary_training_curves.png` — ResNet-18 loss and F1 convergence
- [x] `resnet34_binary_training_curves.png` — ResNet-34 polarimetric convergence
- [x] `resnet18_3class_training_curves.png` — 3-Class directional convergence
- [x] `resnet18_uncertainty_rejection_curve.png` — Selective error rejection curve
- [x] `resnet18_domain_shift_curves.png` — Meteo-marine degradation curves
- [x] `resnet18_failure_gallery.png` — Systematic physical failure gallery

### 13.2 Machine-Readable Metrics (`outputs/metrics/`):
- [x] `dual_ensemble_test_metrics.json`
- [x] `resnet18_binary_test_metrics.json`
- [x] `resnet34_binary_test_metrics.json`
- [x] `resnet18_3class_test_metrics.json`
- [x] `resnet18_uncertainty_results.json`
- [x] `resnet18_domain_shift_results.json`
- [x] `resnet18_failure_analysis.json`

### 13.3 Trained Checkpoints (`models/`):
- [x] `best_resnet18_binary_model.pt`
- [x] `best_resnet34_binary_model.pt`
- [x] `best_resnet18_3class_model.pt`

---

## 14. Cross-Platform Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'cv2'` or `libGL.so.1` on Linux / Ubuntu
* **Cause:** Missing OpenGL dependencies on headless Linux systems.
* **Fix:**
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  # Or install the headless version of OpenCV:
  pip install opencv-python-headless
  ```

### Q2: CUDA Out-Of-Memory (OOM) on smaller GPUs (e.g., 4GB VRAM)
* **Cause:** Large batch sizes with high-resolution $384 \times 384$ chips.
* **Fix:** Add `--batch-size 8` to the training commands:
  ```bash
  python -m src.train --arch resnet18 --mode binary --epochs 15 --batch-size 8
  ```

### Q3: Zenodo Download Interruption or Slow Speeds
* **Cause:** Transient network timeouts with European research servers.
* **Fix:** The provided `curl.exe -L -C -` command supports automatic resumption. Re-running the command resumes from the exact byte where it paused.

### Q4: Windows PowerShell script execution policy error
* **Cause:** Default PowerShell execution policy blocks `.venv\Scripts\Activate.ps1`.
* **Fix:** Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` before activating the environment.

---

### You are now ready to run and reproduce this research on any machine!
For in-depth defense examiner questions, formulas, and journal citations, refer to [FULL_THESIS_PROJECT_REPORT.md](FULL_THESIS_PROJECT_REPORT.md).
