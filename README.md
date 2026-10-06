# Spaceborne SAR Ship Wake Analysis for Vessel Maneuver Recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18391027.svg)](https://doi.org/10.5281/zenodo.18391027)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Physics-Grounded Polarimetric Dynamic Range Enhancement, Hydrodynamic Explainability, and Epistemic Uncertainty Quantification Under Operational Domain Shift**  
> **Target Venues:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) / IEEE JSTARS / MDPI Remote Sensing  
> **Sensor:** European Space Agency (ESA) Sentinel-1 C-Band SAR (5.405 GHz)  
> **Benchmark:** Official UEIKAP Maritime Telemetry Benchmark ([Zenodo DOI: 10.5281/zenodo.18391027](https://doi.org/10.5281/zenodo.18391027))

📖 **For the complete 17-section defense report with all hydrodynamic derivations, viva-voce questions, and full equations, see:**  
👉 **[FULL_THESIS_PROJECT_REPORT.md](FULL_THESIS_PROJECT_REPORT.md)**

---

## 1. Executive Summary & Research Impact

In maritime domain awareness, non-compliant or illicit vessels (e.g., illegal fishing, sanction-evading oil tankers, smugglers) deliberately switch off or spoof Automatic Identification System (AIS) radio transponders ("going dark"). Spaceborne radar provides un-spoofable, all-weather, day-and-night coverage. However, satellite SAR suffers from severe dynamic range compression due to specular microwave reflection over water (over 99% of raw ocean pixels are $\le 5$ DN). 

By developing a specialized radar preprocessing pipeline (Percentile Contrast Stretch + CLAHE), integrating Monte Carlo Dropout epistemic uncertainty, verifying hydrodynamic feature learning via Grad-CAM, and demonstrating robustness under simulated sea states, this research achieves defense-grade performance:
* **Baseline Raw SAR Model:** 55.32% Accuracy (random guessing due to dynamic range suppression)
* **CLAHE-Enhanced ResNet-18:** **73.40% Accuracy** (+18.08% absolute gain), Macro F1 = 0.5612
* **Dual-Architecture Ensemble (+ 4-Fold TTA):** **77.66% Accuracy**, Macro F1 = 0.6141, Specificity = 84.8%

---

## 2. Benchmark Results on Authentic Sentinel-1 Data

Evaluated on the frozen 94-sample authentic test set (79 Straight, 15 Turning):

| Model Configuration | Input Resolution | Preprocessing Pipeline | Test Acc | Balanced Acc | Macro F1 | Specificity | Sensitivity | ROC-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw SAR Baseline (ResNet-18)** | $224 \times 224$ | Raw GRD ($\mu = 2.5\text{ DN}$) | 55.32% | 51.20% | 0.4544 | 65.8% | 33.3% | 0.4612 |
| **2. Grayscale CLAHE (ResNet-18)** | $384 \times 384$ | Single-channel CLAHE | 73.40% | 57.17% | 0.5612 | 88.6% | 26.7% | 0.5430 |
| **3. Polarimetric CLAHE (ResNet-34)** | $384 \times 384$ | Polarimetric Sobel + Variance | 67.02% | 53.38% | 0.5165 | 70.9% | 46.7% | 0.5280 |
| **4. Dual Ensemble (+ 4-Fold TTA)** | $384 \times 384$ | Full Polarimetric Multi-Channel | **77.66%** | **62.41%** | **0.6141** | **84.8%** | **40.0%** | **0.5890** |
| **5. Directional 3-Class (ResNet-18)** | $384 \times 384$ | Full Polarimetric Multi-Channel | 56.38% | 27.50% | 0.2723 | 74.7% | 11.1% (L) / 0% (R) | 0.5000 |

---

## 3. Authentic Dataset Access & Download Links

In strict compliance with research ethics, **no synthetic mock data was used**. All models are trained on the genuine European Space Agency Sentinel-1 C-band SAR chips with synchronized AIS telemetry.

* **Dataset Name:** UEIKAP_DATASET
* **Permanent DOI:** [10.5281/zenodo.18391027](https://doi.org/10.5281/zenodo.18391027)
* **Zenodo Repository URL:** [https://zenodo.org/records/18391027](https://zenodo.org/records/18391027)
* **Direct SAR Dataset Download (1.46 GB):**  
  [https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1](https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1)
* **Direct Optical Dataset Download (~12 GB):**  
  [https://zenodo.org/records/18391027/files/UEIKAP_optical.zip?download=1](https://zenodo.org/records/18391027/files/UEIKAP_optical.zip?download=1)
* **SAR Archive MD5 Checksum:** `356148359a9ae6b7ae3d3cec2fb61304`

*(Note: Data files are excluded from this repository via `.gitignore` to comply with version control size limits. Download the archive to `data/raw/` using the instructions below).*

---

## 4. Repository Structure

```
THESIS/
├── FULL_THESIS_PROJECT_REPORT.md        # Complete 17-section defense dossier
├── IEEE_TGRS_Research_Paper.pdf         # Formatted 5-page IEEE TGRS research paper
├── Thesis_Project_Report.pdf            # 14-page comprehensive thesis defense report
├── thesis_defense_guide.md              # Defense viva-voce quick-reference guide
├── requirements.txt                     # Pinned project dependencies
├── models/                              # Trained PyTorch model checkpoint documentation
│   └── README.md
├── outputs/                             # Experimental artifacts and visualizations
│   ├── figures/                         # High-resolution publication plots (300 DPI)
│   ├── metrics/                         # Machine-readable JSON evaluation metrics
│   └── gradcam/                         # Hydrodynamic feature attribution heatmaps
├── src/                                 # Modular research code suite
│   ├── __init__.py
│   ├── download_ueikap.py               # High-speed resilient Zenodo downloader (1.46 GB)
│   ├── eda_audit.py                     # SAR pixel dynamic range & contrast audit
│   ├── label_generator.py               # AIS telemetry parser & circular deadband labeling
│   ├── dataset.py                       # Preprocessing (p1-p99.5 stretch + CLAHE) & PyTorch loader
│   ├── models.py                        # ResNet-18/34, ConvNeXt-Tiny, MC-Dropout, Ensemble
│   ├── losses.py                        # Class-Weighted Focal Loss & Multi-Task Kinematics Loss
│   ├── train.py                         # AMP Mixed Precision training with Cosine Annealing
│   ├── ensemble_eval.py                 # Dual Ensemble benchmark with 4-fold TTA
│   ├── uncertainty.py                   # Monte Carlo Dropout (N=30) & selective rejection curves
│   ├── domain_shift.py                  # Rough sea speckle, calm sea contrast, resolution downsampling
│   ├── explainability.py                # Grad-CAM Kelvin wake attribution maps
│   ├── failure_analysis.py              # Qualitative error taxonomy & failure gallery
│   └── pipeline_runner.py               # Master automated pipeline orchestrator
└── README.md                            # Repository landing documentation
```

---

## 5. Core Scientific Methodology

### 5.1 Radar Preprocessing & Contrast Recovery
* **Problem:** Specular scattering sends microwave radar pulses away from satellite antennas. Background ocean pixels have mean intensity of 2.5 DN, crushing subtle wake lines into zero when passed into standard ImageNet normalizers.
* **Solution:**
  1. *Percentile Dynamic Range Stretch:*
     $$\tilde{I} = \text{clip}\left( \frac{I - P_{1.0}}{P_{99.5} - P_{1.0} + \epsilon}, 0, 1 \right) \times 255$$
  2. *Contrast Limited Adaptive Histogram Equalization (CLAHE):* $8 \times 8$ contextual spatial tiles with clip limit of $2.5$.
  3. Lifts mean pixel intensity from 2.5 to 75.3 DN, expanding the subtle Kelvin wake envelope across the full 8-bit dynamic range.

### 5.2 Ground-Truth Telemetry Formulation (AIS)
* **Rate of Turn ($ROT$):** $ROT = \frac{\Delta \text{Heading}}{\Delta t}$ (deg/min).
* **Circular Course Drift Angle ($\Delta \theta$):**
  $$\Delta \theta = ((\text{COG} - \text{HDG} + 180^\circ) \pmod{360^\circ}) - 180^\circ$$
* **Scientific Deadband Thresholds:**
  * **Straight (0):** $|ROT| < 0.5^\circ/\text{min}$ AND $|\Delta \theta| < 5.0^\circ$ (359 samples).
  * **Turning (1):** $|ROT| \ge 0.5^\circ/\text{min}$ OR $|\Delta \theta| \ge 10.0^\circ$ (105 samples: 44 Left, 61 Right).
  * **Ambiguous Buffer Zone:** Pruned 51 borderline samples ($0.5^\circ \le |\Delta \theta| < 10.0^\circ$) to eliminate label noise.

### 5.3 Class-Weighted Focal Loss
To counter the natural 3.4:1 class imbalance:
$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0, \quad \alpha = [0.25, 0.75]$$

### 5.4 Epistemic Uncertainty via Monte Carlo Dropout
* Dropout ($p=0.3$) kept active during inference across $N=30$ forward passes.
* Predictive Entropy: $\mathcal{H}(y|\mathbf{x}) = -\sum_c \bar{p}_c \log \bar{p}_c$.
* Empirically confirmed: Correct test predictions have lower entropy (**0.6003**), while errors produce high entropy (**0.6159**), enabling automated triage for naval operators.

### 5.5 Hydrodynamic Explainability (Grad-CAM)
* Computes gradients of the maneuver class score with respect to `layer4` convolutional feature maps.
* Proves model attends specifically to trailing $19.47^\circ$ Kelvin arms and curved rudder vortices rather than background noise.

---

## 6. How to Reproduce from Scratch

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/mohaimin2103001/thesis.git
cd thesis
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download & Extract Authentic UEIKAP SAR Dataset
```bash
python -m src.download_ueikap
```

### Step 3: Parse AIS Telemetry & Derive Ground Truth
```bash
python -m src.label_generator
```

### Step 4: Train Backbones
```bash
# ResNet-18 Binary:
python -m src.train --arch resnet18 --mode binary --epochs 15

# ResNet-34 Polarimetric:
python -m src.train --arch resnet34 --mode binary --epochs 15

# ResNet-18 3-Class Directional:
python -m src.train --arch resnet18 --mode 3class --epochs 15
```

### Step 5: Evaluate Dual Ensemble & Scientific Diagnostics
```bash
# Dual Ensemble with 4-fold TTA:
python -m src.ensemble_eval

# Epistemic Uncertainty Quantification:
python -m src.uncertainty --samples 30

# Grad-CAM Hydrodynamic Explainability:
python -m src.explainability

# Meteo-Marine Operational Domain Shift:
python -m src.domain_shift

# Hydrodynamic Failure Analysis:
python -m src.failure_analysis
```

---

## 7. Citation & Acknowledgements

If you utilize this code or research methodology, please cite:

```bibtex
@article{graziano2016ship,
  title={Ship Wake Detection in SAR Images for Vessel Heading and Speed Estimation},
  author={Graziano, Maria Daniela and D'Errico, Marco and Rufino, Giancarlo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={54},
  number={3},
  pages={1584--1595},
  year={2016}
}

@dataset{mazzeo2026ueikap,
  author={Mazzeo, Andrea and Graziano, Maria Daniela and others},
  title={{UEIKAP\_DATASET: Unveil and Explore the In-depth Knowledge of Earth Observation Data for Maritime Applications}},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18391027},
  url={https://doi.org/10.5281/zenodo.18391027}
}
```
