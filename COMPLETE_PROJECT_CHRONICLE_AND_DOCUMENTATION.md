# Spaceborne SAR Ship Wake Analysis for Vessel Maneuver Recognition
## Complete Beginning-to-End Technical Chronicle, Mathematics, Codebase Inventory, Bug Post-Mortems, Benchmark Archive & Replication Manual

* **Repository:** [https://github.com/mohaimin2103001/thesis.git](https://github.com/mohaimin2103001/thesis.git)
* **Author:** Mohaiminul Islam ([@mohaimin2103001](https://github.com/mohaimin2103001))
* **Document Scope:** Complete, unskipped engineering and scientific record from initial workspace setup (**October 6, 2026**, commit `a083756`) through the final trajectory-verified, zero-overfit Cloud GPU benchmark (**November 27, 2026**, commits `e1cc73b` / `7690a87` / `abd351d`, evaluated in `results_metrics_only_clean.zip`).
* **Companion Documents:**
  * Detailed Thesis Dossier: [`FULL_THESIS_PROJECT_REPORT.md`](file:///e:/thesis/FULL_THESIS_PROJECT_REPORT.md)
  * Repository Overview: [`README.md`](file:///e:/thesis/README.md)
  * Cross-Platform Local Setup: [`REPRODUCTION_GUIDE.md`](file:///e:/thesis/REPRODUCTION_GUIDE.md)
  * Kaggle Cloud One-Click Guide: [`KAGGLE_MASTER_EXECUTION_GUIDE.md`](file:///e:/thesis/KAGGLE_MASTER_EXECUTION_GUIDE.md)
  * Defense Quick-Reference: [`thesis_defense_guide.md`](file:///e:/thesis/thesis_defense_guide.md)

---

## Executive Snapshot: Final Verified Achievement

| Criterion | Initial Baseline (`ResNet-18`, Raw SAR) | Mid-Stage Physics Fusion (`UEIKAP` 89 Test) | Mixed MMDEC Run 1 (`fcaab1e`, `N=780`) | Failed Causal/EMA Experiment (`c0ebe12`, `N=476`) | **Final Clean Zero-Overfit Suite (`7690a87`, `N=594`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset Scope** | `464` UEIKAP (`94` test) | `449` UEIKAP (`89` test) | `5,190` (`4,741` MMDEC + `449` UEIKAP; incl. sensor fallback) | `3,008` (`2,559` MMDEC [100% fallback] + `449` UEIKAP) | **`4,108` Pure Trajectory-Verified (`3,659` MMDEC + `449` UEIKAP)** |
| **Turning Vessels** | `105` total (`15` test) | `104` total (`12` test) | `1,175` total (`161` test, ~249 fallback) | `525` total (`84` test; 421/421 fallback) | **`1,222` genuine `d(COG)/dt` (`163` test; `0` fallback)** |
| **ROC-AUC (Combined)** | `0.4612` | `0.6948` (`0.8126` if $\|ROT\|\ge 2$) | `0.7808` (`0.8575` excl. fallback) | `0.5197` | **`0.8165`** (`MMDEC`: **`0.8334`**, $\|ROT\|\ge 2$: **`0.8572`**) |
| **PR-AUC** | — | — | `0.5225` | `0.1808` | **`0.6937`** (`MMDEC`: **`0.7245`**) |
| **Straight / Turning Acc** | `65.8%` / `33.3%` | `67.4%` Bal (`\tau=0.68`) | `84.98%` / `57.14%` (`\tau=0.52`) | `41.07%` / `60.71%` | **`76.10%` / `72.39%`** (`\tau=0.52`); **`79.35%` / `69.94%`** (`\tau=0.56`) |
| **Balanced Accuracy** | `51.20%` | `67.1%` (`\tau^*=0.68`) | `71.06%` | `50.89%` | **`74.25%`** (`\tau=0.52`) / **`74.64%`** (`\tau=0.56`) / **`75.98%` MMDEC** |
| **Overfit Audit (Train-Val Gap)** | Not audited | High overfitting risk (`N_{turn}=76`) | Not logged per epoch | Frozen (`ValAUC 0.5886`) | **Zero Overfit: `-0.0072` (`convnext_tiny`), `+0.0259` (`swinv2_tiny`)** |
| **MC-Dropout Selective Triage** | Entropy `0.6346` vs `0.6330` | — | Top `50%` keep: `88.97%` Ov (`Bal 80.69%`) | Rejected due to freeze | **Top `50%` keep: `AUC 0.8918` (`St 87.44%, Tu 79.59%, Bal 83.51%, Ov 84.85%`)**<br>**Top `40%` keep: `AUC 0.9001` (`St 88.96%, Tu 82.43%, Bal 85.69%, Ov 86.92%`)** |

---

## Table of Contents
1. [Part I: Chronological Engineering Log (All 30 Git Commits & 7 Eras)](#part-i-chronological-engineering-log)
2. [Part II: Authentic Datasets, AIS Kinematics & Sensor Physics](#part-ii-authentic-datasets-ais-kinematics--sensor-physics)
3. [Part III: Preprocessing, Wake-Net Segmentation, Radon & 20-D Physics Features](#part-iii-preprocessing-wake-net-segmentation-radon--geometry)
4. [Part IV: Deep Architectures, Loss Design, D4 Chiral Symmetry & Training Engine](#part-iv-deep-architectures-losses--d4-symmetry)
5. [Part V: Complete Experimental Archive (Every Phase & Every Table)](#part-v-complete-experimental-archive)
6. [Part VI: Engineering Bug Post-Mortems & Diagnostic Proofs](#part-vi-engineering-bug-post-mortems)
7. [Part VII: Complete Repository & File-by-File Inventory](#part-vii-complete-file-inventory)
8. [Part VIII: End-to-End Replication & Operational Usage](#part-viii-end-to-end-replication)
9. [Part IX: Honest Limitations, Defense Viva Synthesis & Future Roadmap](#part-ix-limitations--future-work)

---

## Part I: Chronological Engineering Log

Every milestone in this repository is recorded verbatim in the Git history (`30` commits from `a083756` to `abd351d`):

### Era 1: Data Integrity, Dynamic Range Recovery & Initial CNN Baseline (`2026-10-06`)
* **`a083756`** (*Complete Spaceborne SAR Ship Wake Analysis Research*):
  * Identified and rejected a legacy synthetic script ([`generate_synthetic_data.py`](file:///e:/thesis/generate_synthetic_data.py)) and debunked an invalid Zenodo DOI (`14589078`, which resolved via Zenodo API to a beetle taxonomy paper).
  * Discovered and downloaded the authentic **UEIKAP_DATASET** (`10.5281/zenodo.18391027`, `UEIKAP_SAR.zip`, `1,571,288,093` bytes, MD5 `356148359a9ae6b7ae3d3cec2fb61304`) with resilient `curl`/Python downloaders ([`src/download_ueikap.py`](file:///e:/thesis/src/download_ueikap.py), [`src/fast_download_4parts.py`](file:///e:/thesis/src/fast_download_4parts.py)).
  * Built key normalization matching `515 / 515` SAR PNGs (`FalseColorVVVH`) with AIS CSV telemetry ([`src/label_generator.py`](file:///e:/thesis/src/label_generator.py)); curated `464` clean chips (`277 train / 93 val / 94 test`) after pruning `51` ambiguous samples.
  * Quantified severe specular dynamic-range compression via [`src/eda_audit.py`](file:///e:/thesis/src/eda_audit.py) (`mean = 2.5 DN`, `std = 6.4 DN`), lifting mean to `75.3 DN` (`std = 46.1`) with $P_{1.0}\text{–}P_{99.5}$ stretch + CLAHE (`clipLimit=2.5`), jumping `ResNet-18` test accuracy from `55.32%` to `73.40%`.
  * Trained initial PyTorch models ([`src/train.py`](file:///e:/thesis/src/train.py), [`src/models.py`](file:///e:/thesis/src/models.py), [`src/losses.py`](file:///e:/thesis/src/losses.py)), dual ensemble (`77.66%` / `70.21%` test variants across `outputs/metrics/*`), MC-Dropout uncertainty ([`src/uncertainty.py`](file:///e:/thesis/src/uncertainty.py)), Grad-CAM ([`src/explainability.py`](file:///e:/thesis/src/explainability.py)), domain shift curves ([`src/domain_shift.py`](file:///e:/thesis/src/domain_shift.py)), and error taxonomy ([`src/failure_analysis.py`](file:///e:/thesis/src/failure_analysis.py)).
* **`3735725`** (*Cross-platform reproduction guide*): Added [`REPRODUCTION_GUIDE.md`](file:///e:/thesis/REPRODUCTION_GUIDE.md) and updated [`README.md`](file:///e:/thesis/README.md).

### Era 2: Physics-Guided Wake Geometry & Zero-Leakage MMSI Partitions (`2026-11-02`)
* **`c3907d9` -> `c35ebb8` -> `21a26af` -> `9d1a7e7`**:
  * Built unified UEIKAP metadata parser ([`src/data/ueikap_parser.py`](file:///e:/thesis/src/data/ueikap_parser.py), `531` COCO polygons across `512x512` chips, `ueikap_schema_report.json`), strict MMSI-grouped zero-leakage splits ([`src/data/build_master_dataset.py`](file:///e:/thesis/src/data/build_master_dataset.py) -> `maneuver_master.parquet`: `449` clean ships [`295 train / 65 val / 89 test`, `104 Turning` total]), and analytical centerline skeleton curvature extractor ([`src/geometry/wake_geometry.py`](file:///e:/thesis/src/geometry/wake_geometry.py)).
  * Created Kaggle training harness [`kaggle/KAGGLE_MASTER_TRAINING.py`](file:///e:/thesis/kaggle/KAGGLE_MASTER_TRAINING.py) (`Model 1` to `Model 5`), [`KAGGLE_EXECUTION_GUIDE.md`](file:///e:/thesis/KAGGLE_EXECUTION_GUIDE.md), and [`kaggle_train_and_export.ipynb`](file:///e:/thesis/kaggle_train_and_export.ipynb), resolving Windows-to-Linux relative paths and pure-Python ZIP normalization.

### Era 3: 5-Model Ablation, Physical Nyquist Discovery & Tactical Turns (`2026-11-13` to `2026-11-21`)
* **`3d89bc7` -> `62a8a54` -> `7d55242` -> `51e2a71`**:
  * Evaluated all 5 checkpoints on the frozen `89`-vessel UEIKAP test set ([`src/evaluation/evaluate_checkpoints.py`](file:///e:/thesis/src/evaluation/evaluate_checkpoints.py), [`tune_thresholds.py`](file:///e:/thesis/src/evaluation/tune_thresholds.py), [`generate_publication_figures.py`](file:///e:/thesis/src/evaluation/generate_publication_figures.py)): `Model 5 (Fusion + Physics Guidance)` achieved **`ROC-AUC = 0.6948`**, **`91.67%` turning recall** (`11/12`), **`67.1%` balanced accuracy** at validation threshold $\tau^* = 0.68$, and the **only positive ROT Pearson correlation ($r = +0.126$, $R^2 = +0.007$)**.
  * Validated physical dependency via counterfactual perturbations ([`tables/counterfactual_validation_results.csv`](file:///e:/thesis/tables/counterfactual_validation_results.csv)): erasing wake dropped State F1 from `0.856 -> 0.541`, while erasing the hull preserved `0.812`.
  * Formulated the **0.27-Pixel Sensor Observability Limit**: at $10\text{ kt}$ and $|ROT|=0.5^\circ/\text{min}$, wake deflection over 1 minute is only $2.68\text{ m}$ ($0.27\text{ px}$ at $10\text{ m}$ Sentinel-1 GSD), whereas tactical turns $|ROT| \ge 2.0^\circ/\text{min}$ exceeded $1.5\text{ px}$ and pushed Model 5 ROC-AUC to **`0.8126`**.

### Era 4: Scaling Beyond UEIKAP (`N_turn = 76`) via MMDEC & Hybrid Wake-Net (`2026-11-21` to `2026-11-26`)
* **`a15e878` -> `ae3a626` -> `ae05652` -> `cba096a` -> `c1ac20f` -> `5720c16` -> `cc71361`**:
  * Implemented **Hybrid Wake-Net** ([`src/wake/train_wake_net.py`](file:///e:/thesis/src/wake/train_wake_net.py): ResNet-50 U-Net + `scSE` attention + Radon Transform), **MMDEC Streaming Zip Harvester** ([`src/data/mmdec_extract_and_mask.py`](file:///e:/thesis/src/data/mmdec_extract_and_mask.py), [`kaggle_download_helper.py`](file:///e:/thesis/src/data/kaggle_download_helper.py)), and [`kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py`](file:///e:/thesis/kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py) (Adaptive `7x7` Lee Filter + `ConvNeXt-Tiny` + Geometry MLP + MC-Dropout + Permutation Importance + Raw-vs-Lee Grad-CAM).
  * Fixed PyArrow `datetime64` resolution bugs (`ts_sec` UTC epoch conversion in `5720c16`), sped up sparse Radon Transform 100x, added chiral mirror augmentation (`cc71361`), and generated [`KAGGLE_MASTER_EXECUTION_GUIDE.md`](file:///e:/thesis/KAGGLE_MASTER_EXECUTION_GUIDE.md) + [`kaggle_master_pipeline.ipynb`](file:///e:/thesis/kaggle_master_pipeline.ipynb).

### Era 5: Spatial Stitching, OpenMP Deadlock & Contrast Fixes (`2026-11-26`)
* **`8d75e01` -> `b07cd4e` -> `8606632` -> `94434f0` -> `98684f3` -> `7a26575` -> `cd339ca`**:
  * Added Zenodo HTTP 503 HTML-page guard (`8d75e01`) and reverted an unstable optimizer experiment (`b07cd4e`/`8606632`).
  * Fixed MMDEC `800px` chunk `stride=750` coordinate stitching (`94434f0`), integrated `SatShipAI` radar hull alignment (`<= 1200m`), stabilized Wake-Net (`smp` auto-install + `LayerNorm` heads in `98684f3`).
  * Eliminated Linux `DataLoader` OpenMP fork deadlocks (`cv2.setNumThreads(0)`, `OMP_NUM_THREADS=1`, `num_workers=0`), eliminated an $O(N^2)$ Python list/set skeleton hang in `extract_ordered_centerline`, and added RAM image caching (`7a26575`).
  * Reverted Log-dB stretch & overly tight SatShipAI pruning that depressed AUC (`cd339ca`), reaching baseline expanded **`ROC-AUC = 0.7516` (`St 83.20%, Tu 56.52%, Ov 77.69%`)**.

### Era 6: 3-Ch Wake-Net Adapt, PCA Bend, Dual-Scale `ConvNeXt + SwinV2` (`2026-11-27`)
* **`fcaab1e`**:
  * Solved cross-domain UEIKAP/MMDEC luminance mismatch (`extract_unified_sar_gray`: uses `max(B, R)` when `G` is sparse), built 3-channel Wake-Net (`Dice = 0.5548`, `IoU = 0.3839`, `Detection@IoU30 = 55.17%`, mean Radon error `15.92 deg`), and ran pseudo-label domain adaptation + hysteresis masking (`seed=0.30, grow=0.18` lifting MMDEC strong wake coverage from `52.65%` to `63.87%` on `4,741` ships).
  * Added quadratic-vs-linear PCA centerline bend features (`pca_span`, `pca_width`, `pca_abs_bend`, `pca_signed_bend`, `pca_bend_px`, `pca_lin_quad_gain`, `pca_abs_defl_deg`, `pca_signed_defl_deg`) expanding tabular features to `20` physics + `29` enriched HGB inputs.
  * Trained Dual-Scale `ConvNeXt-Tiny + SwinV2-Tiny` (`5,190` ships, `780` test), reaching **`ROC-AUC = 0.7808` (`St 84.98%, Tu 57.14%, Ov 79.23%`)**, with MC-Dropout 50% triage reaching `88.97%` overall / `80.69%` balanced accuracy.

### Era 7: Root-Cause Sensor Audit, Exact D4 Fix & Final Clean Zero-Overfit Suite (`2026-11-27`)
* **`c0ebe12`**: Attempted causal past-wake AIS window (`[t0 - 240, t0 + 45]`, `crop=640`, `--min_turn_rot 1.25`), 3-backbone training (`convnext_small + swinv2_tiny + resnet50d`), `8x` D4 TTA, `EMA (0.995)` from Epoch 1, `patience=5`, and `label_smooth=0.04` -> **regressed to `AUC = 0.5197` (`44.54%` acc)**.
* **Critical Forensic Audit (`c0ebe12` vs `fcaab1e`):**
  * Proved that `[t0 - 240, t0 + 45]` (`285s < 5 min`) missed sparse offshore AIS pings, throwing **100% (`2,559 / 2,559`)** of MMDEC onto raw `RateOfTurn` sensor-byte fallbacks (`raw_rot`), which achieved only **`19.4%` turning accuracy** in Run 1 (`31` test turns) while pure `d(COG)/dt` trajectories (`rot_raw == -128`, `n=287`) scored **`AUC = 0.8916` (`St 87.3%, Tu 78.4%`)** and fallback-free Run 1 scored **`AUC = 0.8575`** (`MMDEC`: **`0.8832`**)!
* **`e1cc73b` -> `7690a87` -> `abd351d`** (*Final Clean Fix & Documentation*):
  * Restored `window_sec=600` (`min_span_sec=60`) with causal wake tie-breaking and **permanently deleted the `raw_rot` fallback** in [`src/data/mmdec_extract_and_mask.py`](file:///e:/thesis/src/data/mmdec_extract_and_mask.py).
  * Removed broken Epoch-1 EMA, early `patience=5` cutoff, and `label_smooth=0.04` squeeze in [`kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py`](file:///e:/thesis/kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py), keeping `convnext_tiny + swinv2_tiny_window8_256 + geom_mlp`, `4x` Chiral D4 TTA (`D4_corrs = [1.0, 1.0, 1.0]` verified locally), centered stern-aligned View-2 (`scale=0.55`), and live **Train-vs-Val Overfit Audit**.
  * Fixed Jupyter `!git ... && !git ...` shell negation typo (`7690a87`) so Kaggle properly checked out `origin/main`.
  * **Verified Result (`results_metrics_only_clean.zip`, `N = 4,108`, `594` test):** **`AUC = 0.8165` (`MMDEC 0.8334`, `ROT>=2 0.8572`)**, **`PR-AUC = 0.6937`**, **`Turning = 72.39%` (`+15.25%`)**, **`Overfit Gap = -0.0072 / +0.0259`**, **`MC-Dropout 40%–50% triage = 0.8918–0.9001 AUC`**.

---

## Part II: Authentic Datasets, AIS Kinematics & Sensor Physics

### 2.1 Zero-Synthetic Policy & Data Sources
All final results strictly exclude synthetic GAN or procedural SAR scenes:
1. **UEIKAP_DATASET** ([Zenodo `18391027`](https://doi.org/10.5281/zenodo.18391027), Mazzeo, Graziano et al., 2024/2026):
   * Archive: `UEIKAP_SAR.zip` (`1,571,288,093` bytes, MD5 `356148359a9ae6b7ae3d3cec2fb61304`).
   * Raw contents: `515` Sentinel-1 IW GRD `512x512` chips (`FalseColorVVVH`), `522` AIS CSVs, `531` wake polygons (`1.12` avg polygons/chip).
   * Clean MMSI-grouped master subset ([`maneuver_master.parquet`](file:///e:/thesis/data/processed/maneuver_master.parquet)): **`449` vessels** (`345 Straight`, `104 Turning [44 Port, 60 Starboard]`; `295 train [219 St, 76 Tu] / 65 val [49 St, 16 Tu] / 89 test [77 St, 12 Tu]`).
2. **MMDEC Multi-Modal Maritime Dataset** ([Zenodo `17491518`](https://doi.org/10.5281/zenodo.17491518)):
   * `230` Sentinel-1 IW GRD products (`Dataset_S1_chunks.zip`, `64,992` `800x800` JPEG chunks at `stride = 750`, $10\text{ m}$ pixel spacing), `Dataset_S1_chunks_corners.zip` (homography corners), `Dataset_S1_SatShipAI_outputs.zip` (`13,306` ship radar detections -> `8,312` commercial candidates, mean length `153.4 m`), and `Dataset_AIS_POS.parquet` (`19,014,229` AIS records, `430 MB`).
   * **Final Clean Pure-Trajectory MMDEC (`7690a87`):** **`3,659` verified vessels** (`2,541 Straight`, `1,118 Turning [591 Port, 527 Starboard]`; `66.17%` with `rot_raw == -128` recovered cleanly via multi-ping `d(COG)/dt`).
   * **Final Combined Dataset (`master_expanded_mmdec.parquet`):** **`4,108` vessels** (`2,927 train [2,083 St, 844 Tu]`, `587 val [372 St, 215 Tu]`, **`594` test [`431 St, 163 Tu`]**).

### 2.2 AIS Kinematic Ground-Truth Derivation
* **Multi-Ping Unwrapped Trajectory ROT (`d(COG)/dt`):**
  For synchronized overpass ping at $t_0$ (`SOG >= 3.5 knots`), neighbor pings for the same `MMSI` are filtered within `[t0 - 600s, t0 + 600s]` (plus causal wake history window `[t0 - 300s, t0 + 60s]`, requiring $\ge 2$ pings spanning $\ge 60\text{s}$):
  $$\text{COG}_{\text{unwrapped}} = \text{deg}(\text{unwrap}(\text{rad}(\text{COG}))), \qquad ROT = \text{polyfit}\left(\frac{t - t_0}{60}, \text{COG}_{\text{unwrapped}}, 1\right)_0 \; (^\circ/\text{min})$$
* **Deadbands & Physical Sign Convention:**
  * `Straight` (`0`): $|ROT| < 0.45^\circ/\text{min}$ and $|\Delta\text{COG}_{\text{win}}| < 2.5^\circ$.
  * `Turning` (`1`): $|ROT| \ge 1.5^\circ/\text{min}$ (`|ROT| <= 25.0`), split into `Port` ($ROT < 0$) and `Starboard` ($ROT > 0$).
* **The 0.27-Pixel Nyquist Observability Limit:**
  At $U = 10\text{ kt}$ ($5.14\text{ m/s}$), travel per minute is $\Delta s = 308\text{ m}$. At $|ROT| = 0.5^\circ/\text{min}$, lateral curve deflection is $308 \sin(0.5^\circ) = 2.68\text{ m} = 0.27\text{ px}$ at $10\text{ m}$ Sentinel-1 GSD — below noise floor. Raising the MMDEC turn floor to $\ge 1.5^\circ/\text{min}$ and auditing $|ROT| \ge 2.0^\circ/\text{min}$ resolves visible hydrodynamic wake deflection ($> 10\text{–}15\text{ m}$).

---

## Part III: Preprocessing, Wake-Net Segmentation, Radon & 20-D Physics Features

### 3.1 Dynamic Range Recovery & Speckle Filtering
1. **Unified SAR Luminance (`extract_unified_sar_gray`):** Resolves false-color `UEIKAP` PNGs (where green channel is sparse and blue holds `VV/VH` ratio) via `np.maximum(B, R)` when $\bar{G} < 0.40 \bar{B}$, aligning pixel intensities with single-band `MMDEC` backscatter.
2. **Adaptive $7\times 7$ Lee Filter (`lee_filter_fp32`):** Suppresses multiplicative ocean Rayleigh speckle ($\hat{R} = \bar{I} + K(I - \bar{I})$) while preserving sharp Kelvin and centerline edges.
3. **2-D View Input Tensor:** `DualScaleWakeNet` ingests 3 aligned `256x256` spatial channels: `[CLAHE(raw_sar), Lee_filtered_SAR, Wake_Net_probability_mask]`.

### 3.2 Hybrid 3-Channel Wake-Net (`scSE + Radon`)
* **Backbone & Loss ([`src/wake/train_wake_net.py`](file:///e:/thesis/src/wake/train_wake_net.py)):** `ResNet-50` encoder U-Net with Spatial & Channel Squeeze-and-Excitation (`scSE`) decoder attention trained for 35 epochs on `UEIKAP` masks (`IoU = 0.3839`, `Dice = 0.5548`, `Detection@IoU30 = 55.17%`, `Mean Radon Angular Error = 15.92 deg`, `optimal_val_threshold = 0.15`, stored in [`logs/wake_net_evaluation.json`](file:///e:/thesis/logs/wake_net_evaluation.json)).
* **Domain Adaptation & Hysteresis:** Pass-1 pseudo-labeling on MMDEC `train` split only (`344` high-confidence crops merged with `401` UEIKAP GT, 8 adaptation epochs, zero val/test leakage) + morphological hysteresis `(seed=0.30, grow=0.18)`.
* **Radon Transform:** Computes angular energy profile across $\theta \in [0^\circ, 180^\circ)$ along near/far annuli to derive `radon_angle_deg`, `radon_deflection_deg`, and `radon_rot_est`.

### 3.3 The 20 Wake Physics & 29 HGB Tabular Features
* **20 Wake Geometry/Physics Inputs (`PHYSICS_FEATURES` in [`KAGGLE_MASTER_EXPANDED_SUITE.py`](file:///e:/thesis/kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py)):**
  * Skeleton Morphology (`10`): `wake_length`, `mean_curvature`, `median_curvature`, `max_abs_curvature`, `signed_mean_curvature`, `curvature_std`, `wake_asymmetry`, `mask_area`, `geometry_confidence`, `branch_count`
  * Radon (`2`): `radon_deflection_deg`, `radon_rot_est`
  * PCA Linear-vs-Quadratic Curve Fit (`8`): `pca_span`, `pca_width`, `pca_abs_bend`, `pca_signed_bend`, `pca_bend_px`, `pca_lin_quad_gain`, `pca_abs_defl_deg`, `pca_signed_defl_deg`
* **29 Enriched Tabular Features (`build_enriched_tabular_features` for HistGradientBoosting):** Adds 9 scale-free nonlinear ratios (`abs_curv_over_len`, `bend_per_span`, `aspect_ratio`, `radon_pca_agreement`, `sign_consistency`, etc.).
* **28-Column Master Schema Dictionary:** Documented in [`data/processed/column_dictionary.csv`](file:///e:/thesis/data/processed/column_dictionary.csv) (strictly excluding `mmsi` and `sog` from model inputs at inference — **Zero AIS at Inference**).

---

## Part IV: Deep Architectures, Losses & D4 Chiral Symmetry

### 4.1 Dual-Scale Wake-Aligned Multimodal Architecture (`DualScaleWakeNet`)
* **View 1 (Macro Scene):** Full `256x256` 3-channel chip capturing global wake propagation.
* **View 2 (Stern-Aligned Zoom):** Differentiable spatial transformer (`F.affine_grid` + `F.grid_sample`) rotating View 1 by $\alpha = \text{align\_rad} + \pi/2$ and cropping central wake zone (`scale = 0.55`, with slight jitter during training).
* **Fusion & Multi-Task Heads (`192-D` shared representation):** Concatenates `[View1_emb, View2_emb, PhysicsMLP(20-D)]` and predicts 4 coupled heads:
  1. `binary_head`: $P(\text{Turning})$ logit
  2. `direction_head`: 3-class `[Straight, Port, Starboard]`
  3. `turn_lr_head`: Conditional $P(\text{Starboard} \mid \text{Turning})$ logit
  4. `rot_head`: Continuous signed `ROT` ($^\circ/\text{min}$)

### 4.2 Physics-Informed Multi-Task Loss (`PhysicsInformedMultiTaskLoss`)
$$\mathcal{L} = \mathcal{L}_{\text{state}}^{\text{focal}}(\text{pos\_weight}=1.30) + 0.45\,\mathcal{L}_{\text{dir}} + 0.40\,\mathcal{L}_{\text{lr}} + 0.20\,(\mathcal{L}_{\text{rot}} + 0.25\,\mathcal{L}_{\text{straight=0}}) + 0.10\,\mathcal{L}_{\text{phys\_sign}} + 0.25\,\mathcal{L}_{\text{spatial\_wake}}$$
* Uses `rot_clarity` weighting (`0.75 + 0.10|rot_true|`, clipped `[0.75, 1.40]`), straight-vessel ROT zeroing, sign consistency penalty, and spatial attention supervision against the segmented wake mask (`imgs[:, 2:3]`).

### 4.3 Exact D4 Chiral Transformation & 4x TTA (`_transform_batch_torch`)
A horizontal mirror (`hflip=True`) reverses chiral turn sense while rotating ($90^\circ \times k_{\text{rot}}$) preserves chiral turn sense. Under `k_rot` ($k \times 90^\circ$ CCW in pixel coords) and optional `hflip`:
1. Image is rotated (`torch.rot90(..., k_rot)`) then optionally flipped along width (`dims=[3]`).
2. Wake orientation vector $\theta_{\text{rad}}$ transforms identically as `(1 - 2 * hflip) * (theta_rad - k_rot * pi/2)`, verified to produce exact **`corr = 1.0000`** on View-2 aligned tensors across all D4 group elements.
3. Chiral tabular features (`signed_mean_curvature`, `wake_asymmetry`, `radon_deflection_deg`, `radon_rot_est`, `pca_signed_bend`, `pca_signed_defl_deg`) are sign-inverted when `hflip=True`, and `Port/Starboard` probabilities (`dv[:, [0, 2, 1]]`, `1 - lrv`) + `rot_pred` are un-flipped back before TTA averaging!

---

## Part V: Complete Experimental Archive

### 5.1 Stage A: Early Local UEIKAP Experiments (`94` Test Samples, [`outputs/metrics/`](file:///e:/thesis/outputs/metrics/))
* Saved JSON benchmarks from initial `UEIKAP` runs (`ResNet-18 binary`: `test_acc 62.77%`, `bal_acc 53.54%`, `AUC 0.4439`; `ResNet-34 binary`: `test_acc 63.83%`, `AUC 0.5105`; `ResNet-18 3-class`: `test_acc 65.96%`, `bal_acc 31.29%`; `Dual Ensemble TTA`: `test_acc 70.21%`, `bal_acc 47.17%`; Domain Shift under Rayleigh speckle `0.0 -> 1.0`: `81.91% -> 73.40%`). Earlier frozen checkpoint table in Section 8 of `FULL_THESIS_PROJECT_REPORT.md` recorded `77.66%` acc / `62.41%` bal acc after CLAHE preprocessing recovery (`2.5 DN -> 75.3 DN`).

### 5.2 Stage B: Frozen UEIKAP `89`-Sample Physics Fusion & Counterfactuals ([`tables/*.csv`](file:///e:/thesis/tables/))
| Model (`tables/main_ablation_results.csv`, `n=89`) | State Acc | State Bal Acc | Turn Recall | ROC-AUC | Dir Acc | ROT MAE | ROT Pearson $r$ | ROT $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Raw SAR Image-Only** | 0.1798 | 0.4908 | 0.9167 | 0.6840 | 0.4167 | 0.725 | -0.025 | -0.047 |
| **Model 2: Segmented Wake-Only** | 0.5730 | 0.5070 | 0.4167 | 0.4805 | 0.1667 | 0.943 | -0.049 | -0.122 |
| **Model 3: Wake Geometry-Only (MLP)** | 0.1798 | 0.5260 | 1.0000 | 0.4978 | 0.4167 | 0.592 | -0.123 | -0.028 |
| **Model 4: Visual-Geometry Fusion** | 0.7978 | 0.4610 | 0.0000 | 0.4589 | 0.5833 | 1.157 | +0.037 | -0.237 |
| **Model 5: Fusion + Physics Guidance** | **0.2022** (`0.674` at $\tau^*=0.68$) | **0.5038** (`0.671` at $\tau^*$) | **0.9167** | **0.6948** (`0.8126` if $\|ROT\|\ge 2$) | 0.4167 | 0.691 | **+0.126** | **+0.007** |

* **Counterfactual Perturbations ([`tables/counterfactual_validation_results.csv`](file:///e:/thesis/tables/counterfactual_validation_results.csv)):** Original Fused (`State F1=0.856, Dir F1=0.849, ROT MAE=1.12`) -> **Wake Erased** (`0.541 / 0.320 / 4.85`) -> **Hull Erased** (`0.812 / 0.825 / 1.34`) -> **Wake Mirrored** (`0.848 / 0.082 / 5.92`) -> **Wake Straightened** (`0.592 / 0.410 / 3.45`).

### 5.3 Stage C: MMDEC + UEIKAP Expansion Progression (`cd339ca -> fcaab1e -> c0ebe12 -> Clean 7690a87`)
| Suite Iteration | Dataset (`Train / Val / Test`) | Straight Acc | Turning Acc | Balanced Acc | Overall Acc | ROC-AUC | PR-AUC | Overfit Gap / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`cd339ca`** | Mixed MMDEC+UEIKAP (`n_test=780`) | `83.20%` | `56.52%` | `69.86%` | `77.69%` | `0.7516` | — | Baseline expanded dual-scale |
| **`fcaab1e` (Run 1 All)** | `5,190` mixed (`3,555 / 855 / 780`) | `84.98%` | `57.14%` | `71.06%` | `79.23%` | `0.7808` | `0.5225` | Incl. `224` fallback ships (`1,556` in MMDEC) |
| *`fcaab1e` Post-Hoc Clean Audit* | *Excl. `224` fallback (`n_test=556`, $\tau=0.46$)* | *`82.39%`* | *`75.38%`* | *`78.89%`* | *`80.76%`* | *`0.8575`* | — | *`MMDEC n=467`: `AUC 0.8832`; `rot_raw=-128`: `0.8916`* |
| **`c0ebe12` (Run 2 Failed)** | `3,008` (`100%` fallback MMDEC, `n_test=476`) | `41.07%` | `60.71%` | `50.89%` | `44.54%` | `0.5197` | `0.1808` | Frozen by `285s` AIS bug + Ep-1 EMA (`ValAUC 0.5886`) |
| **`7690a87` (FINAL CLEAN)** | **`4,108` Pure Trajectory (`2,927 / 587 / 594`)** | **`76.10%`** (`\tau=0.52`)<br>**`79.35%`** (`\tau=0.56`) | **`72.39%`** (`\tau=0.52`)<br>**`69.94%`** (`\tau=0.56`) | **`74.25%`** (`\tau=0.52`)<br>**`74.64%`** (`\tau=0.56`) | **`75.08%`** (`\tau=0.52`)<br>**`76.77%`** (`\tau=0.56`) | **`0.8165`** | **`0.6937`** | **`convnext`: `-0.0072` \| `swinv2`: `+0.0259` gap!** |

### 5.4 Final Clean Run Detailed Breakdowns ([`master_evaluation_metrics.json`](file:///e:/thesis/checkpoints/convnext_ensemble/master_evaluation_metrics.json))
* **Branch Ablations (`N = 594`, $\tau = 0.52$):**
  * `geom_mlp_only`: `AUC 0.5999`, `PR-AUC 0.3424`, `St 37.82%`, `Tu 74.85%`, `Bal 56.33%`
  * `geom_hgb_only`: `AUC 0.6064`, `PR-AUC 0.3419`, `St 54.29%`, `Tu 64.42%`, `Bal 59.35%`
  * `convnext_tiny`: `AUC 0.7920`, `PR-AUC 0.6796`, `St 76.57%`, `Tu 69.94%`, `Bal 73.25%` (`ValAUC 0.8325`)
  * `swinv2_tiny_window8_256`: `AUC 0.8180`, `PR-AUC 0.6876`, `St 72.85%`, `Tu 73.01%`, `Bal 72.93%` (`ValAUC 0.8495`)
  * **Ensemble (`0.27 ConvNeXt-Tiny + 0.63 SwinV2-Tiny + 0.10 GeomMLP`):** `AUC 0.8165`, `PR-AUC 0.6937`, `St 76.10%` (`328/431`), `Tu 72.39%` (`118/163`), `Bal 74.25%`, `Ov 75.08%` (`TN=328, FP=103, FN=45, TP=118`).
* **Domain & Regime Subsets:**
  * `MMDEC Test (n=505)`: `AUC 0.8334`, `PR-AUC 0.7245`, `St 77.12%`, `Tu 74.83%`, `Bal 75.98%`, `Ov 76.44%`.
  * `UEIKAP Test (n=89)`: `AUC 0.5974`, `St 71.43%`, `Tu 41.67%` (`58.33%` Port/Starboard turn accuracy).
  * `Clear Tactical Turns (|ROT| >= 2.0, n=525)`: `AUC 0.8572`, `St 76.10%`, `Tu 78.72%`, `Bal 77.41%`, `Ov 76.57%`.
  * `SOG Regimes`: `0–5 kt (n=49)`: `AUC 0.6746, Tu 94.74%`; `5–10 kt (n=171)`: **`AUC 0.8011, St 64.36%, Tu 74.29%`**; `10–15 kt (n=280)`: `AUC 0.7014, St 83.54%, Tu 48.65%`; `>= 15 kt (n=94)`: **`AUC 0.7880, St 75.00%, Tu 66.67%`**.
* **MC-Dropout Selective Decision Support (`N = 30` passes, `mean_entropy_correct 0.6231 < mean_entropy_incorrect 0.6551`):**
  * Keep `80%` (`n=475`): `AUC 0.8394`, `St 79.33%`, `Tu 74.66%`, `Bal 76.99%`, `Ov 77.89%`
  * Keep `60%` (`n=356`): `AUC 0.8728`, `St 82.35%`, `Tu 77.97%`, `Bal 80.16%`, `Ov 80.90%`
  * Keep `50%` (`n=297`): **`AUC 0.8918`, `St 87.44%`, `Tu 79.59%`, `Bal 83.51%`, `Ov 84.85%`**
  * Keep `40%` (`n=237`): **`AUC 0.9001`, `St 88.96%`, `Tu 82.43%`, `Bal 85.69%`, `Ov 86.92%`**
* **Top-6 Permutation Feature Importances ([`convnext_physics_feature_importance.csv`](file:///e:/thesis/tables/convnext_physics_feature_importance.csv)):** `pca_width` (`-2.15%`), `pca_abs_defl_deg` (`-1.81%`), `pca_signed_defl_deg` (`-1.77%`), `radon_angle_deg` (`-1.69%`), `pca_bend_px` (`-1.69%`), `signed_mean_curvature` (`-1.65%`).

---

## Part VI: Engineering Bug Post-Mortems

Seven major silent failure modes were caught and eliminated across the project:
1. **Dynamic Range Crushing:** Raw `2.5 DN` ocean images collapsed ImageNet CNNs (`55.32%`) -> fixed via $P_{1.0}\text{–}P_{99.5}$ + CLAHE (`mean = 75.3 DN`).
2. **UEIKAP Channel Mismatch:** UEIKAP `FalseColorVVVH` PNGs have near-empty green channel vs MMDEC grayscale -> fixed via `extract_unified_sar_gray` (`max(B, R)`).
3. **MMDEC `stride=750` Stitching & PyArrow Time Bugs:** `id_chunk` stitching and PyArrow timestamp scaling bugs distorted crops and AIS time deltas -> fixed via `dt_utc` UTC epoch conversion + `stride=750` mosaic stitching + `SatShipAI` radar alignment (`<= 1200m`).
4. **Linux OpenMP Deadlock & $O(N^2)$ Skeleton Hang (`7a26575`):** `num_workers > 0` forked OpenCV/OpenMP threads into deadlocks on Kaggle, and skeleton tracing rebuilt Python lists quadratically -> fixed via `cv2.setNumThreads(0)`, `num_workers=0`, RAM caching, and vectorized boolean chain tracing.
5. **The `raw_rot` Sensor Fallback Trap (`c0ebe12` -> `e1cc73b`):** When `c0ebe12` shrank the AIS window to `285s` (`[t0 - 240, t0 + 45]`), sparse ~5-min pings vanished, pushing `2,559 / 2,559` MMDEC ships onto raw `RateOfTurn` sensor bytes (`19.4%` turning precision!) -> fixed by restoring `window_sec = 600` (`min_span_sec = 60`) and **deleting the `raw_rot` fallback completely**.
6. **Epoch-1 EMA + Early Stop Freeze (`c0ebe12` -> `e1cc73b`):** `ema_decay=0.995` applied from Epoch 1 blended initial random weights while `patience=5` halted training at Ep 7 (`ValAUC 0.5886`) -> removed EMA drag, restored `patience=15`, `label_smooth=0.0`, `backbone_lr_mult=0.35`, jumping `ValAUC` to `0.8325 / 0.8495`.
7. **Jupyter `!git && !git` Shell Bug (`7690a87`):** `!cmd1 && !cmd2` evaluated `!git` as a literal bash command (`command not found`), silently blocking Kaggle from pulling `e1cc73b` -> fixed with standard `subprocess.run(..., check=True)` / single-`!` bash syntax.

---

## Part VII: Complete Repository & File-by-File Inventory

### Root Documentation & Notebooks
* [`COMPLETE_PROJECT_CHRONICLE_AND_DOCUMENTATION.md`](file:///e:/thesis/COMPLETE_PROJECT_CHRONICLE_AND_DOCUMENTATION.md) — This full beginning-to-end chronicle
* [`FULL_THESIS_PROJECT_REPORT.md`](file:///e:/thesis/FULL_THESIS_PROJECT_REPORT.md) — 19-section academic thesis dossier + BibTeX
* [`README.md`](file:///e:/thesis/README.md) — Repository landing page
* [`REPRODUCTION_GUIDE.md`](file:///e:/thesis/REPRODUCTION_GUIDE.md) — Cross-platform local setup guide
* [`KAGGLE_MASTER_EXECUTION_GUIDE.md`](file:///e:/thesis/KAGGLE_MASTER_EXECUTION_GUIDE.md) & [`kaggle_master_pipeline.ipynb`](file:///e:/thesis/kaggle_master_pipeline.ipynb) — Final 3-stage MMDEC+UEIKAP cloud notebook & guide
* [`KAGGLE_EXECUTION_GUIDE.md`](file:///e:/thesis/KAGGLE_EXECUTION_GUIDE.md) & [`kaggle_train_and_export.ipynb`](file:///e:/thesis/kaggle_train_and_export.ipynb) — Mid-stage UEIKAP 5-model cloud runner
* [`thesis_defense_guide.md`](file:///e:/thesis/thesis_defense_guide.md), [`IEEE_TGRS_Research_Paper.pdf`](file:///e:/thesis/IEEE_TGRS_Research_Paper.pdf), [`Thesis_Project_Report.pdf`](file:///e:/thesis/Thesis_Project_Report.pdf), [`requirements.txt`](file:///e:/thesis/requirements.txt), [`generate_synthetic_data.py`](file:///e:/thesis/generate_synthetic_data.py) *(archived/unused)*

### Source Modules (`src/` & `kaggle/`)
* **Phase 1–5 Early Suite:** [`dataset.py`](file:///e:/thesis/src/dataset.py), [`download_ueikap.py`](file:///e:/thesis/src/download_ueikap.py), [`fast_download_4parts.py`](file:///e:/thesis/src/fast_download_4parts.py), [`eda_audit.py`](file:///e:/thesis/src/eda_audit.py), [`label_generator.py`](file:///e:/thesis/src/label_generator.py), [`models.py`](file:///e:/thesis/src/models.py), [`losses.py`](file:///e:/thesis/src/losses.py), [`train.py`](file:///e:/thesis/src/train.py), [`ensemble_eval.py`](file:///e:/thesis/src/ensemble_eval.py), [`uncertainty.py`](file:///e:/thesis/src/uncertainty.py), [`explainability.py`](file:///e:/thesis/src/explainability.py), [`domain_shift.py`](file:///e:/thesis/src/domain_shift.py), [`failure_analysis.py`](file:///e:/thesis/src/failure_analysis.py), [`pipeline_runner.py`](file:///e:/thesis/src/pipeline_runner.py), [`run_all.py`](file:///e:/thesis/src/run_all.py)
* **Data & Sync (`src/data/`, `src/synchronization/`):** [`ueikap_parser.py`](file:///e:/thesis/src/data/ueikap_parser.py), [`build_master_dataset.py`](file:///e:/thesis/src/data/build_master_dataset.py), [`kaggle_download_helper.py`](file:///e:/thesis/src/data/kaggle_download_helper.py), [`download_mmdec_metadata.py`](file:///e:/thesis/src/data/download_mmdec_metadata.py), [`mmdec_harvester.py`](file:///e:/thesis/src/data/mmdec_harvester.py), [`mmdec_extract_and_mask.py`](file:///e:/thesis/src/data/mmdec_extract_and_mask.py), [`build_product_time_index.py`](file:///e:/thesis/src/synchronization/build_product_time_index.py)
* **Wake Segmentation & Geometry (`src/wake/`, `src/geometry/`):** [`package_segmentation_data.py`](file:///e:/thesis/src/wake/package_segmentation_data.py), [`train_segmenter.py`](file:///e:/thesis/src/wake/train_segmenter.py), [`train_wake_net.py`](file:///e:/thesis/src/wake/train_wake_net.py), [`wake_geometry.py`](file:///e:/thesis/src/geometry/wake_geometry.py)
* **Training & Kaggle Orchestrators (`src/training/`, `kaggle/`):** [`train_boosted_wake_attention.py`](file:///e:/thesis/src/training/train_boosted_wake_attention.py), [`train_convnext_ensemble.py`](file:///e:/thesis/src/training/train_convnext_ensemble.py), [`KAGGLE_MASTER_TRAINING.py`](file:///e:/thesis/kaggle/KAGGLE_MASTER_TRAINING.py), [`KAGGLE_MASTER_EXPANDED_SUITE.py`](file:///e:/thesis/kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py)
* **Evaluation (`src/evaluation/`):** [`evaluate_all.py`](file:///e:/thesis/src/evaluation/evaluate_all.py), [`evaluate_checkpoints.py`](file:///e:/thesis/src/evaluation/evaluate_checkpoints.py), [`tune_thresholds.py`](file:///e:/thesis/src/evaluation/tune_thresholds.py), [`generate_model5_gradcam.py`](file:///e:/thesis/src/evaluation/generate_model5_gradcam.py), [`generate_publication_figures.py`](file:///e:/thesis/src/evaluation/generate_publication_figures.py)

### Processed Artifacts & Metrics Outputs
* **Metrics & Predictions:** [`master_evaluation_metrics.json`](file:///e:/thesis/checkpoints/convnext_ensemble/master_evaluation_metrics.json), [`wake_net_evaluation.json`](file:///e:/thesis/logs/wake_net_evaluation.json), [`expanded_test_predictions.parquet`](file:///e:/thesis/data/processed/expanded_test_predictions.parquet), [`master_expanded_mmdec.parquet`](file:///e:/thesis/data/processed/master_expanded_mmdec.parquet), [`maneuver_master.parquet`](file:///e:/thesis/data/processed/maneuver_master.parquet), [`column_dictionary.csv`](file:///e:/thesis/data/processed/column_dictionary.csv)
* **Tables (`tables/`):** [`main_ablation_results.csv`](file:///e:/thesis/tables/main_ablation_results.csv), [`calibrated_state_metrics.csv`](file:///e:/thesis/tables/calibrated_state_metrics.csv), [`state_metrics.csv`](file:///e:/thesis/tables/state_metrics.csv), [`direction_metrics.csv`](file:///e:/thesis/tables/direction_metrics.csv), [`rot_metrics.csv`](file:///e:/thesis/tables/rot_metrics.csv), [`counterfactual_validation_results.csv`](file:///e:/thesis/tables/counterfactual_validation_results.csv), [`convnext_physics_feature_importance.csv`](file:///e:/thesis/tables/convnext_physics_feature_importance.csv), [`convnext_gradcam_analysis.csv`](file:///e:/thesis/tables/convnext_gradcam_analysis.csv), [`model5_gradcam_analysis.csv`](file:///e:/thesis/tables/model5_gradcam_analysis.csv)
* **Figures (`figures/`, `outputs/`):** [`convnext_physics_feature_importance.png`](file:///e:/thesis/figures/convnext_physics_feature_importance.png), [`convnext_uncertainty_rejection_curve.png`](file:///e:/thesis/figures/convnext_uncertainty_rejection_curve.png), [`fig15_main_ablation_comparison.png`](file:///e:/thesis/figures/fig15_main_ablation_comparison.png), [`fig16_rot_scatter_comparison.png`](file:///e:/thesis/figures/fig16_rot_scatter_comparison.png), [`fig18_counterfactual_validation.png`](file:///e:/thesis/figures/fig18_counterfactual_validation.png), [`fig24_robustness_curves.png`](file:///e:/thesis/figures/fig24_robustness_curves.png)

---

## Part VIII: End-to-End Replication

### Cloud GPU Execution (Recommended, ~15–20 min fresh / ~10–12 min cached)
Follow [`KAGGLE_MASTER_EXECUTION_GUIDE.md`](file:///e:/thesis/KAGGLE_MASTER_EXECUTION_GUIDE.md) or run the 3 CLI stages after [`src/data/kaggle_download_helper.py`](file:///e:/thesis/src/data/kaggle_download_helper.py):
```bash
# Stage 1: Unified 3-Channel Wake-Net (ResNet-50 + scSE)
python -u -m src.wake.train_wake_net --epochs 35 --batch_size 8 --lr 2e-4

# Stage 2: Pure Trajectory-Verified MMDEC Harvest (Zero raw_rot fallback) + Auto-Mask + PCA/Radon Geometry
python -u -m src.data.mmdec_extract_and_mask --min_sog 3.5 --min_turn_rot 1.5 --max_straight 4000 --max_turning -1 --crop_size 512 --batch_size 32

# Stage 3: Dual-Scale ConvNeXt-Tiny + SwinV2-Tiny + Geometry MLP + 4x Chiral D4 TTA + Overfit Audit
python -u kaggle/KAGGLE_MASTER_EXPANDED_SUITE.py --epochs 20 --batch_size 24 --lr 2e-4 --img_size 256 --primary_backbone convnext_tiny --second_backbone swinv2_tiny_window8_256 --third_backbone none --gradcam_per_class -1
```

---

## Part IX: Honest Limitations & Future Work

1. **Port vs. Starboard Turn Chirality (`47.24%` test)** and **Continuous Signed ROT ($R^2 = -0.0042$, `MAE = 1.15 deg/min`)**: Single-look C-band magnitude imagery under wind-wave speckle lacks phase/Doppler directional flow cues and suffers amplitude washout for weak turns (`< 2 deg/min`). Treated cleanly as auxiliary outputs while primary operational surveillance relies on `Straight vs. Turning` + `MC-Dropout` entropy triage.
2. **UEIKAP-Only Subset (`n=89`, `AUC 0.5974`)**: Represents only `10.9%` of combined training (`295 / 2,927`) vs dense European shelf traffic in `MMDEC`.
3. **Future Work:** Self-supervised Masked Autoencoder (MAE) pre-training on Sentinel-2 optical wakes (`UEIKAP_optical.zip`, `~12 GB`) paired via cross-modal contrastive alignment (`CLIP/MoCo`) to transfer clean optical cusp geometry into noisy C-band SAR.
