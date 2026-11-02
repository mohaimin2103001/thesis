# Dataset Card: Spaceborne SAR Vessel Maneuver & Wake Geometry Benchmark
## Explainable Physics-Guided Multi-Task Benchmark for Turning State, Handedness & Signed ROT Estimation

---

### 1. Dataset Summary & Permanent Registries
This benchmark pairs spaceborne European Space Agency (ESA) Sentinel-1 C-Band Synthetic Aperture Radar (SAR) imagery with synchronized Automatic Identification System (AIS) kinematic ground truth and explicit hydrodynamic wake morphology.

- **Primary Segmentation Benchmark:** **UEIKAP_DATASET**
  - **Permanent DOI:** [10.5281/zenodo.18391027](https://doi.org/10.5281/zenodo.18391027)
  - **Role:** Supervised wake segmentation ground truth (514 rasterized polygon masks, 512x512 chips).
- **Primary Maneuver & Telemetry Benchmark:** **MMDEC (English Channel)**
  - **Permanent DOI:** [10.5281/zenodo.17491518](https://doi.org/10.5281/zenodo.17491518)
  - **Paper DOI:** [10.1016/j.dib.2026.112629](https://doi.org/10.1016/j.dib.2026.112629)
  - **Role:** Large multimodal SAR chunks, SatShipAI vessel detections, AIS POS telemetry.
- **Auxiliary Benchmark:** **OpenSARWake**
  - **Role:** Additional bounding box wake localization.

---

### 2. Operational Invariance Constraint (Strict Research Rule)
> **FINAL DEPLOYMENT MUST USE SAR PIXELS & SAR-DERIVED WAKE GEOMETRY ONLY.**

At final operational inference, the system receives **only the raw SAR chip**. It has **no access** to:
- MMSI or vessel identity
- AIS Speed Over Ground (SOG)
- AIS Course Over Ground (COG)
- AIS True Heading
- AIS Rate of Turn (ROT)
- AIS historical trajectories or weather metadata

AIS telemetry is used exclusively during dataset construction for automatic ground-truth derivation and scientific validation.

---

### 3. Kinematic Labeling & Scientific Deadbands
Maneuver ground truth is derived using AIS Rate of Turn ($ROT$) and circular course-heading drift angle ($\Delta\theta = ((COG - HDG + 180) \bmod 360) - 180$):

1. **Target A — Maneuver State:**
   - **Straight (0):** $|ROT| < 0.5^\circ/\text{min}$ AND $|\Delta\theta| < 5.0^\circ$
   - **Turning (1):** $|ROT| \ge 0.5^\circ/\text{min}$ OR $|\Delta\theta| \ge 10.0^\circ$
2. **Target B — Turn Handedness (Turning Vessels Only):**
   - **Port / Left Turn (0):** $ROT < -0.5^\circ/\text{min}$ (or $\Delta\theta \le -10.0^\circ$)
   - **Starboard / Right Turn (1):** $ROT > +0.5^\circ/\text{min}$ (or $\Delta\theta \ge +10.0^\circ$)
3. **Target C — Continuous Signed ROT:**
   - High-precision numerical target in degrees/minute ($[-15^\circ/\text{min}, +15^\circ/\text{min}]$).

#### Quality Tiers:
- **Gold-A (High-Confidence):** Stable continuous ROT, time gap $\le 120\text{s}$, unambiguous ship association. Used for all tasks (Classification & Regression).
- **Gold-B (Directional):** Categorical turn ($\pm 127$ raw AIS code or $120\text{s}-300\text{s}$ time window). Used only for State & Direction classification.
- **Reject:** Missing ROT ($-128$), $>300\text{s}$ time difference, multi-vessel positional ambiguity. Documented in `rejected_samples.parquet`.

---

### 4. Mathematical Physics of Wake Geometry
From cleaned wake masks, morphological skeletons are traced from the vessel hull towards the wake tail and fitted with smooth cubic B-splines $(x(s), y(s))$:
$$\kappa(s) = \frac{x'(s)y''(s) - y'(s)x''(s)}{(x'(s)^2 + y'(s)^2)^{3/2}}$$

Explicit features extracted:
- `mean_curvature`: $\frac{1}{L}\int |\kappa(s)| ds$
- `max_abs_curvature`: $\max |\kappa(s)|$
- `signed_mean_curvature`: $\frac{1}{L}\int \kappa(s) ds$ (positive for Starboard, negative for Port)
- `curvature_std`: localized turn sharpness
- `wake_length`: visible hydrodynamic memory length
- `wake_orientation`: chord angle
- `wake_asymmetry`: lateral area displacement
- `geometry_confidence`: skeleton graph continuity

---

### 5. Grouped Partitioning & Zero-Leakage Guarantee
Data splitting is partitioned strictly by **MMSI vessel identifier** and **Sentinel-1 acquisition orbit pass**:
- **Train (295 samples):** 65% of unique vessels
- **Validation (65 samples):** 15% of unique vessels
- **Frozen Test (89 samples):** 20% of unique vessels (completely unseen ships and passes)
- **Rule:** Multiple chips from the same ship never bridge train and test sets.
