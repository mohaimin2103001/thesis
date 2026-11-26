"""
====================================================================================================
KAGGLE_MASTER_EXPANDED_SUITE.py
End-to-End Master Training & Evaluation Pipeline for Expanded MMDEC + UEIKAP Benchmark
====================================================================================================
Architecture & Capabilities (Method 1 + Method 3 Hybrid + Multi-Task Thesis Goals):
  1. Branch A: 7x7 Differentiable Lee Speckle Filter + ConvNeXt-Tiny + 12 Wake/Radon Physics Features
  2. Branch B: Pure Wake Geometry Specialist MLP (Model 3) on Curvature & Radon Deflection
  3. Dynamic Soft-Voting Ensemble: 0.60 * P_ConvNeXt + 0.40 * P_Geometry (calibrated on Val split)
  4. Multi-Task Predictions (Zero AIS at inference time — strictly SAR + Wake Geometry):
       - Task 1: Binary Maneuver State Recognition (Straight vs Turning -> Targeting 88%-92%+)
       - Task 2: Directional 3-Class Classification (Straight [0] vs Port [1] vs Starboard [2])
       - Task 3: Continuous Signed Rate-of-Turn Regression (deg/min, MAE & R^2)
       - Task 4: Real-Time Operational Latency Benchmark (ms/vessel & FPS)
  5. Evaluates on BOTH Expanded Test Set AND Frozen 89-Ship UEIKAP Test Set, plus saves Grad-CAM.
====================================================================================================
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

IS_KAGGLE = os.path.exists("/kaggle/working")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints", "convnext_ensemble")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
GRADCAM_DIR = os.path.join(BASE_DIR, "outputs", "gradcam_convnext")

for d in [CHECKPOINTS_DIR, TABLES_DIR, FIGURES_DIR, GRADCAM_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"[MASTER SUITE] Environment: {'Kaggle Cloud' if IS_KAGGLE else 'Local'} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[MASTER SUITE] GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# Strictly observable SAR/Wake geometry features (Zero AIS at inference time!)
PHYSICS_FEATURES = [
    "wake_length", "wake_orientation", "mean_curvature", "median_curvature",
    "max_abs_curvature", "signed_mean_curvature", "curvature_std",
    "wake_asymmetry", "mask_area", "geometry_confidence",
    "radon_angle_deg", "radon_deflection_deg"
]

GEOMETRY_FEATURES = [
    "mean_curvature", "median_curvature", "max_abs_curvature",
    "signed_mean_curvature", "curvature_std", "wake_asymmetry",
    "radon_deflection_deg"
]

CHIRAL_PHYS_INDICES = [
    PHYSICS_FEATURES.index("signed_mean_curvature"),
    PHYSICS_FEATURES.index("wake_asymmetry"),
    PHYSICS_FEATURES.index("radon_deflection_deg"),
]


def build_enriched_tabular_features(phys_raw):
    """
    Derives 20 pure SAR/Wake tabular features from the 12 observable geometry/Radon features
    (ZERO AIS inputs). Handles missing/sparse wake masks cleanly via magnitude & ratio descriptors.
    """
    p = np.asarray(phys_raw, dtype=np.float32)
    wake_len = p[:, 0:1]
    mean_c = p[:, 2:3]
    med_c = p[:, 3:4]
    max_c = p[:, 4:5]
    s_curv = p[:, 5:6]
    asym = p[:, 7:8]
    area = p[:, 8:9]
    r_defl = p[:, 11:12]

    derived = np.hstack([
        np.abs(s_curv),
        np.abs(asym),
        np.abs(r_defl),
        mean_c * wake_len,
        wake_len / (np.sqrt(np.maximum(area, 0.0)) + 1.0),
        max_c / (med_c + 1e-4),
        (area >= 50.0).astype(np.float32),
        np.sign(r_defl) * np.sqrt(np.abs(r_defl)),
    ])
    return np.hstack([p, derived]).astype(np.float32)

_PATH_CACHE = {}

def resolve_path(p):
    """Cross-platform path resolver bridging Windows paths to Linux/Kaggle workspace."""
    if not p or not isinstance(p, str):
        return p
    if p in _PATH_CACHE:
        return _PATH_CACHE[p]
    norm_p = p.replace("\\", "/")
    if os.path.exists(norm_p):
        _PATH_CACHE[p] = norm_p
        return norm_p
    fname = os.path.basename(norm_p)
    for base in [
        os.path.join(DATA_DIR, "interim", "mmdec_crops"),
        os.path.join(DATA_DIR, "raw", "UEIKAP_SAR", "images"),
        os.path.join(DATA_DIR, "processed", "wake_segmentation_kaggle", "images"),
        os.path.join(DATA_DIR, "interim"),
        os.path.join(DATA_DIR, "raw"),
    ]:
        if os.path.exists(base):
            candidate = os.path.join(base, fname)
            if os.path.exists(candidate):
                _PATH_CACHE[p] = candidate
                return candidate
            for root, _, files in os.walk(base):
                if fname in files:
                    found = os.path.join(root, fname)
                    _PATH_CACHE[p] = found
                    return found
    _PATH_CACHE[p] = norm_p
    return norm_p


# --------------------------------------------------------------------------------------------------
# Adaptive 7x7 Lee Speckle Filter + Dual-Branch Architectures
# --------------------------------------------------------------------------------------------------
class LeeSpeckleFilter(nn.Module):
    """Differentiable FP32 7x7 Lee filter suppressing multiplicative SAR sea clutter speckle."""
    def __init__(self, kernel_size=7, noise_var=0.08):
        super().__init__()
        self.pad = kernel_size // 2
        self.noise_var = noise_var
        box_weight = torch.full((1, 1, kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=torch.float32)
        self.register_buffer("box_weight", box_weight)

    def forward(self, x):
        # Strictly run local variance estimation in FP32 to avoid AMP FP16 cancellation
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            w = self.box_weight.float()
            mean = F.conv2d(xf, w, padding=self.pad)
            sq_mean = F.conv2d(xf * xf, w, padding=self.pad)
            var = torch.clamp(sq_mean - mean * mean, min=1e-6)
            k = torch.clamp((var - (mean * mean) * self.noise_var) / var, min=0.0, max=1.0)
            return torch.clamp(mean + k * (xf - mean), 0.0, 1.0)


class ConvNeXtPhysicsNet(nn.Module):
    """
    Branch A (Method 3): FP32 7x7 Lee Filter + ConvNeXt-Tiny + 12 Wake/Radon Geometry Features.
    Outputs Multi-Task Heads: Binary State Logit, 3-Class Direction Logits, Continuous Signed ROT, Turning Port/Stbd Logit.
    """
    def __init__(self, num_physics_features=12, pretrained=True):
        super().__init__()
        self.speckle_filter = LeeSpeckleFilter(kernel_size=7, noise_var=0.08)
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            drop_rate=0.10,
            drop_path_rate=0.10,
        )
        embed_dim = self.backbone.num_features  # 768

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.physics_mlp = nn.Sequential(
            nn.Linear(num_physics_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.SiLU()
        )

        self.shared_head = nn.Sequential(
            nn.Linear(embed_dim + 64, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.15)
        )

        self.binary_head = nn.Linear(160, 1)
        self.direction_head = nn.Linear(160, 3)
        self.turn_lr_head = nn.Linear(160, 1)
        self.rot_head = nn.Linear(160, 1)

    def forward(self, x_img, x_phys):
        # Accepts 3-ch [log_sar, clahe_sar, blur_mask] or 2-ch [sar, mask]
        log_ch = x_img[:, 0:1]
        clahe_ch = x_img[:, 1:2] if x_img.shape[1] >= 3 else log_ch
        mask_ch = x_img[:, 2:3] if x_img.shape[1] >= 3 else x_img[:, 1:2]
        lee_ch = self.speckle_filter(log_ch)
        # Smooth wake spotlight avoids binary step-edge gradient spikes in stage 0
        spot_ch = torch.clamp(0.55 * clahe_ch.float() + 0.45 * mask_ch.float(), 0.0, 1.0)
        x_raw3 = torch.cat([lee_ch, clahe_ch.float(), spot_ch], dim=1)
        x_3ch = (x_raw3 - self.img_mean) / self.img_std
        v_emb = self.backbone(x_3ch)
        p_emb = self.physics_mlp(x_phys)
        feat = self.shared_head(torch.cat([v_emb, p_emb], dim=1))
        return (
            self.binary_head(feat).squeeze(1),
            self.direction_head(feat),
            self.rot_head(feat).squeeze(1),
            self.turn_lr_head(feat).squeeze(1),
        )


class GeometrySpecialistMLP(nn.Module):
    """
    Branch B (Method 1 Anchor / Model 3): Pure Wake Curvature & Radon Specialist.
    Uses LayerNorm for small-batch stability across sparse/unmasked rows.
    """
    def __init__(self, in_features=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(32, 1)
        self.direction_head = nn.Linear(32, 3)
        self.turn_lr_head = nn.Linear(32, 1)

    def forward(self, x_geom):
        h = self.net(x_geom)
        return self.binary_head(h).squeeze(1), self.direction_head(h), self.turn_lr_head(h).squeeze(1)


# --------------------------------------------------------------------------------------------------
# Multimodal Dataset with Train-Only Normalization (Zero Leakage)
# --------------------------------------------------------------------------------------------------
class ScaledManeuverDataset(Dataset):
    def __init__(self, df, phys_mean=None, phys_std=None, img_size=(320, 320), augment=False, rot_clip=12.0, preload=True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for col in PHYSICS_FEATURES:
            if col not in self.df.columns:
                self.df[col] = 0.0
            self.df[col] = self.df[col].fillna(0.0).astype(np.float32)

        phys_raw = self.df[PHYSICS_FEATURES].values.astype(np.float32)
        if phys_mean is None or phys_std is None:
            self.phys_mean = np.nanmean(phys_raw, axis=0)
            self.phys_std = np.nanstd(phys_raw, axis=0) + 1e-6
        else:
            self.phys_mean = phys_mean
            self.phys_std = phys_std
        self.phys_raw = phys_raw
        self.phys_norm = np.clip((phys_raw - self.phys_mean) / self.phys_std, -8.0, 8.0)

        self.geom_indices = [PHYSICS_FEATURES.index(c) for c in GEOMETRY_FEATURES]
        self.geom_norm = self.phys_norm[:, self.geom_indices]
        self.chiral_indices = [PHYSICS_FEATURES.index(c) for c in ["signed_mean_curvature", "wake_asymmetry", "radon_deflection_deg"]]

        # Resolve ground-truth labels cleanly across both UEIKAP ('state', 'rot_target') and any alias
        state_col = "state" if "state" in self.df.columns else "maneuver_state"
        self.state_labels = self.df[state_col].astype(str).str.lower().eq("turning").astype(int).values

        dir_map = {"none": 0, "straight": 0, "port": 1, "left": 1, "starboard": 2, "right": 2}
        if "direction" in self.df.columns:
            self.dir_labels = self.df["direction"].astype(str).str.lower().map(lambda d: dir_map.get(d, 0)).values
        else:
            rot_vals = self.df.get("rot_target", self.df.get("rot_deg_min", pd.Series(np.zeros(len(self.df))))).values
            self.dir_labels = np.where(np.abs(rot_vals) < 0.5, 0, np.where(rot_vals < 0, 1, 2))

        rot_col = "rot_target" if "rot_target" in self.df.columns else ("rot_deg_min" if "rot_deg_min" in self.df.columns else "rot")
        raw_rot = self.df.get(rot_col, pd.Series(np.zeros(len(self.df)))).fillna(0.0).values.astype(np.float32)
        self.rot_values = np.clip(raw_rot, -float(rot_clip), float(rot_clip))

        # Preload preprocessed [log_u8, clahe_u8, blur_mask_u8] in RAM (keeps GPU at 90% & removes fork deadlock)
        self.cached_imgs = None
        est_bytes = len(self.df) * 3 * int(img_size[0]) * int(img_size[1])
        if preload and len(self.df) > 0 and est_bytes <= 2_400_000_000:
            cache = np.empty((len(self.df), 3, int(img_size[1]), int(img_size[0])), dtype=np.uint8)
            for idx in range(len(self.df)):
                cache[idx] = self._load_item_u8(idx)
            self.cached_imgs = cache

    def _load_item_u8(self, idx):
        row = self.df.iloc[idx]
        p_crop = resolve_path(str(row.get("crop_path", "")))
        p_orig = resolve_path(str(row.get("original_image_path", "")))
        img_p = p_crop if os.path.exists(p_crop) else p_orig

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load SAR chip for sample {row.get('sample_id')}: {img_p}")
        img_res = cv2.resize(img, self.img_size)

        # Fast strided percentile [p1, p99.5] in log-dB space (~160x faster than full-frame sort)
        log_g = np.log1p(img_res.astype(np.float32))
        lo, hi = np.percentile(log_g[::4, ::4], [1.0, 99.5])
        if hi - lo > 1e-4:
            log_n = np.clip((log_g - lo) / (hi - lo), 0.0, 1.0)
        else:
            log_n = img_res.astype(np.float32) / 255.0
        log_u8 = (log_n * 255.0).astype(np.uint8)
        clahe_u8 = self.clahe.apply(log_u8)

        m_path = resolve_path(str(row.get("wake_mask_path", "")))
        if os.path.exists(m_path):
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(m_img, self.img_size, interpolation=cv2.INTER_NEAREST) if m_img is not None else np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        mask_blur = cv2.GaussianBlur((mask > 127).astype(np.float32), (9, 9), 2.5)
        mask_u8 = np.clip(mask_blur * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.stack([log_u8, clahe_u8, mask_u8], axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        arr_u8 = self.cached_imgs[idx] if self.cached_imgs is not None else self._load_item_u8(idx)
        img_3ch = arr_u8.astype(np.float32) / 255.0

        phys_vec = self.phys_raw[idx].copy()
        dir_val = int(self.dir_labels[idx])
        rot_val = float(self.rot_values[idx])

        if self.augment:
            if np.random.rand() < 0.5:
                img_3ch = img_3ch[:, :, ::-1].copy()
                if dir_val == 1:
                    dir_val = 2
                elif dir_val == 2:
                    dir_val = 1
                rot_val = -rot_val
                for ci in self.chiral_indices:
                    phys_vec[ci] = -phys_vec[ci]
            if np.random.rand() < 0.4:
                gamma = np.random.uniform(0.90, 1.10)
                img_3ch[:2] = np.clip(img_3ch[:2] * gamma, 0.0, 1.0)

        phys_norm_vec = np.clip((phys_vec - self.phys_mean) / self.phys_std, -8.0, 8.0).astype(np.float32)
        geom_norm_vec = phys_norm_vec[self.geom_indices]

        img_t = torch.from_numpy(img_3ch)
        phys_t = torch.from_numpy(phys_norm_vec)
        geom_t = torch.from_numpy(geom_norm_vec)
        state_t = torch.tensor(self.state_labels[idx], dtype=torch.float32)
        dir_t = torch.tensor(dir_val, dtype=torch.long)
        rot_t = torch.tensor(rot_val, dtype=torch.float32)

        return img_t, phys_t, geom_t, state_t, dir_t, rot_t


# --------------------------------------------------------------------------------------------------
# Physics-Informed Multi-Task Loss & Hierarchical Chiral Inference
# --------------------------------------------------------------------------------------------------
class PhysicsInformedMultiTaskLoss(nn.Module):
    def __init__(self, pos_weight=1.25, dir_weights=None, lambda_dir=0.55, lambda_lr=0.55, lambda_rot=0.25, lambda_phys=0.10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
        if dir_weights is not None:
            self.ce_dir = nn.CrossEntropyLoss(weight=torch.tensor(dir_weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
        else:
            self.ce_dir = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.bce_lr = nn.BCEWithLogitsLoss()
        self.lambda_dir = lambda_dir
        self.lambda_lr = lambda_lr
        self.lambda_rot = lambda_rot
        self.lambda_phys = lambda_phys

    def forward(self, state_logit, dir_logits, rot_pred, lr_logit, state_t, dir_t, rot_t):
        l_state = self.bce(state_logit, state_t)
        l_dir = self.ce_dir(dir_logits, dir_t)

        # Dedicated Port (0) vs Starboard (1) loss strictly on true Turning vessels (avoids class-0 interference)
        turn_mask = state_t > 0.5
        if torch.any(turn_mask):
            lr_target = (dir_t[turn_mask] == 2).float() * 0.90 + 0.05  # label smoothing 0.05
            l_lr = self.bce_lr(lr_logit[turn_mask], lr_target)
        else:
            l_lr = state_logit.sum() * 0.0

        # Clipped Signed ROT regression + Straight zero pull + Chiral sign coherence
        rot_weights = 1.0 + 1.5 * state_t
        l_rot = torch.mean(rot_weights * F.smooth_l1_loss(rot_pred, rot_t, reduction="none", beta=0.75))
        l_straight_zero = torch.mean((1.0 - state_t) * (rot_pred ** 2)) * 0.25
        l_phys = torch.mean(state_t * F.relu(-rot_pred * torch.sign(rot_t)))
        return l_state + self.lambda_dir * l_dir + self.lambda_lr * l_lr + self.lambda_rot * (l_rot + l_straight_zero) + self.lambda_phys * l_phys


def build_hierarchical_dir_probs(state_probs, p_stbd_cond):
    """Combines P(Turn) with P(Starboard | Turn) into a calibrated 3-class distribution."""
    p_st = np.clip(1.0 - state_probs, 0.0, 1.0)
    p_sb = np.clip(state_probs * p_stbd_cond, 0.0, 1.0)
    p_pt = np.clip(state_probs * (1.0 - p_stbd_cond), 0.0, 1.0)
    return np.column_stack([p_st, p_pt, p_sb])


def evaluate_predictions(state_probs, dir_probs, rot_preds, state_true, dir_true, rot_true, tau=0.50, p_stbd_cond=None, tau_lr=0.50):
    pred_bin = (state_probs >= tau).astype(int)
    st_mask = state_true == 0
    tu_mask = state_true == 1

    st_acc = float(np.mean(pred_bin[st_mask] == 0)) if np.any(st_mask) else 0.0
    tu_acc = float(np.mean(pred_bin[tu_mask] == 1)) if np.any(tu_mask) else 0.0
    bal_acc = 0.5 * (st_acc + tu_acc)
    overall_acc = float(np.mean(pred_bin == state_true))

    auc_roc = float(roc_auc_score(state_true, state_probs)) if len(np.unique(state_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(state_true, state_probs)) if len(np.unique(state_true)) > 1 else 0.5

    # Hierarchical coherent 3-Class (Straight=0 if p_turn < tau else Port=1 / Starboard=2)
    if p_stbd_cond is None:
        denom = dir_probs[:, 1] + dir_probs[:, 2] + 1e-8
        p_stbd_cond = dir_probs[:, 2] / denom
    hier_pred = np.where(pred_bin == 0, 0, np.where(p_stbd_cond >= tau_lr, 2, 1))
    softmax_pred = np.argmax(dir_probs, axis=1)

    dir_overall = float(np.mean(hier_pred == dir_true))
    c_accs = []
    for c in (0, 1, 2):
        m = dir_true == c
        c_accs.append(float(np.mean(hier_pred[m] == c)) if np.any(m) else 0.0)
    dir_bal = float(np.mean(c_accs))

    sm_accs = [float(np.mean(softmax_pred[dir_true == c] == c)) if np.any(dir_true == c) else 0.0 for c in (0, 1, 2)]

    # Port vs Starboard purely on true Turning vessels
    turn_mask = (dir_true == 1) | (dir_true == 2)
    turn_dir_acc = float(np.mean(np.where(p_stbd_cond[turn_mask] >= tau_lr, 2, 1) == dir_true[turn_mask])) if np.any(turn_mask) else 0.0

    # Continuous Signed ROT
    rot_mae = float(np.mean(np.abs(rot_preds - rot_true)))
    ss_res = float(np.sum((rot_true - rot_preds) ** 2))
    ss_tot = float(np.sum((rot_true - np.mean(rot_true)) ** 2)) + 1e-6
    rot_r2 = 1.0 - (ss_res / ss_tot)

    return {
        "threshold": float(tau),
        "threshold_lr": float(tau_lr),
        "n_samples": int(len(state_true)),
        "n_straight": int(np.sum(st_mask)),
        "n_turning": int(np.sum(tu_mask)),
        "balanced_accuracy": bal_acc,
        "overall_accuracy": overall_acc,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "straight_accuracy": st_acc,
        "turning_accuracy": tu_acc,
        "dir_3class_overall_acc": dir_overall,
        "dir_3class_balanced_acc": dir_bal,
        "dir_straight_acc": c_accs[0],
        "dir_port_acc": c_accs[1],
        "dir_starboard_acc": c_accs[2],
        "softmax_dir_3class_overall_acc": float(np.mean(softmax_pred == dir_true)),
        "softmax_dir_3class_balanced_acc": float(np.mean(sm_accs)),
        "turn_port_vs_starboard_acc": turn_dir_acc,
        "rot_mae_deg_min": rot_mae,
        "rot_r2": rot_r2,
    }


def generate_lee_vs_raw_gradcam(convnext_net, test_ds, out_dir, num_per_class=-1):
    """
    Generates comparative Raw SAR vs 7x7 Lee-Filtered ConvNeXt Grad-CAM visualizations
    across ALL test vessels (when num_per_class=-1) or balanced subsets, overlays wake mask contours,
    and quantifies Inside-Wake Attention Concentration in tables/convnext_gradcam_analysis.csv.
    """
    convnext_net.eval()
    target_layer = convnext_net.backbone.stages[-1]
    activations = []
    gradients = []

    def fwd_hook(_, __, out):
        activations.append(out.detach())

    def bwd_hook(_, __, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    if num_per_class <= 0:
        selected_indices = list(range(len(test_ds)))
    else:
        selected_indices = []
        for cls_id in (0, 1, 2):
            cls_idxs = [i for i in range(len(test_ds)) if test_ds.dir_labels[i] == cls_id][:num_per_class]
            selected_indices.extend(cls_idxs)

    print(f"[GRAD-CAM] Generating Raw vs 7x7 Lee Grad-CAM heatmaps for all {len(selected_indices)} test images...")
    audit_rows = []
    for k, idx in enumerate(selected_indices):
        activations.clear()
        gradients.clear()
        img_t, phys_t, _, st_true, dir_true, rot_true = test_ds[idx]
        x_img = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
        x_phys = phys_t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            lee_img = convnext_net.speckle_filter(x_img[:, 0:1]).squeeze().cpu().numpy()

        st_logit, dir_logits, rot_pred, lr_logit = convnext_net(x_img, x_phys)
        convnext_net.zero_grad()

        # Target hierarchical predicted turning/direction logit (falls back cleanly for straight ships)
        p_turn_cx = float(torch.sigmoid(st_logit).item())
        if p_turn_cx >= 0.5:
            score = lr_logit[0] if float(torch.sigmoid(lr_logit).item()) >= 0.5 else -lr_logit[0]
        else:
            score = -st_logit[0]
        score.backward()

        h_out, w_out = int(img_t.shape[1]), int(img_t.shape[2])
        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_res = cv2.resize(cam, (w_out, h_out))

        # Overlay wake mask contour & compute wake attention concentration
        row = test_ds.df.iloc[idx]
        m_path = resolve_path(str(row.get("wake_mask_path", "")))
        mask_bin = None
        wake_energy_ratio = np.nan
        if os.path.exists(m_path):
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m_img is not None:
                m_res = cv2.resize(m_img, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
                mask_bin = m_res > 127
                if np.any(mask_bin):
                    total_e = float(np.sum(cam_res)) + 1e-8
                    wake_energy_ratio = float(np.sum(cam_res[mask_bin]) / total_e)

        raw_np = img_t[0].numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_np, cmap="gray")
        axes[0].set_title(f"Raw SAR ({row.get('source_dataset', 'SAR')})")
        axes[1].imshow(lee_img, cmap="gray")
        axes[1].set_title("7x7 Lee Speckle Filtered")
        axes[2].imshow(lee_img, cmap="gray")
        axes[2].imshow(cam_res, cmap="jet", alpha=0.45)
        if mask_bin is not None and np.any(mask_bin):
            axes[2].contour(mask_bin.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        dir_str = row.get("direction", "None")
        axes[2].set_title(f"Grad-CAM | GT:{row.get('state')}/{dir_str} | ROT:{rot_true:.1f}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        sid_safe = str(row.get("sample_id", f"idx_{idx}")).replace("/", "_").replace("\\", "_")
        plt.savefig(os.path.join(out_dir, f"convnext_gradcam_{k+1:04d}_{sid_safe}.png"), dpi=110)
        plt.close(fig)

        audit_rows.append({
            "sample_id": row.get("sample_id", f"idx_{idx}"),
            "source_dataset": row.get("source_dataset", "SAR"),
            "state_true": row.get("state", "Unknown"),
            "direction_true": dir_str,
            "rot_true": float(rot_true),
            "prob_turn_convnext": float(torch.sigmoid(st_logit).item()),
            "rot_pred": float(rot_pred.item()),
            "inside_wake_energy_ratio": wake_energy_ratio
        })

    h1.remove()
    h2.remove()
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(os.path.join(TABLES_DIR, "convnext_gradcam_analysis.csv"), index=False)


def run_mc_dropout_uncertainty(convnext_net, geom_mlp, test_loader, state_true, tau=0.50, w_cx=0.70, num_passes=30):
    """
    Executes N=30 stochastic MC-Dropout forward passes on cached test tensors (< 5s total),
    computes Predictive Entropy and Entropy Gap (Incorrect vs Correct), and saves the
    Selective Error Rejection Curve (figures/convnext_uncertainty_rejection_curve.png).
    """
    # Cache test batches once so 30 MC passes don't re-iterate dataset I/O 30 times
    cached_batches = [(imgs, phys, geom) for imgs, phys, geom, _, _, _ in test_loader]

    convnext_net.eval()
    geom_mlp.eval()
    for m in list(convnext_net.modules()) + list(geom_mlp.modules()):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    pass_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            p_batch = []
            for imgs, phys, geom in cached_batches:
                imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
                st_log, _, _, _ = convnext_net(imgs, phys)
                g_st_log, _, _ = geom_mlp(geom)
                p_ens = (w_cx * torch.sigmoid(st_log) + (1.0 - w_cx) * torch.sigmoid(g_st_log)).cpu().numpy()
                p_batch.append(p_ens)
            pass_probs.append(np.concatenate(p_batch))

    convnext_net.eval()
    geom_mlp.eval()

    mc_matrix = np.stack(pass_probs, axis=0)  # (num_passes, N_test)
    mean_prob = np.mean(mc_matrix, axis=0)
    epistemic_std = np.std(mc_matrix, axis=0)
    eps = 1e-7
    p_clip = np.clip(mean_prob, eps, 1.0 - eps)
    entropy = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))

    preds = (mean_prob >= tau).astype(int)
    correct_mask = preds == state_true
    incorrect_mask = ~correct_mask

    overall_acc = float(np.mean(correct_mask))
    ent_correct = float(np.mean(entropy[correct_mask])) if np.any(correct_mask) else 0.0
    ent_incorrect = float(np.mean(entropy[incorrect_mask])) if np.any(incorrect_mask) else 0.0

    # Selective Rejection Curve
    sorted_idx = np.argsort(entropy)
    rej_rates = np.linspace(0.0, 0.50, 21)
    retained_err = []
    for r in rej_rates:
        keep_n = max(1, int(len(sorted_idx) * (1.0 - r)))
        k_idx = sorted_idx[:keep_n]
        retained_err.append(1.0 - float(np.mean(preds[k_idx] == state_true[k_idx])))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rej_rates * 100, np.array(retained_err) * 100, marker="o", lw=2.2, color="#d62728", label="MC-Dropout Selective Error")
    ax.axhline(y=(1.0 - overall_acc) * 100, color="gray", linestyle="--", label=f"Unfiltered Error ({(1.0 - overall_acc)*100:.1f}%)")
    ax.set_title("ConvNeXt + Lee + Geometry Ensemble: Selective Error Rejection Curve (N=30 MC Passes)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rejected High-Entropy Fraction (% deferred to operator)", fontsize=10)
    ax.set_ylabel("Error Rate on Retained Vessels (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "convnext_uncertainty_rejection_curve.png"), dpi=200)
    plt.close(fig)

    return {
        "mc_passes": int(num_passes),
        "mean_epistemic_std": float(np.mean(epistemic_std)),
        "mean_entropy_correct": ent_correct,
        "mean_entropy_incorrect": ent_incorrect,
        "entropy_gap": ent_incorrect - ent_correct,
        "error_at_0pct_rejection": float(retained_err[0]),
        "error_at_20pct_rejection": float(retained_err[8]),
        "error_at_40pct_rejection": float(retained_err[16]),
    }, mean_prob, epistemic_std, entropy


def compute_physics_permutation_importance(convnext_net, geom_mlp, test_ds, base_bal_acc, tau=0.50, w_cx=0.70, batch_size=24):
    """
    Quantifies Balanced Accuracy drop when each Wake/Radon physics feature is randomly permuted.
    Caches ConvNeXt visual backbone embeddings once since image pixels do not change under physics permutation.
    """
    convnext_net.eval()
    geom_mlp.eval()
    imp_rows = []

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    v_embs = []
    with torch.no_grad():
        for imgs, _, _, _, _, _ in loader:
            x_img = imgs.to(DEVICE)
            log_ch = x_img[:, 0:1]
            clahe_ch = x_img[:, 1:2] if x_img.shape[1] >= 3 else log_ch
            mask_ch = x_img[:, 2:3] if x_img.shape[1] >= 3 else x_img[:, 1:2]
            lee_ch = convnext_net.speckle_filter(log_ch)
            spot_ch = torch.clamp(0.55 * clahe_ch.float() + 0.45 * mask_ch.float(), 0.0, 1.0)
            x_raw3 = torch.cat([lee_ch, clahe_ch.float(), spot_ch], dim=1)
            x_3ch = (x_raw3 - convnext_net.img_mean) / convnext_net.img_std
            v_embs.append(convnext_net.backbone(x_3ch).cpu())
    v_emb_all = torch.cat(v_embs, dim=0).to(DEVICE)

    orig_phys_raw = test_ds.phys_raw.copy()
    geom_indices = [PHYSICS_FEATURES.index(c) for c in GEOMETRY_FEATURES]

    for f_idx, fname in enumerate(PHYSICS_FEATURES):
        perm_raw = orig_phys_raw.copy()
        perm_raw[:, f_idx] = np.random.permutation(perm_raw[:, f_idx])
        perm_norm = np.clip((perm_raw - test_ds.phys_mean) / test_ds.phys_std, -8.0, 8.0).astype(np.float32)
        p_t = torch.from_numpy(perm_norm).to(DEVICE)
        g_t = p_t[:, geom_indices]

        with torch.no_grad():
            p_emb = convnext_net.physics_mlp(p_t)
            feat = convnext_net.shared_head(torch.cat([v_emb_all, p_emb], dim=1))
            st_log = convnext_net.binary_head(feat).squeeze(1)
            g_st_log, _, _ = geom_mlp(g_t)
            p_all = (w_cx * torch.sigmoid(st_log) + (1.0 - w_cx) * torch.sigmoid(g_st_log)).cpu().numpy()

        b = (p_all >= tau).astype(int)
        st_acc = np.mean(b[test_ds.state_labels == 0] == 0)
        tu_acc = np.mean(b[test_ds.state_labels == 1] == 1)
        bal = 0.5 * (st_acc + tu_acc)
        imp_rows.append({"feature": fname, "permuted_bal_acc": float(bal), "importance_drop": float(base_bal_acc - bal)})

    imp_df = pd.DataFrame(imp_rows).sort_values("importance_drop", ascending=False)
    imp_df.to_csv(os.path.join(TABLES_DIR, "convnext_physics_feature_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = imp_df.sort_values("importance_drop", ascending=True)
    ax.barh(plot_df["feature"], plot_df["importance_drop"] * 100, color="#1f77b4")
    ax.set_xlabel("Balanced Accuracy Drop When Permuted (%)", fontsize=10)
    ax.set_title("Hydrodynamic Wake & Radon Feature Importance (Explainability)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "convnext_physics_feature_importance.png"), dpi=200)
    plt.close(fig)
    return imp_df


def _flip_physics_batch(phys_norm, geom_norm, phys_mean, phys_std, geom_indices):
    """Applies exact horizontal mirror sign flip in normalized physics space (odd chiral symmetry)."""
    phys_raw = phys_norm * phys_std + phys_mean
    phys_flip_raw = phys_raw.clone()
    for ci in CHIRAL_PHYS_INDICES:
        phys_flip_raw[:, ci] = -phys_flip_raw[:, ci]
    phys_flip_norm = torch.clamp((phys_flip_raw - phys_mean) / phys_std, -8.0, 8.0)
    geom_flip_norm = phys_flip_norm[:, geom_indices]
    return phys_flip_norm, geom_flip_norm


def predict_neural_branches(convnext_net, geom_mlp, loader, phys_mean_np, phys_std_np, geom_indices, use_tta=True):
    """Runs Val/Test inference with optional 2x Chiral-Consistent Mirror TTA for exact Port/Starboard symmetry."""
    convnext_net.eval()
    geom_mlp.eval()
    phys_mean_t = torch.tensor(phys_mean_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    phys_std_t = torch.tensor(phys_std_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    cx_p, cx_dir, cx_lr, cx_rot = [], [], [], []
    g_p, g_dir, g_lr = [], [], []
    times = []

    with torch.no_grad():
        for imgs, phys, geom, _, _, _ in loader:
            imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
            t0 = time.time()
            st_log, dir_log, rot_out, lr_log = convnext_net(imgs, phys)
            g_st_log, g_dir_log, g_lr_log = geom_mlp(geom)

            p_cx = torch.sigmoid(st_log)
            d_cx = torch.softmax(dir_log, dim=1)
            lr_cx = torch.sigmoid(lr_log)
            r_cx = rot_out

            p_g = torch.sigmoid(g_st_log)
            d_g = torch.softmax(g_dir_log, dim=1)
            lr_g = torch.sigmoid(g_lr_log)

            if use_tta:
                imgs_f = torch.flip(imgs, dims=[3])
                phys_f, geom_f = _flip_physics_batch(phys, geom, phys_mean_t, phys_std_t, geom_indices)
                st_f, dir_f, rot_f, lr_f = convnext_net(imgs_f, phys_f)
                g_st_f, g_dir_f, g_lr_f = geom_mlp(geom_f)

                p_cx = 0.5 * (p_cx + torch.sigmoid(st_f))
                d_cx_f = torch.softmax(dir_f, dim=1)[:, [0, 2, 1]]  # swap Port (1) <-> Starboard (2)
                d_cx = 0.5 * (d_cx + d_cx_f)
                lr_cx = 0.5 * (lr_cx + (1.0 - torch.sigmoid(lr_f)))
                r_cx = 0.5 * (r_cx - rot_f)

                p_g = 0.5 * (p_g + torch.sigmoid(g_st_f))
                d_g_f = torch.softmax(g_dir_f, dim=1)[:, [0, 2, 1]]
                d_g = 0.5 * (d_g + d_g_f)
                lr_g = 0.5 * (lr_g + (1.0 - torch.sigmoid(g_lr_f)))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.time() - t0) / len(imgs))

            cx_p.append(p_cx.cpu().numpy())
            cx_dir.append(d_cx.cpu().numpy())
            cx_lr.append(lr_cx.cpu().numpy())
            cx_rot.append(r_cx.cpu().numpy())
            g_p.append(p_g.cpu().numpy())
            g_dir.append(d_g.cpu().numpy())
            g_lr.append(lr_g.cpu().numpy())

    return {
        "cx_p": np.concatenate(cx_p),
        "cx_dir": np.concatenate(cx_dir),
        "cx_lr": np.concatenate(cx_lr),
        "cx_rot": np.concatenate(cx_rot),
        "mlp_p": np.concatenate(g_p),
        "mlp_dir": np.concatenate(g_dir),
        "mlp_lr": np.concatenate(g_lr),
        "mean_latency_s": float(np.mean(times)),
    }


def train_and_predict_hgb(train_ds, eval_datasets):
    """
    Branch B2: Enriched Physics HistGradientBoosting Specialist (State + Turning Port/Stbd + Signed ROT).
    Handles sparse/zero masks natively via tree splits and applies 2x Chiral TTA.
    """
    X_tr = build_enriched_tabular_features(train_ds.phys_raw)
    y_st_tr = train_ds.state_labels
    y_rot_tr = train_ds.rot_values
    turn_mask_tr = y_st_tr == 1
    y_lr_tr = (train_ds.dir_labels[turn_mask_tr] == 2).astype(int)

    clf_state = HistGradientBoostingClassifier(
        max_iter=250, max_depth=5, learning_rate=0.04, l2_regularization=2.0,
        class_weight="balanced", random_state=42
    )
    clf_state.fit(X_tr, y_st_tr)

    clf_lr = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.04, l2_regularization=2.0,
        class_weight="balanced", random_state=42
    )
    if np.sum(turn_mask_tr) >= 10 and len(np.unique(y_lr_tr)) == 2:
        clf_lr.fit(X_tr[turn_mask_tr], y_lr_tr)
        has_lr = True
    else:
        has_lr = False

    reg_rot = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=250, max_depth=5, learning_rate=0.04,
        l2_regularization=2.0, random_state=42
    )
    reg_rot.fit(X_tr, y_rot_tr)

    outputs = []
    for ds in eval_datasets:
        raw_orig = ds.phys_raw
        raw_flip = raw_orig.copy()
        for ci in CHIRAL_PHYS_INDICES:
            raw_flip[:, ci] = -raw_flip[:, ci]
        X_e = build_enriched_tabular_features(raw_orig)
        X_f = build_enriched_tabular_features(raw_flip)

        p_st = 0.5 * (clf_state.predict_proba(X_e)[:, 1] + clf_state.predict_proba(X_f)[:, 1])
        if has_lr:
            p_lr = 0.5 * (clf_lr.predict_proba(X_e)[:, 1] + (1.0 - clf_lr.predict_proba(X_f)[:, 1]))
        else:
            p_lr = np.full(len(ds), 0.5, dtype=np.float32)
        rot_e = 0.5 * (reg_rot.predict(X_e) - reg_rot.predict(X_f))
        outputs.append({"hgb_p": p_st.astype(np.float32), "hgb_lr": p_lr.astype(np.float32), "hgb_rot": rot_e.astype(np.float32)})
    return outputs


def run_master_suite(epochs=35, batch_size=24, lr=2e-4, img_size=320, gradcam_per_class=-1, allow_partial_data=False):
    # 1. Load Expanded MMDEC+UEIKAP Dataset if present, else UEIKAP master
    expanded_path = os.path.join(DATA_DIR, "processed", "master_expanded_mmdec.parquet")
    ueikap_path = os.path.join(DATA_DIR, "processed", "maneuver_master.parquet")

    if os.path.exists(expanded_path):
        master_df = pd.read_parquet(expanded_path)
        print(f"[DATA] Loaded EXPANDED MMDEC+UEIKAP dataset: {len(master_df):,} ships!")
    elif os.path.exists(ueikap_path):
        master_df = pd.read_parquet(ueikap_path)
        print(f"[DATA] Expanded parquet not found; loaded UEIKAP master: {len(master_df):,} ships.")
    else:
        raise FileNotFoundError("Missing dataset parquet files.")

    # Verify all image paths exist before training
    valid_mask = master_df.apply(
        lambda r: os.path.exists(resolve_path(str(r.get("crop_path", "")))) or os.path.exists(resolve_path(str(r.get("original_image_path", "")))),
        axis=1
    )
    if valid_mask.sum() < len(master_df):
        missing_n = int(len(master_df) - valid_mask.sum())
        if not allow_partial_data and len(master_df) > 1000 and valid_mask.sum() < 0.5 * len(master_df):
            raise RuntimeError(
                f"{missing_n} / {len(master_df)} crops are missing (did Stage 2 mmdec_extract_and_mask fail?). "
                f"Re-run Stage 2 after installing dependencies, or pass --allow_partial_data."
            )
        print(f"[DATA] Filtered {missing_n} missing-image rows; retaining {valid_mask.sum():,} valid chips.")
        master_df = master_df[valid_mask].reset_index(drop=True)

    train_df = master_df[master_df["split"] == "train"].copy()
    val_df = master_df[master_df["split"] == "val"].copy()
    test_df = master_df[master_df["split"] == "test"].copy()

    print(f"[SPLITS] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"[TRAIN BALANCE] State: {train_df['state'].value_counts().to_dict()} | Dir: {train_df['direction'].value_counts().to_dict()}")

    # Fit normalization strictly on Train split
    train_ds = ScaledManeuverDataset(train_df, img_size=(img_size, img_size), augment=True)
    val_ds = ScaledManeuverDataset(val_df, phys_mean=train_ds.phys_mean, phys_std=train_ds.phys_std, img_size=(img_size, img_size), augment=False)
    test_ds = ScaledManeuverDataset(test_df, phys_mean=train_ds.phys_mean, phys_std=train_ds.phys_std, img_size=(img_size, img_size), augment=False)

    # Save scaler stats for future standalone inference
    np.savez(
        os.path.join(CHECKPOINTS_DIR, "feature_scaler.npz"),
        phys_mean=train_ds.phys_mean,
        phys_std=train_ds.phys_std,
        features=np.array(PHYSICS_FEATURES)
    )

    # Balanced sampling without extreme class-weight stacking (plus mild 1.8x boost for manual UEIKAP gold wakes)
    n_st = max(1, int(np.sum(train_ds.state_labels == 0)))
    n_tu = max(1, int(np.sum(train_ds.state_labels == 1)))
    state_bal_w = float(np.clip(n_st / n_tu, 1.0, 2.5))
    sample_weights = np.where(train_ds.state_labels == 1, state_bal_w, 1.0).astype(np.float64)
    if "source_dataset" in train_df.columns:
        is_ueikap = (train_df["source_dataset"].values == "UEIKAP")
        sample_weights = sample_weights * np.where(is_ueikap, 1.8, 1.0)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    # RAM-cached preprocessed uint8 dataset eliminates fork() deadlocks -> num_workers=0 runs at full GPU speed
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate Dual-Branch Neural Models
    convnext_net = ConvNeXtPhysicsNet(num_physics_features=len(PHYSICS_FEATURES), pretrained=True).to(DEVICE)
    geom_mlp = GeometrySpecialistMLP(in_features=len(GEOMETRY_FEATURES)).to(DEVICE)

    dir_weights = np.array([0.85, 1.15, 1.15], dtype=np.float32)
    criterion = PhysicsInformedMultiTaskLoss(pos_weight=1.25, dir_weights=dir_weights).to(DEVICE)
    bce_geom = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25], device=DEVICE))
    ce_geom_dir = nn.CrossEntropyLoss(weight=torch.tensor(dir_weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
    bce_geom_lr = nn.BCEWithLogitsLoss()

    # Decoupled optimizers: ConvNeXt (warmup + cosine) & Geometry MLP (FP32)
    opt_cx = torch.optim.AdamW([
        {"params": convnext_net.backbone.parameters(), "lr": lr * 0.5},
        {"params": convnext_net.physics_mlp.parameters(), "lr": lr * 2.5},
        {"params": convnext_net.shared_head.parameters(), "lr": lr * 2.5},
        {"params": convnext_net.binary_head.parameters(), "lr": lr * 2.5},
        {"params": convnext_net.direction_head.parameters(), "lr": lr * 2.5},
        {"params": convnext_net.turn_lr_head.parameters(), "lr": lr * 2.5},
        {"params": convnext_net.rot_head.parameters(), "lr": lr * 2.5},
    ], weight_decay=1e-4)
    warmup_epochs = min(3, max(1, epochs // 10))

    def _lr_lambda(ep_idx):
        if ep_idx < warmup_epochs:
            return float(ep_idx + 1) / float(warmup_epochs)
        progress = float(ep_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sched_cx = torch.optim.lr_scheduler.LambdaLR(opt_cx, lr_lambda=_lr_lambda)
    opt_geom = torch.optim.AdamW(geom_mlp.parameters(), lr=max(lr * 3.0, 5e-4), weight_decay=1e-4)
    sched_geom = torch.optim.lr_scheduler.CosineAnnealingLR(opt_geom, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_cx_score = -1.0
    best_geom_score = -1.0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        convnext_net.train()
        geom_mlp.train()
        running_cx_loss = 0.0
        running_g_loss = 0.0

        for imgs, phys, geom, state_t, dir_t, rot_t in train_loader:
            imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
            state_t, dir_t, rot_t = state_t.to(DEVICE), dir_t.to(DEVICE), rot_t.to(DEVICE)

            # Branch A: ConvNeXt-Tiny + Lee + Physics (AMP)
            opt_cx.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                st_log, dir_log, rot_out, lr_log = convnext_net(imgs, phys)
                loss_cx = criterion(st_log, dir_log, rot_out, lr_log, state_t, dir_t, rot_t)
            scaler.scale(loss_cx).backward()
            scaler.unscale_(opt_cx)
            torch.nn.utils.clip_grad_norm_(convnext_net.parameters(), max_norm=5.0)
            scaler.step(opt_cx)
            scaler.update()

            # Branch B1: Geometry Specialist MLP (clean FP32)
            opt_geom.zero_grad()
            g_st_log, g_dir_log, g_lr_log = geom_mlp(geom)
            turn_m = state_t > 0.5
            loss_g_lr = bce_geom_lr(g_lr_log[turn_m], (dir_t[turn_m] == 2).float()) if torch.any(turn_m) else (g_st_log.sum() * 0.0)
            loss_g = bce_geom(g_st_log, state_t) + 0.5 * ce_geom_dir(g_dir_log, dir_t) + 0.5 * loss_g_lr
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(geom_mlp.parameters(), max_norm=5.0)
            opt_geom.step()

            running_cx_loss += loss_cx.item() * len(state_t)
            running_g_loss += loss_g.item() * len(state_t)

        sched_cx.step()
        sched_geom.step()
        train_cx_loss = running_cx_loss / max(1, len(train_loader.dataset))
        train_g_loss = running_g_loss / max(1, len(train_loader.dataset))

        # Validation with 2x Chiral TTA
        val_out = predict_neural_branches(
            convnext_net, geom_mlp, val_loader, train_ds.phys_mean, train_ds.phys_std, train_ds.geom_indices, use_tta=True
        )
        cx_stbd = 0.5 * val_out["cx_lr"] + 0.5 * (val_out["cx_dir"][:, 2] / (val_out["cx_dir"][:, 1] + val_out["cx_dir"][:, 2] + 1e-8))
        g_stbd = 0.5 * val_out["mlp_lr"] + 0.5 * (val_out["mlp_dir"][:, 2] / (val_out["mlp_dir"][:, 1] + val_out["mlp_dir"][:, 2] + 1e-8))

        cx_res = evaluate_predictions(
            val_out["cx_p"], val_out["cx_dir"], val_out["cx_rot"],
            val_ds.state_labels, val_ds.dir_labels, val_ds.rot_values, tau=0.50, p_stbd_cond=cx_stbd
        )
        g_res = evaluate_predictions(
            val_out["mlp_p"], val_out["mlp_dir"], val_out["cx_rot"],
            val_ds.state_labels, val_ds.dir_labels, val_ds.rot_values, tau=0.50, p_stbd_cond=g_stbd
        )

        cx_score = 0.45 * cx_res["balanced_accuracy"] + 0.25 * cx_res["auc_roc"] + 0.30 * cx_res["dir_3class_balanced_acc"]
        g_score = 0.50 * g_res["balanced_accuracy"] + 0.25 * g_res["auc_roc"] + 0.25 * g_res["dir_3class_balanced_acc"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | LossCX: {train_cx_loss:.4f} | LossG: {train_g_loss:.4f} | "
            f"ConvNeXt BalAcc: {cx_res['balanced_accuracy']*100:.1f}% (AUC:{cx_res['auc_roc']:.3f}, Dir3:{cx_res['dir_3class_balanced_acc']*100:.1f}%, P/S:{cx_res['turn_port_vs_starboard_acc']*100:.1f}%) | "
            f"GeomMLP BalAcc: {g_res['balanced_accuracy']*100:.1f}%"
        )

        if cx_score > best_cx_score:
            best_cx_score = cx_score
            torch.save(convnext_net.state_dict(), os.path.join(CHECKPOINTS_DIR, "convnext_multitask_best.pt"))
        if g_score > best_geom_score:
            best_geom_score = g_score
            torch.save(geom_mlp.state_dict(), os.path.join(CHECKPOINTS_DIR, "geometry_mlp_best.pt"))

    print(f"\n[TRAINING COMPLETE] Elapsed: {(time.time() - t_start)/60.0:.2f} min")

    # ----------------------------------------------------------------------------------------------
    # Train Branch B2: Pure Physics HistGradientBoosting Specialist + Multi-Branch Val Calibration
    # ----------------------------------------------------------------------------------------------
    val_hgb, test_hgb = train_and_predict_hgb(train_ds, [val_ds, test_ds])

    convnext_net.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "convnext_multitask_best.pt"), map_location=DEVICE))
    geom_mlp.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "geometry_mlp_best.pt"), map_location=DEVICE))

    val_neural = predict_neural_branches(
        convnext_net, geom_mlp, val_loader, train_ds.phys_mean, train_ds.phys_std, train_ds.geom_indices, use_tta=True
    )
    test_neural = predict_neural_branches(
        convnext_net, geom_mlp, test_loader, train_ds.phys_mean, train_ds.phys_std, train_ds.geom_indices, use_tta=True
    )

    v_st_t = val_ds.state_labels
    v_dir_t = val_ds.dir_labels
    v_rot_t = val_ds.rot_values

    # 1. Calibrate Binary Ensemble Weights (w_cx, w_mlp, w_hgb) & Threshold tau on Val
    best_weights = (0.75, 0.10, 0.15)
    best_tau = 0.50
    best_val_obj = -1.0
    for w_cx in (0.50, 0.60, 0.70, 0.80, 0.90):
        for w_hgb in (0.00, 0.10, 0.20, 0.30):
            w_mlp = round(1.0 - w_cx - w_hgb, 2)
            if w_mlp < -1e-6 or w_mlp > 0.40:
                continue
            w_mlp = max(0.0, w_mlp)
            v_ens = w_cx * val_neural["cx_p"] + w_mlp * val_neural["mlp_p"] + w_hgb * val_hgb["hgb_p"]
            v_auc = float(roc_auc_score(v_st_t, v_ens)) if len(np.unique(v_st_t)) > 1 else 0.5
            for tau in np.arange(0.32, 0.74, 0.02):
                b = (v_ens >= tau).astype(int)
                st_a = np.mean(b[v_st_t == 0] == 0)
                tu_a = np.mean(b[v_st_t == 1] == 1)
                bal = 0.5 * (st_a + tu_a)
                obj = 0.85 * bal + 0.15 * v_auc
                if obj > best_val_obj:
                    best_val_obj = obj
                    best_weights = (float(w_cx), float(w_mlp), float(w_hgb))
                    best_tau = float(tau)

    w_cx, w_mlp, w_hgb = best_weights
    v_ens = w_cx * val_neural["cx_p"] + w_mlp * val_neural["mlp_p"] + w_hgb * val_hgb["hgb_p"]
    v_auc = float(roc_auc_score(v_st_t, v_ens)) if len(np.unique(v_st_t)) > 1 else 0.5
    v_bal = 0.5 * (np.mean((v_ens >= best_tau)[v_st_t == 0] == 0) + np.mean((v_ens >= best_tau)[v_st_t == 1] == 1))

    # 2. Calibrate Port vs Starboard Mix & Threshold tau_lr Strictly on Turning Validation Samples
    val_cx_lr = 0.65 * val_neural["cx_lr"] + 0.35 * (val_neural["cx_dir"][:, 2] / (val_neural["cx_dir"][:, 1] + val_neural["cx_dir"][:, 2] + 1e-8))
    val_mlp_lr = val_neural["mlp_lr"]
    val_turn_m = v_st_t == 1
    best_lr_mix, best_tau_lr, best_lr_acc = (0.75, 0.10, 0.15), 0.50, -1.0
    if np.any(val_turn_m):
        for lc in (0.60, 0.75, 0.90, 1.00):
            for lh in (0.00, 0.15, 0.25):
                lm = round(1.0 - lc - lh, 2)
                if lm < -1e-6:
                    continue
                lm = max(0.0, lm)
                p_lr_v = lc * val_cx_lr + lm * val_mlp_lr + lh * val_hgb["hgb_lr"]
                for t_lr in np.arange(0.42, 0.59, 0.02):
                    pred_lr = np.where(p_lr_v[val_turn_m] >= t_lr, 2, 1)
                    acc_lr = float(np.mean(pred_lr == v_dir_t[val_turn_m]))
                    if acc_lr > best_lr_acc:
                        best_lr_acc = acc_lr
                        best_lr_mix = (float(lc), float(lm), float(lh))
                        best_tau_lr = float(t_lr)

    lc, lm, lh = best_lr_mix

    # 3. Calibrate Signed ROT Regression Blend & Straight Gate Strictly on Validation R^2
    best_alpha_rot, best_st_scale, best_val_r2 = 0.85, 0.20, -99.0
    v_pred_bin = (v_ens >= best_tau)
    for alpha_r in (0.65, 0.80, 0.90, 1.00):
        r_blend = alpha_r * val_neural["cx_rot"] + (1.0 - alpha_r) * val_hgb["hgb_rot"]
        for st_sc in (0.0, 0.20, 0.50, 1.0):
            r_eval = np.where(v_pred_bin, r_blend, st_sc * r_blend)
            ss_res = np.sum((v_rot_t - r_eval) ** 2)
            ss_tot = np.sum((v_rot_t - np.mean(v_rot_t)) ** 2) + 1e-6
            r2 = float(1.0 - ss_res / ss_tot)
            if r2 > best_val_r2:
                best_val_r2 = r2
                best_alpha_rot = float(alpha_r)
                best_st_scale = float(st_sc)

    print(f"[CALIBRATION ON VAL] Weights(CX/MLP/HGB)=({w_cx:.2f}, {w_mlp:.2f}, {w_hgb:.2f}), tau={best_tau:.2f} -> Val BalAcc={v_bal*100:.2f}%, Val AUC={v_auc:.4f}")
    print(f"[CALIBRATION DIR/ROT] LR Mix=({lc:.2f}, {lm:.2f}, {lh:.2f}), tau_lr={best_tau_lr:.2f} (Val Turn P/S={best_lr_acc*100:.1f}%), alpha_rot={best_alpha_rot:.2f}, st_scale={best_st_scale:.2f} (Val R2={best_val_r2:.3f})")

    # ----------------------------------------------------------------------------------------------
    # Final Evaluation on Unseen Test Split (Full Ensemble + Branch Ablations + Frozen UEIKAP)
    # ----------------------------------------------------------------------------------------------
    t_st_p = w_cx * test_neural["cx_p"] + w_mlp * test_neural["mlp_p"] + w_hgb * test_hgb["hgb_p"]
    test_cx_lr = 0.65 * test_neural["cx_lr"] + 0.35 * (test_neural["cx_dir"][:, 2] / (test_neural["cx_dir"][:, 1] + test_neural["cx_dir"][:, 2] + 1e-8))
    t_stbd_cond = lc * test_cx_lr + lm * test_neural["mlp_lr"] + lh * test_hgb["hgb_lr"]
    t_dir_p = build_hierarchical_dir_probs(t_st_p, t_stbd_cond)

    t_rot_raw = best_alpha_rot * test_neural["cx_rot"] + (1.0 - best_alpha_rot) * test_hgb["hgb_rot"]
    t_rot_p = np.where(t_st_p >= best_tau, t_rot_raw, best_st_scale * t_rot_raw)

    full_metrics = evaluate_predictions(
        t_st_p, t_dir_p, t_rot_p,
        test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
        tau=best_tau, p_stbd_cond=t_stbd_cond, tau_lr=best_tau_lr
    )
    full_metrics["calibrated_weights"] = {"convnext": w_cx, "geom_mlp": w_mlp, "geom_hgb": w_hgb}
    full_metrics["latency_ms_per_ship"] = float(test_neural["mean_latency_s"] * 1000.0)
    full_metrics["throughput_fps"] = float(1.0 / (test_neural["mean_latency_s"] + 1e-6))

    results = {"expanded_test": full_metrics}

    # Branch Ablation Audit (All thresholds 0.50 or Val-calibrated cleanly on test)
    results["branch_ablations"] = {
        "convnext_only": evaluate_predictions(
            test_neural["cx_p"], test_neural["cx_dir"], test_neural["cx_rot"],
            test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
            tau=best_tau, p_stbd_cond=test_cx_lr, tau_lr=best_tau_lr
        ),
        "geom_mlp_only": evaluate_predictions(
            test_neural["mlp_p"], test_neural["mlp_dir"], test_hgb["hgb_rot"],
            test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
            tau=0.50, p_stbd_cond=test_neural["mlp_lr"], tau_lr=0.50
        ),
        "geom_hgb_only": evaluate_predictions(
            test_hgb["hgb_p"], build_hierarchical_dir_probs(test_hgb["hgb_p"], test_hgb["hgb_lr"]), test_hgb["hgb_rot"],
            test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
            tau=0.50, p_stbd_cond=test_hgb["hgb_lr"], tau_lr=0.50
        ),
    }

    # Evaluate separately on Original Frozen UEIKAP Test Set if present
    if "source_dataset" in test_df.columns:
        ueikap_mask = (test_df["source_dataset"] == "UEIKAP").values
        if np.any(ueikap_mask):
            ueikap_metrics = evaluate_predictions(
                t_st_p[ueikap_mask], t_dir_p[ueikap_mask], t_rot_p[ueikap_mask],
                test_ds.state_labels[ueikap_mask], test_ds.dir_labels[ueikap_mask], test_ds.rot_values[ueikap_mask],
                tau=best_tau, p_stbd_cond=t_stbd_cond[ueikap_mask], tau_lr=best_tau_lr
            )
            results["ueikap_frozen_test"] = ueikap_metrics

    # ----------------------------------------------------------------------------------------------
    # Epistemic Uncertainty Quantification (N=30 MC-Dropout Passes) & Selective Rejection Curve
    # ----------------------------------------------------------------------------------------------
    w_neural_sum = max(1e-6, w_cx + w_mlp)
    print("\n[UNCERTAINTY] Running N=30 MC-Dropout passes across unseen test set...")
    unc_metrics, mc_mean, mc_std, mc_entropy = run_mc_dropout_uncertainty(
        convnext_net, geom_mlp, test_loader, test_ds.state_labels, tau=best_tau, w_cx=(w_cx / w_neural_sum), num_passes=30
    )
    results["mc_dropout_uncertainty"] = unc_metrics

    # ----------------------------------------------------------------------------------------------
    # Hydrodynamic Explainability: Physics/Radon Feature Permutation Importance
    # ----------------------------------------------------------------------------------------------
    print("[EXPLAINABILITY] Computing permutation importance across 12 Wake/Radon features...")
    imp_df = compute_physics_permutation_importance(
        convnext_net, geom_mlp, test_ds, full_metrics["balanced_accuracy"], tau=best_tau, w_cx=(w_cx / w_neural_sum), batch_size=batch_size
    )
    results["top_physics_features"] = imp_df.head(5).to_dict("records")

    print("\n" + "=" * 80)
    print("FINAL PUBLICATION BENCHMARK SUMMARY (CALIBRATED ON VAL, FROZEN ON TEST)")
    print("=" * 80)
    print(json.dumps(results, indent=2))

    with open(os.path.join(CHECKPOINTS_DIR, "master_evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save coherent hierarchical test predictions parquet
    pred_out = test_df[["sample_id", "source_dataset", "state", "direction"]].copy() if "source_dataset" in test_df.columns else test_df[["sample_id", "state", "direction"]].copy()
    pred_out["prob_turn_ensemble"] = t_st_p
    pred_out["prob_starboard_cond"] = t_stbd_cond
    pred_out["mc_mean_prob"] = mc_mean
    pred_out["epistemic_std"] = mc_std
    pred_out["predictive_entropy"] = mc_entropy
    pred_bin_out = (t_st_p >= best_tau)
    pred_out["pred_state"] = np.where(pred_bin_out, "Turning", "Straight")
    pred_out["pred_direction"] = np.where(~pred_bin_out, "None", np.where(t_stbd_cond >= best_tau_lr, "Starboard", "Port"))
    pred_out["pred_rot_deg_min"] = t_rot_p
    pred_out.to_parquet(os.path.join(DATA_DIR, "processed", "expanded_test_predictions.parquet"), index=False)

    # Generate Before-vs-After Lee Filter Grad-CAM + Wake Overlap Audit across ALL test images (-1)
    try:
        generate_lee_vs_raw_gradcam(convnext_net, test_ds, GRADCAM_DIR, num_per_class=gradcam_per_class)
        print(f"[GRAD-CAM] Saved comparative visualizations & wake overlap CSV in {GRADCAM_DIR} and {TABLES_DIR}")
    except Exception as e:
        print(f"[WARNING] Grad-CAM visualization skipped: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Branch ConvNeXt + Lee + Geometry Ensemble")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--gradcam_per_class", type=int, default=-1, help="-1 generates Grad-CAM for ALL test images")
    parser.add_argument("--allow_partial_data", action="store_true", help="Allow running without full MMDEC crops")
    args = parser.parse_args()
    run_master_suite(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, img_size=args.img_size, gradcam_per_class=args.gradcam_per_class, allow_partial_data=args.allow_partial_data)
