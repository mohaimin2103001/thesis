"""
====================================================================================================
KAGGLE_MASTER_EXPANDED_SUITE.py
End-to-End Master Training & Evaluation Pipeline for Expanded MMDEC + UEIKAP Benchmark
====================================================================================================
Anti-Overfitting + High Turning/Straight Accuracy Suite:
  1. Unified Domain-Invariant SAR Preprocessing + Per-Chip Standardization
  2. Dual-Scale Stern-Anchored Canonical Wake-Up Vision (GPU Affine Reflection-Padded Grid):
       - View 1: Full Macro 6.4 km Context View (7x7 Lee + CLAHE + Wake Spotlight)
       - View 2: Stern-Anchored Wake-Up Aligned View (ship stern near bottom, FULL 0-4km wake plume
         stretched vertically upward -> ~2x wake curvature resolution with zero wasted bow water)
  3. Residual Spatial Wake-Attention Pooling + Auxiliary Mask Consistency Loss
  4. Anti-Overfitting Stack: EMA weights (decay=0.995), Label Smoothing (0.04), DropPath (0.30),
     Weight Decay (2e-2), Early Stopping (patience=5), and Automated Train-vs-Val Overfit Audit
  5. Heterogeneous 3-Family Ensemble: ConvNeXt-Small + SwinV2-Tiny + ResNet50d + GeomHGB/MLP
     evaluated with Full 8x Chiral D4 Test-Time Augmentation
====================================================================================================
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import math
import time
import json
import copy
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
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from geometry.wake_geometry import extract_wake_features_from_mask, estimate_sar_streak_angle_deg
from wake.train_wake_net import extract_unified_sar_gray

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

PHYSICS_FEATURES = [
    "wake_length", "wake_orientation", "mean_curvature", "median_curvature",
    "max_abs_curvature", "signed_mean_curvature", "curvature_std",
    "wake_asymmetry", "mask_area", "geometry_confidence",
    "radon_angle_deg", "radon_deflection_deg",
    "pca_span", "pca_width", "pca_abs_bend", "pca_signed_bend",
    "pca_bend_px", "pca_lin_quad_gain", "pca_abs_defl_deg", "pca_signed_defl_deg",
]

GEOMETRY_FEATURES = [
    "mean_curvature", "median_curvature", "max_abs_curvature",
    "signed_mean_curvature", "curvature_std", "wake_asymmetry",
    "radon_deflection_deg",
    "pca_span", "pca_width", "pca_abs_bend", "pca_signed_bend",
    "pca_bend_px", "pca_lin_quad_gain", "pca_abs_defl_deg", "pca_signed_defl_deg",
]

CHIRAL_PHYS_NAMES = (
    "signed_mean_curvature", "wake_asymmetry", "radon_deflection_deg",
    "pca_signed_bend", "pca_signed_defl_deg",
)
CHIRAL_PHYS_INDICES = [PHYSICS_FEATURES.index(c) for c in CHIRAL_PHYS_NAMES]
IDX_WAKE_ORIENT = PHYSICS_FEATURES.index("wake_orientation")
IDX_RADON_ANGLE = PHYSICS_FEATURES.index("radon_angle_deg")


def build_enriched_tabular_features(phys_raw):
    """Derives 29 pure SAR/Wake tabular features from 20 observable geometry/Radon/PCA descriptors."""
    p = np.asarray(phys_raw, dtype=np.float32)
    wake_len = p[:, 0:1]
    med_c = p[:, 3:4]
    max_c = p[:, 4:5]
    s_curv = p[:, 5:6]
    asym = p[:, 7:8]
    area = p[:, 8:9]
    r_defl = p[:, 11:12]
    pca_span = p[:, 12:13]
    pca_width = p[:, 13:14]
    pca_bend_px = p[:, 16:17]
    pca_lgain = p[:, 17:18]

    derived = np.hstack([
        np.abs(s_curv),
        np.abs(asym),
        np.abs(r_defl),
        p[:, 2:3] * wake_len,
        wake_len / (np.sqrt(np.maximum(area, 0.0)) + 1.0),
        max_c / (med_c + 1e-4),
        (area >= 50.0).astype(np.float32),
        pca_width / (pca_span + 1.0),
        pca_bend_px * pca_lgain,
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
        os.path.join(DATA_DIR, "interim", "mmdec_masks"),
        os.path.join(DATA_DIR, "interim", "ueikap_masks"),
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


class LeeSpeckleFilter(nn.Module):
    """Differentiable FP32 7x7 Lee filter suppressing multiplicative SAR sea clutter speckle."""
    def __init__(self, kernel_size=7, noise_var=0.08):
        super().__init__()
        self.pad = kernel_size // 2
        self.noise_var = noise_var
        box_weight = torch.full((1, 1, kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=torch.float32)
        self.register_buffer("box_weight", box_weight)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            w = self.box_weight.float()
            mean = F.conv2d(xf, w, padding=self.pad)
            sq_mean = F.conv2d(xf * xf, w, padding=self.pad)
            var = torch.clamp(sq_mean - mean * mean, min=1e-6)
            k = torch.clamp((var - (mean * mean) * self.noise_var) / var, min=0.0, max=1.0)
            return torch.clamp(mean + k * (xf - mean), 0.0, 1.0)


def _robust_chip_norm(ch_4d):
    """Per-chip intensity standardization neutralizing UEIKAP vs MMDEC global brightness shift."""
    mu = ch_4d.mean(dim=(2, 3), keepdim=True)
    sd = ch_4d.std(dim=(2, 3), keepdim=True) + 0.04
    return torch.clamp((ch_4d - mu) / sd, -3.0, 3.0) / 6.0 + 0.5


class DualScaleWakeNet(nn.Module):
    """
    Upgraded Multi-Scale Hydrodynamic Vision Architecture:
      - View 1 (Macro): centered full context view (7x7 Lee + CLAHE + Wake Spotlight)
      - View 2 (Stern-Anchored Wake-Up View): rotates wake axis vertically upward (-Y) AND
        shifts center along wake tail (+0.26 along align_rad) at scale=0.56 so the hull sits near
        the bottom edge and the FULL trailing wake plume stretches vertically across View 2!
      - Spatial Wake-Attention Pooling + Auxiliary Mask Supervision Map (Macro)
    """
    def __init__(self, backbone_name="convnext_small", num_physics_features=20, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.speckle_filter = LeeSpeckleFilter(kernel_size=7, noise_var=0.08)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.30,
            drop_path_rate=0.30,
        )
        embed_dim = int(self.backbone.num_features)
        self.embed_dim = embed_dim

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.spatial_attn = nn.Conv2d(embed_dim, 1, kernel_size=1)

        self.physics_mlp = nn.Sequential(
            nn.Linear(num_physics_features, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.SiLU(),
        )

        self.shared_head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 64, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.35),
        )

        self.binary_head = nn.Linear(192, 1)
        self.direction_head = nn.Linear(192, 3)
        self.turn_lr_head = nn.Linear(192, 1)
        self.rot_head = nn.Linear(192, 1)

    def _extract_spatial_map(self, x_3ch):
        fmap = self.backbone.forward_features(x_3ch)
        if fmap.ndim == 4 and fmap.shape[1] != self.embed_dim and fmap.shape[-1] == self.embed_dim:
            fmap = fmap.permute(0, 3, 1, 2).contiguous()
        return fmap

    def _build_aligned_zoom(self, x_macro, align_rad):
        bsz = x_macro.shape[0]
        ang = align_rad.to(dtype=torch.float32, device=x_macro.device)
        alpha = ang + (math.pi * 0.5)
        if self.training:
            ang = ang + (torch.rand(bsz, device=x_macro.device) * 0.16 - 0.08)
            alpha = ang + (math.pi * 0.5)
            scale = 0.52 + 0.10 * torch.rand(bsz, device=x_macro.device)
            shift = torch.full((bsz,), 0.26, device=x_macro.device) + (torch.rand(bsz, device=x_macro.device) * 0.06 - 0.03)
        else:
            scale = torch.full((bsz,), 0.56, device=x_macro.device)
            shift = torch.full((bsz,), 0.26, device=x_macro.device)

        cos_a = torch.cos(alpha) * scale
        sin_a = torch.sin(alpha) * scale
        # Translate window center along wake vector (cos(ang), sin(ang)) so stern anchors near bottom!
        tx = shift * torch.cos(ang)
        ty = shift * torch.sin(ang)
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, tx], dim=1),
            torch.stack([sin_a, cos_a, ty], dim=1),
        ], dim=1)

        with torch.amp.autocast("cuda", enabled=False):
            grid = F.affine_grid(theta, x_macro.size(), align_corners=False)
            x_zoom = F.grid_sample(x_macro.float(), grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        return x_zoom.to(dtype=x_macro.dtype)

    def extract_visual_and_attn(self, x_img, align_rad):
        log_ch = x_img[:, 0:1].float()
        clahe_ch = x_img[:, 1:2].float() if x_img.shape[1] >= 3 else log_ch
        mask_ch = x_img[:, 2:3].float() if x_img.shape[1] >= 3 else x_img[:, 1:2].float()

        lee_ch = self.speckle_filter(log_ch)
        lee_n = _robust_chip_norm(lee_ch)
        clahe_n = _robust_chip_norm(clahe_ch)
        spot_ch = torch.clamp(0.60 * clahe_n + 0.40 * mask_ch, 0.0, 1.0)

        x_raw3 = torch.cat([lee_n, clahe_n, spot_ch], dim=1)
        x_macro = (x_raw3 - self.img_mean) / self.img_std
        x_zoom = self._build_aligned_zoom(x_macro, align_rad)

        bsz = x_macro.shape[0]
        x_both = torch.cat([x_macro, x_zoom], dim=0)
        fmap_both = self._extract_spatial_map(x_both)

        attn_logits = self.spatial_attn(fmap_both)
        attn_probs = torch.softmax(attn_logits.flatten(2), dim=-1).view_as(attn_logits)

        v_gap = fmap_both.mean(dim=(2, 3))
        v_wake = (fmap_both * attn_probs).sum(dim=(2, 3))
        v_both = 0.5 * (v_gap + v_wake)

        v_vis = torch.cat([v_both[:bsz], v_both[bsz:]], dim=1)
        return v_vis, attn_logits[:bsz]

    def forward(self, x_img, x_phys, align_rad, return_attn=False):
        v_vis, attn_macro = self.extract_visual_and_attn(x_img, align_rad)
        p_emb = self.physics_mlp(x_phys)
        feat = self.shared_head(torch.cat([v_vis, p_emb], dim=1))
        out = (
            self.binary_head(feat).squeeze(1),
            self.direction_head(feat),
            self.rot_head(feat).squeeze(1),
            self.turn_lr_head(feat).squeeze(1),
        )
        if return_attn:
            return (*out, attn_macro)
        return out


class GeometrySpecialistMLP(nn.Module):
    """Branch B1: Regularized Wake Curvature & PCA Quadratic Bend Specialist MLP."""
    def __init__(self, in_features=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(48, 1)
        self.direction_head = nn.Linear(48, 3)
        self.turn_lr_head = nn.Linear(48, 1)

    def forward(self, x_geom):
        h = self.net(x_geom)
        return self.binary_head(h).squeeze(1), self.direction_head(h), self.turn_lr_head(h).squeeze(1)


class ScaledManeuverDataset(Dataset):
    def __init__(self, df, phys_mean=None, phys_std=None, img_size=(256, 256), augment=False, rot_clip=12.0, preload=True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if len(self.df) > 0 and "pca_abs_bend" not in self.df.columns:
            self._backfill_pca_columns()

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
        self.chiral_indices = list(CHIRAL_PHYS_INDICES)

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

        self.cached_imgs = None
        self.align_rads = np.full(len(self.df), -math.pi * 0.5, dtype=np.float32)
        est_bytes = len(self.df) * 3 * int(img_size[0]) * int(img_size[1])
        if preload and len(self.df) > 0 and est_bytes <= 2_400_000_000:
            cache = np.empty((len(self.df), 3, int(img_size[1]), int(img_size[0])), dtype=np.uint8)
            for idx in range(len(self.df)):
                arr, ang_rad = self._load_item_u8_and_angle(idx)
                cache[idx] = arr
                self.align_rads[idx] = ang_rad
            self.cached_imgs = cache
        else:
            for idx in range(len(self.df)):
                self.align_rads[idx] = self._resolve_align_rad(idx, base_u8=None)

    def _backfill_pca_columns(self):
        pca_keys = (
            "pca_span", "pca_width", "pca_abs_bend", "pca_signed_bend",
            "pca_bend_px", "pca_lin_quad_gain", "pca_abs_defl_deg",
            "pca_signed_defl_deg", "pca_axis_angle_deg",
        )
        rows = []
        for _, r in self.df.iterrows():
            mp = resolve_path(str(r.get("wake_mask_path", "")))
            sp = (float(r["ship_x"]), float(r["ship_y"])) if pd.notna(r.get("ship_x", np.nan)) else (256.0, 256.0)
            g = extract_wake_features_from_mask(mp, ship_point=sp) if os.path.exists(mp) else None
            rows.append({k: float(g.get(k, 0.0)) for k in pca_keys} if g is not None else {k: 0.0 for k in pca_keys})
        pdf = pd.DataFrame(rows)
        for col in pdf.columns:
            self.df[col] = pdf[col].values

    def _resolve_align_rad(self, idx, base_u8=None):
        row = self.df.iloc[idx]
        if float(row.get("mask_area", 0.0)) >= 50.0:
            ang = row.get("pca_axis_angle_deg", np.nan)
            if pd.notna(ang) and abs(float(ang)) > 1e-4:
                return float(np.radians(float(ang)))
            wo = float(row.get("wake_orientation", -90.0))
            if abs(wo) > 1e-4:
                return float(np.radians(wo))
        ang_fb = row.get("img_axis_angle_deg", np.nan)
        if pd.notna(ang_fb) and abs(float(ang_fb)) > 1e-4:
            return float(np.radians(float(ang_fb)))
        if base_u8 is not None:
            return float(np.radians(estimate_sar_streak_angle_deg(base_u8)))
        return -math.pi * 0.5

    def _load_item_u8_and_angle(self, idx):
        row = self.df.iloc[idx]
        p_crop = resolve_path(str(row.get("crop_path", "")))
        p_orig = resolve_path(str(row.get("original_image_path", "")))
        img_p = p_crop if os.path.exists(p_crop) else p_orig

        bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not load SAR chip for sample {row.get('sample_id')}: {img_p}")
        bgr_res = cv2.resize(bgr, self.img_size)
        base_u8 = extract_unified_sar_gray(bgr_res)
        clahe_u8 = self.clahe.apply(base_u8)

        m_path = resolve_path(str(row.get("wake_mask_path", "")))
        if os.path.exists(m_path):
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(m_img, self.img_size, interpolation=cv2.INTER_NEAREST) if m_img is not None else np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        mask_blur = cv2.GaussianBlur((mask > 127).astype(np.float32), (7, 7), 2.0)
        mask_u8 = np.clip(mask_blur * 255.0, 0.0, 255.0).astype(np.uint8)
        ang_rad = self._resolve_align_rad(idx, base_u8=base_u8)
        return np.stack([base_u8, clahe_u8, mask_u8], axis=0), ang_rad

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cached_imgs is not None:
            arr_u8 = self.cached_imgs[idx]
            ang_rad = float(self.align_rads[idx])
        else:
            arr_u8, ang_rad = self._load_item_u8_and_angle(idx)
        img_3ch = arr_u8.astype(np.float32) / 255.0

        phys_vec = self.phys_raw[idx].copy()
        dir_val = int(self.dir_labels[idx])
        rot_val = float(self.rot_values[idx])

        if self.augment:
            k_rot = int(np.random.randint(0, 4))
            if k_rot > 0:
                img_3ch = np.rot90(img_3ch, k_rot, axes=(1, 2)).copy()
                ang_rad = ((ang_rad - k_rot * (math.pi * 0.5) + math.pi) % (2.0 * math.pi)) - math.pi
                phys_vec[IDX_WAKE_ORIENT] = ((phys_vec[IDX_WAKE_ORIENT] - 90.0 * k_rot + 180.0) % 360.0) - 180.0
                phys_vec[IDX_RADON_ANGLE] = (phys_vec[IDX_RADON_ANGLE] + 90.0 * k_rot) % 180.0

            if np.random.rand() < 0.5:
                img_3ch = img_3ch[:, :, ::-1].copy()
                ang_rad = ((math.pi - ang_rad + math.pi) % (2.0 * math.pi)) - math.pi
                phys_vec[IDX_WAKE_ORIENT] = ((180.0 - phys_vec[IDX_WAKE_ORIENT] + 180.0) % 360.0) - 180.0
                phys_vec[IDX_RADON_ANGLE] = (180.0 - phys_vec[IDX_RADON_ANGLE]) % 180.0
                if dir_val == 1:
                    dir_val = 2
                elif dir_val == 2:
                    dir_val = 1
                rot_val = -rot_val
                for ci in self.chiral_indices:
                    phys_vec[ci] = -phys_vec[ci]

            if np.random.rand() < 0.45:
                gamma = np.random.uniform(0.88, 1.12)
                img_3ch[:2] = np.clip(img_3ch[:2] * gamma, 0.0, 1.0)
            # Anti-overfitting multiplicative speckle perturbation (preserves real ocean physics)
            if np.random.rand() < 0.35:
                speckle = 1.0 + np.random.normal(0.0, 0.025, size=img_3ch[:2].shape).astype(np.float32)
                img_3ch[:2] = np.clip(img_3ch[:2] * speckle, 0.0, 1.0)

        phys_norm_vec = np.clip((phys_vec - self.phys_mean) / self.phys_std, -8.0, 8.0).astype(np.float32)
        geom_norm_vec = phys_norm_vec[self.geom_indices]

        return (
            torch.from_numpy(img_3ch),
            torch.from_numpy(phys_norm_vec),
            torch.from_numpy(geom_norm_vec),
            torch.tensor(ang_rad, dtype=torch.float32),
            torch.tensor(self.state_labels[idx], dtype=torch.float32),
            torch.tensor(dir_val, dtype=torch.long),
            torch.tensor(rot_val, dtype=torch.float32),
        )


class PhysicsInformedMultiTaskLoss(nn.Module):
    def __init__(self, pos_weight=1.35, label_smooth=0.04, dir_weights=None, lambda_dir=0.45, lambda_lr=0.40, lambda_rot=0.20, lambda_phys=0.10, lambda_spatial=0.25):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smooth = label_smooth
        if dir_weights is not None:
            self.ce_dir = nn.CrossEntropyLoss(weight=torch.tensor(dir_weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
        else:
            self.ce_dir = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.bce_lr = nn.BCEWithLogitsLoss()
        self.lambda_dir = lambda_dir
        self.lambda_lr = lambda_lr
        self.lambda_rot = lambda_rot
        self.lambda_phys = lambda_phys
        self.lambda_spatial = lambda_spatial

    def forward(self, state_logit, dir_logits, rot_pred, lr_logit, attn_macro, imgs, state_t, dir_t, rot_t):
        # Anti-overfit smooth target [0.04, 0.96] + focal margin + ROT clarity weighting
        st_smooth = state_t * (1.0 - 2.0 * self.label_smooth) + self.label_smooth
        bce_raw = F.binary_cross_entropy_with_logits(
            state_logit, st_smooth,
            pos_weight=torch.tensor([self.pos_weight], device=state_logit.device, dtype=state_logit.dtype),
            reduction="none",
        )
        p_t = torch.sigmoid(state_logit) * state_t + (1.0 - torch.sigmoid(state_logit)) * (1.0 - state_t)
        focal_mod = torch.clamp((1.0 - p_t.detach()) ** 1.0, min=0.25, max=1.0)
        rot_clarity = torch.where(state_t > 0.5, torch.clamp(0.75 + 0.10 * torch.abs(rot_t), 0.75, 1.40), torch.ones_like(rot_t))
        l_state = torch.mean(bce_raw * focal_mod * rot_clarity)

        l_dir = self.ce_dir(dir_logits, dir_t)
        turn_mask = state_t > 0.5
        if torch.any(turn_mask):
            lr_target = (dir_t[turn_mask] == 2).float() * 0.90 + 0.05
            l_lr = self.bce_lr(lr_logit[turn_mask], lr_target)
        else:
            l_lr = state_logit.sum() * 0.0

        rot_weights = 1.0 + 1.5 * state_t
        l_rot = torch.mean(rot_weights * F.smooth_l1_loss(rot_pred, rot_t, reduction="none", beta=0.75))
        l_straight_zero = torch.mean((1.0 - state_t) * (rot_pred ** 2)) * 0.25
        l_phys = torch.mean(state_t * F.relu(-rot_pred * torch.sign(rot_t)))

        mask_full = imgs[:, 2:3].float()
        has_mask = mask_full.flatten(1).sum(dim=1) > 15.0
        if torch.any(has_mask):
            m_ds = F.adaptive_avg_pool2d(mask_full[has_mask], attn_macro.shape[-2:])
            m_tgt = torch.clamp(0.10 + 0.85 * (m_ds / (m_ds.amax(dim=(2, 3), keepdim=True) + 1e-4)), 0.10, 0.95)
            l_spatial = F.binary_cross_entropy_with_logits(attn_macro[has_mask], m_tgt)
        else:
            l_spatial = state_logit.sum() * 0.0

        return (
            l_state
            + self.lambda_dir * l_dir
            + self.lambda_lr * l_lr
            + self.lambda_rot * (l_rot + l_straight_zero)
            + self.lambda_phys * l_phys
            + self.lambda_spatial * l_spatial
        )


def build_hierarchical_dir_probs(state_probs, p_stbd_cond):
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

    if p_stbd_cond is None:
        denom = dir_probs[:, 1] + dir_probs[:, 2] + 1e-8
        p_stbd_cond = dir_probs[:, 2] / denom
    hier_pred = np.where(pred_bin == 0, 0, np.where(p_stbd_cond >= tau_lr, 2, 1))
    softmax_pred = np.argmax(dir_probs, axis=1)

    dir_overall = float(np.mean(hier_pred == dir_true))
    c_accs = [float(np.mean(hier_pred[dir_true == c] == c)) if np.any(dir_true == c) else 0.0 for c in (0, 1, 2)]
    sm_accs = [float(np.mean(softmax_pred[dir_true == c] == c)) if np.any(dir_true == c) else 0.0 for c in (0, 1, 2)]
    turn_mask = (dir_true == 1) | (dir_true == 2)
    turn_dir_acc = float(np.mean(np.where(p_stbd_cond[turn_mask] >= tau_lr, 2, 1) == dir_true[turn_mask])) if np.any(turn_mask) else 0.0

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
        "dir_3class_balanced_acc": float(np.mean(c_accs)),
        "dir_straight_acc": c_accs[0],
        "dir_port_acc": c_accs[1],
        "dir_starboard_acc": c_accs[2],
        "softmax_dir_3class_overall_acc": float(np.mean(softmax_pred == dir_true)),
        "softmax_dir_3class_balanced_acc": float(np.mean(sm_accs)),
        "turn_port_vs_starboard_acc": turn_dir_acc,
        "rot_mae_deg_min": rot_mae,
        "rot_r2": rot_r2,
    }


def _transform_batch_torch(imgs, phys_norm, align_rad, phys_mean_t, phys_std_t, geom_indices, k_rot=0, hflip=False):
    im = imgs
    ang = align_rad.clone()
    phys_raw = phys_norm * phys_std_t + phys_mean_t

    if k_rot > 0:
        im = torch.rot90(im, k_rot, dims=[2, 3])
        ang = torch.remainder(ang - k_rot * (math.pi * 0.5) + math.pi, 2.0 * math.pi) - math.pi
        phys_raw[:, IDX_WAKE_ORIENT] = torch.remainder(phys_raw[:, IDX_WAKE_ORIENT] - 90.0 * k_rot + 180.0, 360.0) - 180.0
        phys_raw[:, IDX_RADON_ANGLE] = torch.remainder(phys_raw[:, IDX_RADON_ANGLE] + 90.0 * k_rot, 180.0)

    if hflip:
        im = torch.flip(im, dims=[3])
        ang = torch.remainder((math.pi - ang) + math.pi, 2.0 * math.pi) - math.pi
        phys_raw[:, IDX_WAKE_ORIENT] = torch.remainder((180.0 - phys_raw[:, IDX_WAKE_ORIENT]) + 180.0, 360.0) - 180.0
        phys_raw[:, IDX_RADON_ANGLE] = torch.remainder(180.0 - phys_raw[:, IDX_RADON_ANGLE], 180.0)
        for ci in CHIRAL_PHYS_INDICES:
            phys_raw[:, ci] = -phys_raw[:, ci]

    phys_t = torch.clamp((phys_raw - phys_mean_t) / phys_std_t, -8.0, 8.0)
    geom_t = phys_t[:, geom_indices]
    return im, phys_t, geom_t, ang


def predict_neural_branches(vision_nets, geom_mlp, loader, phys_mean_np, phys_std_np, geom_indices, tta_mode="8x"):
    """
    Runs Val/Test inference across any list of vision backbones + GeomMLP
    with Full 8x D4 Chiral TTA (4 rotations x 2 mirror states) or 4x TTA.
    """
    for net in vision_nets:
        net.eval()
    geom_mlp.eval()

    phys_mean_t = torch.tensor(phys_mean_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    phys_std_t = torch.tensor(phys_std_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    if tta_mode == "8x":
        tta_ops = [(k, hf) for k in range(4) for hf in (False, True)]
    elif tta_mode == "4x":
        tta_ops = [(0, False), (0, True), (1, False), (2, False)]
    else:
        tta_ops = [(0, False)]

    per_net = [{"p": [], "dir": [], "lr": [], "rot": []} for _ in vision_nets]
    g_p, g_dir, g_lr = [], [], []
    times = []

    def _run_tta(net, im_b, ph_b, ang_b):
        p_s, d_s, lr_s, r_s = 0.0, 0.0, 0.0, 0.0
        for k_r, hf in tta_ops:
            im_t, ph_t, _, an_t = _transform_batch_torch(im_b, ph_b, ang_b, phys_mean_t, phys_std_t, geom_indices, k_rot=k_r, hflip=hf)
            st_l, dir_l, rot_l, lr_l = net(im_t, ph_t, an_t)
            pv, dv, lrv = torch.sigmoid(st_l), torch.softmax(dir_l, dim=1), torch.sigmoid(lr_l)
            if hf:
                dv = dv[:, [0, 2, 1]]
                lrv = 1.0 - lrv
                rot_l = -rot_l
            p_s = p_s + pv; d_s = d_s + dv; lr_s = lr_s + lrv; r_s = r_s + rot_l
        inv = 1.0 / float(len(tta_ops))
        return p_s * inv, d_s * inv, lr_s * inv, r_s * inv

    with torch.no_grad():
        for imgs, phys, geom, ang, _, _, _ in loader:
            imgs, phys, geom, ang = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE), ang.to(DEVICE)
            t0 = time.time()

            for ni, net in enumerate(vision_nets):
                pv, dv, lrv, rv = _run_tta(net, imgs, phys, ang)
                per_net[ni]["p"].append(pv.cpu().numpy())
                per_net[ni]["dir"].append(dv.cpu().numpy())
                per_net[ni]["lr"].append(lrv.cpu().numpy())
                per_net[ni]["rot"].append(rv.cpu().numpy())

            gst, gdir, glr = geom_mlp(geom)
            pg, dg, lrg = torch.sigmoid(gst), torch.softmax(gdir, dim=1), torch.sigmoid(glr)
            if tta_mode != "1x":
                _, _, geom_f, _ = _transform_batch_torch(imgs, phys, ang, phys_mean_t, phys_std_t, geom_indices, k_rot=0, hflip=True)
                gst_f, gdir_f, glr_f = geom_mlp(geom_f)
                pg = 0.5 * (pg + torch.sigmoid(gst_f))
                dg = 0.5 * (dg + torch.softmax(gdir_f, dim=1)[:, [0, 2, 1]])
                lrg = 0.5 * (lrg + (1.0 - torch.sigmoid(glr_f)))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.time() - t0) / len(imgs))

            g_p.append(pg.cpu().numpy()); g_dir.append(dg.cpu().numpy()); g_lr.append(lrg.cpu().numpy())

    vis_out = [{k: np.concatenate(v) for k, v in d.items()} for d in per_net]
    return {
        "vis": vis_out,
        "mlp_p": np.concatenate(g_p),
        "mlp_dir": np.concatenate(g_dir),
        "mlp_lr": np.concatenate(g_lr),
        "mean_latency_s": float(np.mean(times)),
    }


def train_and_predict_hgb(train_ds, eval_datasets):
    """Branch B2: Enriched 29-D Wake/PCA/Radon HistGradientBoosting Specialist."""
    X_tr = build_enriched_tabular_features(train_ds.phys_raw)
    y_st_tr = train_ds.state_labels
    y_rot_tr = train_ds.rot_values
    turn_mask_tr = y_st_tr == 1
    y_lr_tr = (train_ds.dir_labels[turn_mask_tr] == 2).astype(int)

    clf_state = HistGradientBoostingClassifier(
        max_iter=220, max_depth=3, min_samples_leaf=30, learning_rate=0.04,
        l2_regularization=6.0, class_weight="balanced", random_state=42,
    )
    clf_state.fit(X_tr, y_st_tr)

    clf_lr = HistGradientBoostingClassifier(
        max_iter=160, max_depth=3, min_samples_leaf=20, learning_rate=0.04,
        l2_regularization=6.0, class_weight="balanced", random_state=42,
    )
    has_lr = np.sum(turn_mask_tr) >= 10 and len(np.unique(y_lr_tr)) == 2
    if has_lr:
        clf_lr.fit(X_tr[turn_mask_tr], y_lr_tr)

    reg_rot = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=200, max_depth=3, min_samples_leaf=25,
        learning_rate=0.04, l2_regularization=6.0, random_state=42,
    )
    reg_rot.fit(X_tr, y_rot_tr)

    outputs = []
    for ds in eval_datasets:
        raw_orig = ds.phys_raw
        raw_flip = raw_orig.copy()
        raw_flip[:, IDX_WAKE_ORIENT] = ((180.0 - raw_flip[:, IDX_WAKE_ORIENT] + 180.0) % 360.0) - 180.0
        raw_flip[:, IDX_RADON_ANGLE] = (180.0 - raw_flip[:, IDX_RADON_ANGLE]) % 180.0
        for ci in CHIRAL_PHYS_INDICES:
            raw_flip[:, ci] = -raw_flip[:, ci]
        X_e = build_enriched_tabular_features(raw_orig)
        X_f = build_enriched_tabular_features(raw_flip)

        p_st = 0.5 * (clf_state.predict_proba(X_e)[:, 1] + clf_state.predict_proba(X_f)[:, 1])
        p_lr = 0.5 * (clf_lr.predict_proba(X_e)[:, 1] + (1.0 - clf_lr.predict_proba(X_f)[:, 1])) if has_lr else np.full(len(ds), 0.5, dtype=np.float32)
        rot_e = 0.5 * (reg_rot.predict(X_e) - reg_rot.predict(X_f))
        outputs.append({"hgb_p": p_st.astype(np.float32), "hgb_lr": p_lr.astype(np.float32), "hgb_rot": rot_e.astype(np.float32)})
    return outputs


def run_mc_dropout_uncertainty(primary_net, geom_mlp, test_loader, state_true, tau=0.50, w_cx=0.80, num_passes=30):
    """Runs N=30 stochastic MC-Dropout passes on cached test tensors and saves Selective Rejection Curve."""
    cached_batches = [(imgs, phys, geom, ang) for imgs, phys, geom, ang, _, _, _ in test_loader]

    primary_net.eval()
    geom_mlp.eval()
    for m in list(primary_net.modules()) + list(geom_mlp.modules()):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    pass_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            p_batch = []
            for imgs, phys, geom, ang in cached_batches:
                imgs, phys, geom, ang = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE), ang.to(DEVICE)
                st_log, _, _, _ = primary_net(imgs, phys, ang)
                g_st_log, _, _ = geom_mlp(geom)
                p_ens = (w_cx * torch.sigmoid(st_log) + (1.0 - w_cx) * torch.sigmoid(g_st_log)).cpu().numpy()
                p_batch.append(p_ens)
            pass_probs.append(np.concatenate(p_batch))

    primary_net.eval()
    geom_mlp.eval()

    mc_matrix = np.stack(pass_probs, axis=0)
    mean_prob = np.mean(mc_matrix, axis=0)
    epistemic_std = np.std(mc_matrix, axis=0)
    eps = 1e-7
    p_clip = np.clip(mean_prob, eps, 1.0 - eps)
    entropy = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))

    preds = (mean_prob >= tau).astype(int)
    correct_mask = preds == state_true
    overall_acc = float(np.mean(correct_mask))
    ent_correct = float(np.mean(entropy[correct_mask])) if np.any(correct_mask) else 0.0
    ent_incorrect = float(np.mean(entropy[~correct_mask])) if np.any(~correct_mask) else 0.0

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
    ax.set_title("Stern-Aligned Multi-Backbone Ensemble: Selective Error Rejection Curve (N=30 MC)", fontsize=11, fontweight="bold")
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


def compute_physics_permutation_importance(primary_net, geom_mlp, test_ds, base_bal_acc, tau=0.50, w_cx=0.80, batch_size=24):
    """Permutation importance across all 20 Wake/Radon/PCA physics features with cached visual embeddings."""
    primary_net.eval()
    geom_mlp.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    v_embs = []
    with torch.no_grad():
        for imgs, _, _, ang, _, _, _ in loader:
            v_vis, _ = primary_net.extract_visual_and_attn(imgs.to(DEVICE), ang.to(DEVICE))
            v_embs.append(v_vis.cpu())
    v_emb_all = torch.cat(v_embs, dim=0).to(DEVICE)

    orig_phys_raw = test_ds.phys_raw.copy()
    geom_indices = [PHYSICS_FEATURES.index(c) for c in GEOMETRY_FEATURES]
    imp_rows = []

    for f_idx, fname in enumerate(PHYSICS_FEATURES):
        perm_raw = orig_phys_raw.copy()
        perm_raw[:, f_idx] = np.random.permutation(perm_raw[:, f_idx])
        perm_norm = np.clip((perm_raw - test_ds.phys_mean) / test_ds.phys_std, -8.0, 8.0).astype(np.float32)
        p_t = torch.from_numpy(perm_norm).to(DEVICE)
        g_t = p_t[:, geom_indices]

        with torch.no_grad():
            p_emb = primary_net.physics_mlp(p_t)
            feat = primary_net.shared_head(torch.cat([v_emb_all, p_emb], dim=1))
            st_log = primary_net.binary_head(feat).squeeze(1)
            g_st_log, _, _ = geom_mlp(g_t)
            p_all = (w_cx * torch.sigmoid(st_log) + (1.0 - w_cx) * torch.sigmoid(g_st_log)).cpu().numpy()

        b = (p_all >= tau).astype(int)
        st_acc = np.mean(b[test_ds.state_labels == 0] == 0)
        tu_acc = np.mean(b[test_ds.state_labels == 1] == 1)
        bal = 0.5 * (st_acc + tu_acc)
        imp_rows.append({"feature": fname, "permuted_bal_acc": float(bal), "importance_drop": float(base_bal_acc - bal)})

    imp_df = pd.DataFrame(imp_rows).sort_values("importance_drop", ascending=False)
    imp_df.to_csv(os.path.join(TABLES_DIR, "convnext_physics_feature_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = imp_df.sort_values("importance_drop", ascending=True)
    ax.barh(plot_df["feature"], plot_df["importance_drop"] * 100, color="#1f77b4")
    ax.set_xlabel("Balanced Accuracy Drop When Permuted (%)", fontsize=10)
    ax.set_title("Hydrodynamic Wake, Radon & PCA Bend Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "convnext_physics_feature_importance.png"), dpi=200)
    plt.close(fig)
    return imp_df


def generate_lee_vs_raw_gradcam(convnext_net, test_ds, out_dir, num_per_class=-1):
    """Generates Raw SAR vs 7x7 Lee Grad-CAM visualizations + quantifies Inside-Wake Attention Concentration."""
    convnext_net.eval()
    target_layer = convnext_net.backbone.stages[-1]
    activations, gradients = [], []

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

    print(f"[GRAD-CAM] Generating Raw vs 7x7 Lee Grad-CAM heatmaps for {len(selected_indices)} test images...")
    audit_rows = []
    for k, idx in enumerate(selected_indices):
        activations.clear()
        gradients.clear()
        img_t, phys_t, _, ang_t, _, _, rot_true = test_ds[idx]
        x_img = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
        x_phys = phys_t.unsqueeze(0).to(DEVICE)
        x_ang = ang_t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            lee_img = convnext_net.speckle_filter(x_img[:, 0:1]).squeeze().cpu().numpy()

        st_logit, _, rot_pred, _ = convnext_net(x_img, x_phys, x_ang)
        convnext_net.zero_grad()
        score = st_logit[0] if torch.sigmoid(st_logit[0]).item() >= 0.5 else -st_logit[0]
        score.backward()

        h_out, w_out = int(img_t.shape[1]), int(img_t.shape[2])
        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_res = cv2.resize(cam, (w_out, h_out))

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
                    wake_energy_ratio = float(np.sum(cam_res[mask_bin]) / (np.sum(cam_res) + 1e-8))

        raw_np = img_t[0].numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_np, cmap="gray"); axes[0].set_title(f"Raw SAR ({row.get('source_dataset', 'SAR')})")
        axes[1].imshow(lee_img, cmap="gray"); axes[1].set_title("7x7 Lee Speckle Filtered")
        axes[2].imshow(lee_img, cmap="gray"); axes[2].imshow(cam_res, cmap="jet", alpha=0.45)
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
            "inside_wake_energy_ratio": wake_energy_ratio,
        })

    h1.remove()
    h2.remove()
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(os.path.join(TABLES_DIR, "convnext_gradcam_analysis.csv"), index=False)


def _train_vision_backbone(model, model_tag, ckpt_name, train_loader, val_loader, train_ds, val_ds, epochs, lr, backbone_lr_mult=0.28, patience=5):
    """
    AMP cosine training loop with Exponential Moving Average (EMA, decay=0.995),
    Early Stopping (patience=5), and Train-vs-Val Overfit Gap Audit.
    """
    dir_weights = np.array([0.85, 1.15, 1.15], dtype=np.float32)
    criterion = PhysicsInformedMultiTaskLoss(pos_weight=1.35, label_smooth=0.04, dir_weights=dir_weights).to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * backbone_lr_mult},
        {"params": model.spatial_attn.parameters(), "lr": lr},
        {"params": model.physics_mlp.parameters(), "lr": lr},
        {"params": model.shared_head.parameters(), "lr": lr},
        {"params": model.binary_head.parameters(), "lr": lr},
        {"params": model.direction_head.parameters(), "lr": lr},
        {"params": model.turn_lr_head.parameters(), "lr": lr},
        {"params": model.rot_head.parameters(), "lr": lr},
    ], weight_decay=2e-2)
    warmup_ep = min(3, max(1, epochs // 7))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda e: (e + 1) / warmup_ep if e < warmup_ep else max(0.02, 0.5 * (1.0 + math.cos(math.pi * (e - warmup_ep) / max(1, epochs - warmup_ep)))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ema_model = copy.deepcopy(model).to(DEVICE)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_decay = 0.995

    best_score = -1.0
    best_val_auc = 0.5
    best_train_auc = 0.5
    no_improve = 0
    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        tr_probs_b, tr_true_b = [], []
        for imgs, phys, _, ang, state_t, dir_t, rot_t in train_loader:
            imgs, phys, ang = imgs.to(DEVICE), phys.to(DEVICE), ang.to(DEVICE)
            state_t, dir_t, rot_t = state_t.to(DEVICE), dir_t.to(DEVICE), rot_t.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                st_log, dir_log, rot_out, lr_log, attn_macro = model(imgs, phys, ang, return_attn=True)
                loss = criterion(st_log, dir_log, rot_out, lr_log, attn_macro, imgs, state_t, dir_t, rot_t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            scaler.step(opt)
            scaler.update()

            # EMA parameter update after epoch 1
            with torch.no_grad():
                if ep == 1:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    for ep_p, mp in zip(ema_model.parameters(), model.parameters()):
                        ep_p.data.mul_(ema_decay).add_(mp.data, alpha=(1.0 - ema_decay))
                    for ep_b, mb in zip(ema_model.buffers(), model.buffers()):
                        ep_b.copy_(mb)

            tot_loss += loss.item() * len(state_t)
            tr_probs_b.append(torch.sigmoid(st_log.detach()).cpu().numpy())
            tr_true_b.append(state_t.cpu().numpy())
        sched.step()

        tr_p = np.concatenate(tr_probs_b); tr_y = np.concatenate(tr_true_b)
        tr_auc = float(roc_auc_score(tr_y, tr_p)) if len(np.unique(tr_y)) > 1 else 0.5

        eval_net = ema_model if ep >= 2 else model
        eval_net.eval()
        vp, vdir, vrot = [], [], []
        with torch.no_grad():
            for imgs, phys, _, ang, _, _, _ in val_loader:
                st_l, dir_l, rot_l, _ = eval_net(imgs.to(DEVICE), phys.to(DEVICE), ang.to(DEVICE))
                vp.append(torch.sigmoid(st_l).cpu().numpy())
                vdir.append(torch.softmax(dir_l, dim=1).cpu().numpy())
                vrot.append(rot_l.cpu().numpy())
        vp = np.concatenate(vp); vdir = np.concatenate(vdir); vrot = np.concatenate(vrot)
        vr = evaluate_predictions(vp, vdir, vrot, val_ds.state_labels, val_ds.dir_labels, val_ds.rot_values, tau=0.50)

        # Penalize if train_auc far exceeds val_auc (guards against overfitting!)
        gap = max(0.0, tr_auc - vr["auc_roc"])
        score = 0.70 * vr["auc_roc"] + 0.30 * vr["balanced_accuracy"] - 0.08 * max(0.0, gap - 0.12)

        print(f"[{model_tag}] Ep {ep:02d}/{epochs:02d} | Loss:{tot_loss/max(1,len(train_loader.dataset)):.4f} | "
              f"TrainAUC:{tr_auc:.3f} | ValAUC:{vr['auc_roc']:.4f} (Gap:{tr_auc-vr['auc_roc']:+.3f}) | "
              f"ValBal:{vr['balanced_accuracy']*100:.1f}% (St:{vr['straight_accuracy']*100:.1f}%, Tu:{vr['turning_accuracy']*100:.1f}%)")

        if score > best_score:
            best_score = score
            best_val_auc = vr["auc_roc"]
            best_train_auc = tr_auc
            no_improve = 0
            torch.save(eval_net.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience and ep >= warmup_ep + 4:
                print(f"[{model_tag}] Early stopping at epoch {ep} (no val improvement for {patience} epochs; prevents overfit).")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model, {
        "best_val_auc": float(best_val_auc),
        "train_auc_at_best": float(best_train_auc),
        "train_val_auc_gap": float(best_train_auc - best_val_auc),
    }


def _load_backbone_safe(candidates, num_physics_features, pretrained):
    for bb in candidates:
        if not bb or bb.lower() == "none":
            continue
        try:
            net = DualScaleWakeNet(bb, num_physics_features=num_physics_features, pretrained=pretrained).to(DEVICE)
            return net, bb
        except Exception as e:
            print(f"[WARNING] Could not load {bb} ({e}); trying fallback...")
    return None, None


def run_master_suite(
    epochs=18,
    batch_size=24,
    lr=2e-4,
    img_size=256,
    gradcam_per_class=-1,
    primary_backbone="convnext_small",
    second_backbone="swinv2_tiny_window8_256",
    third_backbone="resnet50d",
    allow_partial_data=False,
    pretrained=True,
):
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

    valid_mask = master_df.apply(
        lambda r: os.path.exists(resolve_path(str(r.get("crop_path", "")))) or os.path.exists(resolve_path(str(r.get("original_image_path", "")))),
        axis=1,
    )
    if valid_mask.sum() < len(master_df):
        missing_n = int(len(master_df) - valid_mask.sum())
        if not allow_partial_data and len(master_df) > 1000 and valid_mask.sum() < 0.5 * len(master_df):
            raise RuntimeError(f"{missing_n} / {len(master_df)} crops are missing. Re-run Stage 2 or pass --allow_partial_data.")
        print(f"[DATA] Filtered {missing_n} missing-image rows; retaining {valid_mask.sum():,} valid chips.")
        master_df = master_df[valid_mask].reset_index(drop=True)

    train_df = master_df[master_df["split"] == "train"].copy()
    val_df = master_df[master_df["split"] == "val"].copy()
    test_df = master_df[master_df["split"] == "test"].copy()
    print(f"[SPLITS] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_ds = ScaledManeuverDataset(train_df, img_size=(img_size, img_size), augment=True)
    val_ds = ScaledManeuverDataset(val_df, phys_mean=train_ds.phys_mean, phys_std=train_ds.phys_std, img_size=(img_size, img_size), augment=False)
    test_ds = ScaledManeuverDataset(test_df, phys_mean=train_ds.phys_mean, phys_std=train_ds.phys_std, img_size=(img_size, img_size), augment=False)

    np.savez(
        os.path.join(CHECKPOINTS_DIR, "feature_scaler.npz"),
        phys_mean=train_ds.phys_mean,
        phys_std=train_ds.phys_std,
        features=np.array(PHYSICS_FEATURES),
    )

    n_st = max(1, int(np.sum(train_ds.state_labels == 0)))
    n_tu = max(1, int(np.sum(train_ds.state_labels == 1)))
    state_bal_w = float(np.clip(n_st / n_tu, 1.0, 2.5))
    dom_w = np.where(train_df["source_dataset"].astype(str).eq("UEIKAP").values, 1.6, 1.0) if "source_dataset" in train_df.columns else np.ones(len(train_ds))
    sample_weights = np.where(train_ds.state_labels == 1, state_bal_w, 1.0) * dom_w
    sampler = WeightedRandomSampler(weights=sample_weights.astype(np.float64), num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=len(train_ds) >= batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    t_start = time.time()
    overfit_audit = {}
    vision_nets = []
    vision_names = []

    p_net, p_name = _load_backbone_safe([primary_backbone, "convnext_tiny"], len(PHYSICS_FEATURES), pretrained)
    p_net, p_audit = _train_vision_backbone(
        p_net, f"Primary-{p_name}", "convnext_multitask_best.pt",
        train_loader, val_loader, train_ds, val_ds, epochs=epochs, lr=lr, backbone_lr_mult=0.25, patience=5,
    )
    overfit_audit[p_name] = p_audit
    vision_nets.append(p_net); vision_names.append(p_name)

    if second_backbone and second_backbone.lower() != "none":
        s_net, s_name = _load_backbone_safe([second_backbone], len(PHYSICS_FEATURES), pretrained)
        if s_net is not None:
            s_net, s_audit = _train_vision_backbone(
                s_net, f"Second-{s_name}", "second_backbone_best.pt",
                train_loader, val_loader, train_ds, val_ds, epochs=epochs, lr=lr, backbone_lr_mult=0.30, patience=5,
            )
            overfit_audit[s_name] = s_audit
            vision_nets.append(s_net); vision_names.append(s_name)

    if third_backbone and third_backbone.lower() != "none":
        r_net, r_name = _load_backbone_safe([third_backbone], len(PHYSICS_FEATURES), pretrained)
        if r_net is not None:
            r_net, r_audit = _train_vision_backbone(
                r_net, f"Third-{r_name}", "third_backbone_best.pt",
                train_loader, val_loader, train_ds, val_ds, epochs=max(2, epochs - 2), lr=lr, backbone_lr_mult=0.35, patience=5,
            )
            overfit_audit[r_name] = r_audit
            vision_nets.append(r_net); vision_names.append(r_name)

    # Train Branch B1: Geometry Specialist MLP
    geom_mlp = GeometrySpecialistMLP(in_features=len(GEOMETRY_FEATURES)).to(DEVICE)
    opt_geom = torch.optim.AdamW(geom_mlp.parameters(), lr=max(lr * 2.5, 4e-4), weight_decay=2e-3)
    sched_geom = torch.optim.lr_scheduler.CosineAnnealingLR(opt_geom, T_max=epochs, eta_min=1e-6)
    dir_weights = np.array([0.85, 1.15, 1.15], dtype=np.float32)
    bce_g = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.30], device=DEVICE))
    ce_gd = nn.CrossEntropyLoss(weight=torch.tensor(dir_weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
    bce_glr = nn.BCEWithLogitsLoss()
    best_g_score = -1.0
    g_ckpt = os.path.join(CHECKPOINTS_DIR, "geometry_mlp_best.pt")

    for _ in range(epochs):
        geom_mlp.train()
        for _, _, geom, _, state_t, dir_t, _ in train_loader:
            geom, state_t, dir_t = geom.to(DEVICE), state_t.to(DEVICE), dir_t.to(DEVICE)
            opt_geom.zero_grad()
            gst, gdir, glr = geom_mlp(geom)
            tm = state_t > 0.5
            l_glr = bce_glr(glr[tm], (dir_t[tm] == 2).float()) if torch.any(tm) else (gst.sum() * 0.0)
            loss_g = bce_g(gst, state_t) + 0.5 * ce_gd(gdir, dir_t) + 0.5 * l_glr
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(geom_mlp.parameters(), max_norm=4.0)
            opt_geom.step()
        sched_geom.step()
        geom_mlp.eval()
        gp = []
        with torch.no_grad():
            for _, _, geom, _, _, _, _ in val_loader:
                gst, _, _ = geom_mlp(geom.to(DEVICE))
                gp.append(torch.sigmoid(gst).cpu().numpy())
        gp = np.concatenate(gp)
        g_auc = float(roc_auc_score(val_ds.state_labels, gp)) if len(np.unique(val_ds.state_labels)) > 1 else 0.5
        if g_auc > best_g_score:
            best_g_score = g_auc
            torch.save(geom_mlp.state_dict(), g_ckpt)

    geom_mlp.load_state_dict(torch.load(g_ckpt, map_location=DEVICE))
    geom_mlp.eval()
    print(f"\n[TRAINING COMPLETE] Total GPU Training Elapsed: {(time.time() - t_start)/60.0:.2f} min")

    val_hgb, test_hgb = train_and_predict_hgb(train_ds, [val_ds, test_ds])

    # Run Full 8x Chiral D4 TTA across all trained EMA vision models
    val_neural = predict_neural_branches(vision_nets, geom_mlp, val_loader, train_ds.phys_mean, train_ds.phys_std, train_ds.geom_indices, tta_mode="8x")
    test_neural = predict_neural_branches(vision_nets, geom_mlp, test_loader, train_ds.phys_mean, train_ds.phys_std, train_ds.geom_indices, tta_mode="8x")

    v_st_t = val_ds.state_labels
    v_dir_t = val_ds.dir_labels
    v_rot_t = val_ds.rot_values

    # Search convex weights across loaded vision backbones (up to 3) + HGB + MLP on Val
    # Objective includes min(straight_acc, turning_acc) so Turning recall improves without sacrificing Straight!
    n_vis = len(vision_nets)
    best_vis_w = [1.0 / n_vis] * n_vis
    best_tab_w = (0.0, 0.0)  # hgb, mlp
    best_tau = 0.50
    best_val_obj = -1.0

    # Generate clean simplex grid over vision models
    vis_grids = []
    if n_vis == 1:
        vis_grids = [(1.0,)]
    elif n_vis == 2:
        for w0 in (0.30, 0.50, 0.65, 0.80, 0.90):
            vis_grids.append((w0, round(1.0 - w0, 2)))
    else:
        for w0 in (0.30, 0.45, 0.60, 0.75):
            for w1 in (0.10, 0.25, 0.40):
                w2 = round(1.0 - w0 - w1, 2)
                if 0.0 <= w2 <= 0.45:
                    vis_grids.append((w0, w1, w2))

    v_p_list = [d["p"] for d in val_neural["vis"]]
    for vw in vis_grids:
        v_vis_comb = sum(w * vp for w, vp in zip(vw, v_p_list))
        for w_hgb in (0.00, 0.10, 0.20):
            for w_mlp in (0.00, 0.10):
                w_vis = 1.0 - w_hgb - w_mlp
                if w_vis < 0.65:
                    continue
                v_ens = w_vis * v_vis_comb + w_hgb * val_hgb["hgb_p"] + w_mlp * val_neural["mlp_p"]
                v_auc = float(roc_auc_score(v_st_t, v_ens)) if len(np.unique(v_st_t)) > 1 else 0.5
                for tau in np.arange(0.32, 0.70, 0.02):
                    b = (v_ens >= tau).astype(int)
                    pf = float(np.mean(b))
                    if pf < 0.10 or pf > 0.70:
                        continue
                    st_a = float(np.mean(b[v_st_t == 0] == 0))
                    tu_a = float(np.mean(b[v_st_t == 1] == 1))
                    bal = 0.5 * (st_a + tu_a)
                    obj = 0.55 * bal + 0.25 * v_auc + 0.20 * min(st_a, tu_a)
                    if obj > best_val_obj:
                        best_val_obj = obj
                        best_vis_w = [w * w_vis for w in vw]
                        best_tab_w = (float(w_hgb), float(w_mlp))
                        best_tau = float(tau)

    w_hgb, w_mlp = best_tab_w
    w_vis_sum = max(1e-6, sum(best_vis_w))
    v_ens = sum(w * vp for w, vp in zip(best_vis_w, v_p_list)) + w_hgb * val_hgb["hgb_p"] + w_mlp * val_neural["mlp_p"]
    v_auc = float(roc_auc_score(v_st_t, v_ens)) if len(np.unique(v_st_t)) > 1 else 0.5
    v_st_acc = float(np.mean((v_ens >= best_tau)[v_st_t == 0] == 0))
    v_tu_acc = float(np.mean((v_ens >= best_tau)[v_st_t == 1] == 1))

    # Calibrate optional Port vs Starboard & ROT on Val
    v_vis_lr_list = [0.65 * d["lr"] + 0.35 * (d["dir"][:, 2] / (d["dir"][:, 1] + d["dir"][:, 2] + 1e-8)) for d in val_neural["vis"]]
    val_vis_lr = sum((w / w_vis_sum) * lrv for w, lrv in zip(best_vis_w, v_vis_lr_list))
    best_lr_mix, best_tau_lr = (0.80, 0.20), 0.50
    if np.any(v_st_t == 1):
        y_lr_v = v_dir_t[v_st_t == 1]
        best_lr_acc = -1.0
        for lv in (0.65, 0.80, 1.00):
            lh = round(1.0 - lv, 2)
            p_lr_v = lv * val_vis_lr[v_st_t == 1] + lh * val_hgb["hgb_lr"][v_st_t == 1]
            for t_lr in np.quantile(p_lr_v, np.linspace(0.35, 0.65, 13)):
                pred_lr = np.where(p_lr_v >= t_lr, 2, 1)
                ap = float(np.mean(pred_lr[y_lr_v == 1] == 1)) if np.any(y_lr_v == 1) else 0.0
                as_ = float(np.mean(pred_lr[y_lr_v == 2] == 2)) if np.any(y_lr_v == 2) else 0.0
                if 0.5 * (ap + as_) > best_lr_acc:
                    best_lr_acc = 0.5 * (ap + as_); best_lr_mix = (float(lv), float(lh)); best_tau_lr = float(t_lr)

    lv, lh = best_lr_mix
    val_vis_rot = sum((w / w_vis_sum) * d["rot"] for w, d in zip(best_vis_w, val_neural["vis"]))
    best_alpha_rot, best_st_scale, best_r2 = 0.85, 0.20, -99.0
    v_pred_bin = v_ens >= best_tau
    for ar in (0.65, 0.85, 1.00):
        rb = ar * val_vis_rot + (1.0 - ar) * val_hgb["hgb_rot"]
        for sc in (0.0, 0.20, 0.50):
            re_ = np.where(v_pred_bin, rb, sc * rb)
            r2 = float(1.0 - np.sum((v_rot_t - re_) ** 2) / (np.sum((v_rot_t - np.mean(v_rot_t)) ** 2) + 1e-6))
            if r2 > best_r2:
                best_r2 = r2; best_alpha_rot = float(ar); best_st_scale = float(sc)

    weights_dict = {name: float(round(w, 3)) for name, w in zip(vision_names, best_vis_w)}
    weights_dict.update({"geom_hgb": w_hgb, "geom_mlp": w_mlp})
    print(f"[CALIBRATION ON VAL] Weights={weights_dict}, tau={best_tau:.2f} -> "
          f"Val Bal={0.5*(v_st_acc+v_tu_acc)*100:.2f}% (St:{v_st_acc*100:.1f}%, Tu:{v_tu_acc*100:.1f}%), Val AUC={v_auc:.4f}")

    # Test Evaluation
    t_p_list = [d["p"] for d in test_neural["vis"]]
    t_st_p = sum(w * tp for w, tp in zip(best_vis_w, t_p_list)) + w_hgb * test_hgb["hgb_p"] + w_mlp * test_neural["mlp_p"]

    t_vis_lr_list = [0.65 * d["lr"] + 0.35 * (d["dir"][:, 2] / (d["dir"][:, 1] + d["dir"][:, 2] + 1e-8)) for d in test_neural["vis"]]
    test_vis_lr = sum((w / w_vis_sum) * lrt for w, lrt in zip(best_vis_w, t_vis_lr_list))
    t_stbd_cond = lv * test_vis_lr + lh * test_hgb["hgb_lr"]
    t_dir_p = build_hierarchical_dir_probs(t_st_p, t_stbd_cond)

    test_vis_rot = sum((w / w_vis_sum) * d["rot"] for w, d in zip(best_vis_w, test_neural["vis"]))
    t_rot_raw = best_alpha_rot * test_vis_rot + (1.0 - best_alpha_rot) * test_hgb["hgb_rot"]
    t_rot_p = np.where(t_st_p >= best_tau, t_rot_raw, best_st_scale * t_rot_raw)

    full_metrics = evaluate_predictions(
        t_st_p, t_dir_p, t_rot_p,
        test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
        tau=best_tau, p_stbd_cond=t_stbd_cond, tau_lr=best_tau_lr,
    )
    full_metrics["calibrated_weights"] = weights_dict
    full_metrics["tta_mode"] = "8x_D4_Chiral"
    full_metrics["latency_ms_per_ship"] = float(test_neural["mean_latency_s"] * 1000.0)
    full_metrics["throughput_fps"] = float(1.0 / (test_neural["mean_latency_s"] + 1e-6))

    results = {"expanded_test": full_metrics, "overfit_audit": overfit_audit}
    results["branch_ablations"] = {
        "geom_mlp_only": evaluate_predictions(test_neural["mlp_p"], test_neural["mlp_dir"], test_hgb["hgb_rot"], test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values, tau=0.50, p_stbd_cond=test_neural["mlp_lr"], tau_lr=0.50),
        "geom_hgb_only": evaluate_predictions(test_hgb["hgb_p"], build_hierarchical_dir_probs(test_hgb["hgb_p"], test_hgb["hgb_lr"]), test_hgb["hgb_rot"], test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values, tau=0.50, p_stbd_cond=test_hgb["hgb_lr"], tau_lr=0.50),
    }
    for name, d, lr_t in zip(vision_names, test_neural["vis"], t_vis_lr_list):
        results["branch_ablations"][name] = evaluate_predictions(
            d["p"], d["dir"], d["rot"], test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
            tau=best_tau, p_stbd_cond=lr_t, tau_lr=best_tau_lr,
        )

    abs_rot_te = np.abs(test_df["rot_deg_min"].fillna(0.0).values) if "rot_deg_min" in test_df.columns else np.abs(test_ds.rot_values)
    rot2_mask = (test_ds.state_labels == 0) | ((test_ds.state_labels == 1) & (abs_rot_te >= 2.0))
    if np.any(rot2_mask) and np.sum(test_ds.state_labels[rot2_mask] == 1) >= 5:
        results["clear_tactical_turns_rot2"] = evaluate_predictions(
            t_st_p[rot2_mask], t_dir_p[rot2_mask], t_rot_p[rot2_mask],
            test_ds.state_labels[rot2_mask], test_ds.dir_labels[rot2_mask], test_ds.rot_values[rot2_mask],
            tau=best_tau, p_stbd_cond=t_stbd_cond[rot2_mask], tau_lr=best_tau_lr,
        )

    if "source_dataset" in test_df.columns:
        for src in ("MMDEC", "UEIKAP"):
            sm = (test_df["source_dataset"] == src).values
            if np.any(sm) and len(np.unique(test_ds.state_labels[sm])) > 1:
                results[f"{src.lower()}_test"] = evaluate_predictions(
                    t_st_p[sm], t_dir_p[sm], t_rot_p[sm],
                    test_ds.state_labels[sm], test_ds.dir_labels[sm], test_ds.rot_values[sm],
                    tau=best_tau, p_stbd_cond=t_stbd_cond[sm], tau_lr=best_tau_lr,
                )

    if "sog" in test_df.columns:
        sog_vals = test_df["sog"].fillna(-1.0).values
        sog_report = {}
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 999)):
            gm = (sog_vals >= lo) & (sog_vals < hi)
            if np.sum(gm) >= 5 and len(np.unique(test_ds.state_labels[gm])) > 1:
                yy = test_ds.state_labels[gm]
                pr = (t_st_p[gm] >= best_tau).astype(int)
                sog_report[f"sog_{lo}_{hi}"] = {
                    "n": int(np.sum(gm)),
                    "straight_acc": float(np.mean(pr[yy == 0] == 0)),
                    "turning_acc": float(np.mean(pr[yy == 1] == 1)),
                    "auc_roc": float(roc_auc_score(yy, t_st_p[gm])),
                }
        results["sog_regime_breakdown"] = sog_report

    w0 = best_vis_w[0]
    w_neural_sum = max(1e-6, w0 + w_mlp)
    print("\n[UNCERTAINTY] Running N=30 MC-Dropout passes across unseen test set...")
    unc_metrics, mc_mean, mc_std, mc_entropy = run_mc_dropout_uncertainty(
        p_net, geom_mlp, test_loader, test_ds.state_labels, tau=best_tau, w_cx=(w0 / w_neural_sum), num_passes=30,
    )
    results["mc_dropout_uncertainty"] = unc_metrics

    print("[EXPLAINABILITY] Computing permutation importance across 20 Wake/Radon/PCA features...")
    imp_df = compute_physics_permutation_importance(
        p_net, geom_mlp, test_ds, full_metrics["balanced_accuracy"], tau=best_tau, w_cx=(w0 / w_neural_sum), batch_size=batch_size,
    )
    results["top_physics_features"] = imp_df.head(6).to_dict("records")

    print("\n" + "=" * 80)
    print("FINAL PUBLICATION BENCHMARK SUMMARY (STERN-ALIGNED + EMA + 3-BACKBONE + 8x D4 TTA)")
    print("=" * 80)
    print(json.dumps(results, indent=2))

    with open(os.path.join(CHECKPOINTS_DIR, "master_evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    pred_out = test_df[["sample_id", "source_dataset", "state", "direction"]].copy() if "source_dataset" in test_df.columns else test_df[["sample_id", "state", "direction"]].copy()
    pred_out["prob_turn_ensemble"] = t_st_p
    pred_out["prob_starboard_cond"] = t_stbd_cond
    pred_out["mc_mean_prob"] = mc_mean
    pred_out["epistemic_std"] = mc_std
    pred_out["predictive_entropy"] = mc_entropy
    pred_bin_out = t_st_p >= best_tau
    pred_out["pred_state"] = np.where(pred_bin_out, "Turning", "Straight")
    pred_out["pred_direction"] = np.where(~pred_bin_out, "None", np.where(t_stbd_cond >= best_tau_lr, "Starboard", "Port"))
    pred_out["pred_rot_deg_min"] = t_rot_p
    pred_out.to_parquet(os.path.join(DATA_DIR, "processed", "expanded_test_predictions.parquet"), index=False)

    if gradcam_per_class != 0:
        try:
            generate_lee_vs_raw_gradcam(p_net, test_ds, GRADCAM_DIR, num_per_class=gradcam_per_class)
            print(f"[GRAD-CAM] Saved comparative visualizations & wake overlap CSV in {GRADCAM_DIR} and {TABLES_DIR}")
        except Exception as e:
            print(f"[WARNING] Grad-CAM visualization skipped: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stern-Aligned Anti-Overfit Multi-Backbone Ensemble")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--primary_backbone", type=str, default="convnext_small")
    parser.add_argument("--second_backbone", type=str, default="swinv2_tiny_window8_256")
    parser.add_argument("--third_backbone", type=str, default="resnet50d")
    parser.add_argument("--gradcam_per_class", type=int, default=-1, help="-1 generates all, 0 skips")
    parser.add_argument("--allow_partial_data", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()
    run_master_suite(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        gradcam_per_class=args.gradcam_per_class,
        primary_backbone=args.primary_backbone,
        second_backbone=args.second_backbone,
        third_backbone=args.third_backbone,
        allow_partial_data=args.allow_partial_data,
        pretrained=(not args.no_pretrained),
    )
