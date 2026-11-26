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
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm

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
    """Differentiable 7x7 Lee filter suppressing multiplicative SAR sea clutter speckle."""
    def __init__(self, kernel_size=7, noise_var=0.08):
        super().__init__()
        self.pad = kernel_size // 2
        self.noise_var = noise_var
        box_weight = torch.full((1, 1, kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=torch.float32)
        self.register_buffer("box_weight", box_weight)

    def forward(self, x):
        mean = F.conv2d(x, self.box_weight, padding=self.pad)
        sq_mean = F.conv2d(x * x, self.box_weight, padding=self.pad)
        var = torch.clamp(sq_mean - mean * mean, min=1e-6)
        # Local adaptive gain K in [0, 1]
        k = torch.clamp((var - (mean * mean) * self.noise_var) / var, min=0.0, max=1.0)
        return torch.clamp(mean + k * (x - mean), 0.0, 1.0)


class ConvNeXtPhysicsNet(nn.Module):
    """
    Branch A (Method 3): Adaptive 7x7 Lee Filter + ConvNeXt-Tiny + 12 Wake/Radon Geometry Features.
    Outputs Multi-Task Heads: Binary State Logit, 3-Class Direction Logits, Continuous Signed ROT.
    """
    def __init__(self, num_physics_features=12, pretrained=True):
        super().__init__()
        self.speckle_filter = LeeSpeckleFilter(kernel_size=7, noise_var=0.08)
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            drop_rate=0.25,
            drop_path_rate=0.20,
        )
        embed_dim = self.backbone.num_features  # 768

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.physics_mlp = nn.Sequential(
            nn.Linear(num_physics_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.SiLU()
        )

        self.shared_head = nn.Sequential(
            nn.Linear(embed_dim + 64, 160),
            nn.BatchNorm1d(160),
            nn.SiLU(),
            nn.Dropout(0.30)
        )

        self.binary_head = nn.Linear(160, 1)
        self.direction_head = nn.Linear(160, 3)
        self.rot_head = nn.Linear(160, 1)

    def forward(self, x_img, x_phys):
        # x_img: (B, 2, H, W) -> Ch 0 = SAR grayscale, Ch 1 = Wake-Net Mask
        sar_ch = x_img[:, 0:1]
        mask_ch = x_img[:, 1:2]
        lee_ch = self.speckle_filter(sar_ch)
        # 3-channel physics-guided tensor: [Lee-filtered SAR, Contrast SAR, Wake Mask Spotlight]
        x_raw3 = torch.cat([lee_ch, sar_ch, mask_ch], dim=1)
        x_3ch = (x_raw3 - self.img_mean) / self.img_std
        v_emb = self.backbone(x_3ch)
        p_emb = self.physics_mlp(x_phys)
        feat = self.shared_head(torch.cat([v_emb, p_emb], dim=1))
        return self.binary_head(feat).squeeze(1), self.direction_head(feat), self.rot_head(feat).squeeze(1)


class GeometrySpecialistMLP(nn.Module):
    """
    Branch B (Method 1 Anchor / Model 3): Pure Wake Curvature & Radon Specialist.
    Acts as a mathematical veto against sea-clutter visual false positives.
    """
    def __init__(self, in_features=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(32, 1)
        self.direction_head = nn.Linear(32, 3)

    def forward(self, x_geom):
        h = self.net(x_geom)
        return self.binary_head(h).squeeze(1), self.direction_head(h)


# --------------------------------------------------------------------------------------------------
# Multimodal Dataset with Train-Only Normalization (Zero Leakage)
# --------------------------------------------------------------------------------------------------
class ScaledManeuverDataset(Dataset):
    def __init__(self, df, phys_mean=None, phys_std=None, img_size=(256, 256), augment=False):
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
        self.rot_values = self.df.get(rot_col, pd.Series(np.zeros(len(self.df)))).fillna(0.0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p_crop = resolve_path(str(row.get("crop_path", "")))
        p_orig = resolve_path(str(row.get("original_image_path", "")))
        img_p = p_crop if os.path.exists(p_crop) else p_orig

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load SAR chip for sample {row.get('sample_id')}: {img_p}")
        img = self.clahe.apply(cv2.resize(img, self.img_size))

        # Load Wake-Net / GT segmented wake mask as 2nd channel spotlight
        m_path = resolve_path(str(row.get("wake_mask_path", "")))
        if os.path.exists(m_path):
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(m_img, self.img_size, interpolation=cv2.INTER_NEAREST) if m_img is not None else np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        phys_vec = self.phys_raw[idx].copy()
        dir_val = int(self.dir_labels[idx])
        rot_val = float(self.rot_values[idx])

        if self.augment:
            # Physical horizontal mirror augmentation: swaps Port (1) <-> Starboard (2) and negates chiral physics!
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
                if dir_val == 1:
                    dir_val = 2
                elif dir_val == 2:
                    dir_val = 1
                rot_val = -rot_val
                for ci in self.chiral_indices:
                    phys_vec[ci] = -phys_vec[ci]
            if np.random.rand() < 0.4:
                gamma = np.random.uniform(0.90, 1.10)
                img = np.clip(img.astype(np.float32) * gamma, 0, 255).astype(np.uint8)

        phys_norm_vec = np.clip((phys_vec - self.phys_mean) / self.phys_std, -8.0, 8.0).astype(np.float32)
        geom_norm_vec = phys_norm_vec[self.geom_indices]

        img_2ch = np.stack([img.astype(np.float32) / 255.0, (mask > 127).astype(np.float32)], axis=0)
        img_t = torch.from_numpy(img_2ch)
        phys_t = torch.from_numpy(phys_norm_vec)
        geom_t = torch.from_numpy(geom_norm_vec)
        state_t = torch.tensor(self.state_labels[idx], dtype=torch.float32)
        dir_t = torch.tensor(dir_val, dtype=torch.long)
        rot_t = torch.tensor(rot_val, dtype=torch.float32)

        return img_t, phys_t, geom_t, state_t, dir_t, rot_t


# --------------------------------------------------------------------------------------------------
# Physics-Informed Multi-Task Loss
# --------------------------------------------------------------------------------------------------
class PhysicsInformedMultiTaskLoss(nn.Module):
    def __init__(self, pos_weight=2.0, dir_weights=None, lambda_dir=0.45, lambda_rot=0.45, lambda_phys=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
        if dir_weights is not None:
            self.ce_dir = nn.CrossEntropyLoss(weight=torch.tensor(dir_weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
        else:
            self.ce_dir = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.lambda_dir = lambda_dir
        self.lambda_rot = lambda_rot
        self.lambda_phys = lambda_phys

    def forward(self, state_logit, dir_logits, rot_pred, state_t, dir_t, rot_t, signed_curv):
        l_state = self.bce(state_logit, state_t)
        l_dir = self.ce_dir(dir_logits, dir_t)
        # Weight turning vessels 3x in signed ROT regression so turning magnitude is learned cleanly
        rot_weights = 1.0 + 3.0 * state_t
        l_rot = torch.mean(rot_weights * F.smooth_l1_loss(rot_pred, rot_t, reduction="none", beta=1.0))
        # Light sign-coherence constraint on turning vessels
        l_phys = torch.mean(state_t * torch.relu(-rot_pred * torch.sign(rot_t)))
        return l_state + self.lambda_dir * l_dir + self.lambda_rot * l_rot + self.lambda_phys * l_phys


def evaluate_predictions(state_probs, dir_probs, rot_preds, state_true, dir_true, rot_true, tau=0.50):
    pred_bin = (state_probs >= tau).astype(int)
    st_mask = state_true == 0
    tu_mask = state_true == 1

    st_acc = float(np.mean(pred_bin[st_mask] == 0)) if np.any(st_mask) else 0.0
    tu_acc = float(np.mean(pred_bin[tu_mask] == 1)) if np.any(tu_mask) else 0.0
    bal_acc = 0.5 * (st_acc + tu_acc)
    overall_acc = float(np.mean(pred_bin == state_true))

    # Directional 3-Class (Straight=0, Port=1, Starboard=2)
    dir_pred = np.argmax(dir_probs, axis=1)
    dir_overall = float(np.mean(dir_pred == dir_true))
    c_accs = []
    for c in (0, 1, 2):
        m = dir_true == c
        c_accs.append(float(np.mean(dir_pred[m] == c)) if np.any(m) else 0.0)
    dir_bal = float(np.mean(c_accs))

    # Port vs Starboard purely on true Turning vessels
    turn_mask = (dir_true == 1) | (dir_true == 2)
    turn_dir_acc = float(np.mean(np.where(dir_probs[turn_mask, 1] >= dir_probs[turn_mask, 2], 1, 2) == dir_true[turn_mask])) if np.any(turn_mask) else 0.0

    # Continuous Signed ROT
    rot_mae = float(np.mean(np.abs(rot_preds - rot_true)))
    ss_res = float(np.sum((rot_true - rot_preds) ** 2))
    ss_tot = float(np.sum((rot_true - np.mean(rot_true)) ** 2)) + 1e-6
    rot_r2 = 1.0 - (ss_res / ss_tot)

    return {
        "threshold": float(tau),
        "n_samples": int(len(state_true)),
        "n_straight": int(np.sum(st_mask)),
        "n_turning": int(np.sum(tu_mask)),
        "balanced_accuracy": bal_acc,
        "overall_accuracy": overall_acc,
        "straight_accuracy": st_acc,
        "turning_accuracy": tu_acc,
        "dir_3class_overall_acc": dir_overall,
        "dir_3class_balanced_acc": dir_bal,
        "dir_straight_acc": c_accs[0],
        "dir_port_acc": c_accs[1],
        "dir_starboard_acc": c_accs[2],
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

        st_logit, dir_logits, rot_pred = convnext_net(x_img, x_phys)
        convnext_net.zero_grad()

        # Target predicted directional class (or binary maneuver score when straight)
        pred_cls = int(torch.argmax(dir_logits, dim=1).item())
        score = dir_logits[0, pred_cls] if pred_cls != 0 else st_logit[0]
        score.backward()

        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_res = cv2.resize(cam, (256, 256))

        # Overlay wake mask contour & compute wake attention concentration
        row = test_ds.df.iloc[idx]
        m_path = resolve_path(str(row.get("wake_mask_path", "")))
        mask_bin = None
        wake_energy_ratio = np.nan
        if os.path.exists(m_path):
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m_img is not None:
                m_res = cv2.resize(m_img, (256, 256), interpolation=cv2.INTER_NEAREST)
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


def run_mc_dropout_uncertainty(convnext_net, geom_mlp, test_loader, state_true, tau=0.50, w_cx=0.60, num_passes=30):
    """
    Executes N=30 stochastic MC-Dropout forward passes on the Dual-Branch Ensemble,
    computes Predictive Entropy and Entropy Gap (Incorrect vs Correct), and saves the
    Selective Error Rejection Curve (figures/convnext_uncertainty_rejection_curve.png).
    """
    convnext_net.eval()
    geom_mlp.eval()
    # Activate ONLY Dropout layers while keeping BatchNorm frozen in eval mode
    for m in list(convnext_net.modules()) + list(geom_mlp.modules()):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    pass_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            p_batch = []
            for imgs, phys, geom, _, _, _ in test_loader:
                imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
                st_log, _, _ = convnext_net(imgs, phys)
                g_st_log, _ = geom_mlp(geom)
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


def compute_physics_permutation_importance(convnext_net, geom_mlp, test_ds, base_bal_acc, tau=0.50, w_cx=0.60, batch_size=32):
    """Quantifies Balanced Accuracy drop when each Wake/Radon physics feature is randomly permuted."""
    convnext_net.eval()
    geom_mlp.eval()
    geom_indices = [PHYSICS_FEATURES.index(c) for c in GEOMETRY_FEATURES]
    imp_rows = []

    orig_phys_raw = test_ds.phys_raw.copy()

    for f_idx, fname in enumerate(PHYSICS_FEATURES):
        perm_raw = orig_phys_raw.copy()
        perm_raw[:, f_idx] = np.random.permutation(perm_raw[:, f_idx])
        test_ds.phys_raw = perm_raw

        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        p_all = []
        with torch.no_grad():
            for imgs, phys, geom, _, _, _ in loader:
                st_log, _, _ = convnext_net(imgs.to(DEVICE), phys.to(DEVICE))
                g_st_log, _ = geom_mlp(geom.to(DEVICE))
                p_all.append((w_cx * torch.sigmoid(st_log) + (1.0 - w_cx) * torch.sigmoid(g_st_log)).cpu().numpy())
        p_all = np.concatenate(p_all)
        b = (p_all >= tau).astype(int)
        st_acc = np.mean(b[test_ds.state_labels == 0] == 0)
        tu_acc = np.mean(b[test_ds.state_labels == 1] == 1)
        bal = 0.5 * (st_acc + tu_acc)
        imp_rows.append({"feature": fname, "permuted_bal_acc": float(bal), "importance_drop": float(base_bal_acc - bal)})

    test_ds.phys_raw = orig_phys_raw

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


def run_master_suite(epochs=25, batch_size=32, lr=2e-4, gradcam_per_class=-1):
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
        print(f"[DATA] Filtered {len(master_df) - valid_mask.sum()} missing-image rows; retaining {valid_mask.sum():,} valid chips.")
        master_df = master_df[valid_mask].reset_index(drop=True)

    train_df = master_df[master_df["split"] == "train"].copy()
    val_df = master_df[master_df["split"] == "val"].copy()
    test_df = master_df[master_df["split"] == "test"].copy()

    print(f"[SPLITS] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"[TRAIN BALANCE] State: {train_df['state'].value_counts().to_dict()} | Dir: {train_df['direction'].value_counts().to_dict()}")

    # Fit normalization strictly on Train split
    train_ds = ScaledManeuverDataset(train_df, augment=True)
    val_ds = ScaledManeuverDataset(val_df, phys_mean=train_ds.phys_mean, phys_std=train_ds.phys_std, augment=False)
    test_ds = ScaledManeuverDataset(test_df, phys_mean=train_ds.phys_mean, phys_std=train_ds.phys_std, augment=False)

    # Save scaler stats for future standalone inference
    np.savez(
        os.path.join(CHECKPOINTS_DIR, "feature_scaler.npz"),
        phys_mean=train_ds.phys_mean,
        phys_std=train_ds.phys_std,
        features=np.array(PHYSICS_FEATURES)
    )

    # Class balancing sampler + dynamic loss weights
    n_st = max(1, int(np.sum(train_ds.state_labels == 0)))
    n_tu = max(1, int(np.sum(train_ds.state_labels == 1)))
    pos_weight = float(np.clip(n_st / n_tu, 1.0, 4.0))

    dir_counts = np.bincount(train_ds.dir_labels, minlength=3).astype(np.float32) + 1.0
    dir_weights = (dir_counts.sum() / (3.0 * dir_counts))
    dir_weights = np.clip(dir_weights, 0.5, 5.0)

    sample_weights = np.where(train_ds.state_labels == 1, pos_weight, 1.0)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=2 if IS_KAGGLE else 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Instantiate Dual-Branch Ensemble Models
    convnext_net = ConvNeXtPhysicsNet(num_physics_features=len(PHYSICS_FEATURES), pretrained=True).to(DEVICE)
    geom_mlp = GeometrySpecialistMLP(in_features=len(GEOMETRY_FEATURES)).to(DEVICE)

    criterion = PhysicsInformedMultiTaskLoss(pos_weight=1.0, dir_weights=dir_weights).to(DEVICE)
    bce_geom = nn.BCEWithLogitsLoss()
    ce_geom_dir = nn.CrossEntropyLoss(weight=torch.tensor(dir_weights, dtype=torch.float32, device=DEVICE))

    optimizer = torch.optim.AdamW([
        {"params": convnext_net.backbone.parameters(), "lr": lr * 0.5},
        {"params": convnext_net.physics_mlp.parameters(), "lr": lr},
        {"params": convnext_net.shared_head.parameters(), "lr": lr},
        {"params": convnext_net.binary_head.parameters(), "lr": lr},
        {"params": convnext_net.direction_head.parameters(), "lr": lr},
        {"params": convnext_net.rot_head.parameters(), "lr": lr},
        {"params": geom_mlp.parameters(), "lr": lr * 2.0},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_cx_score = -1.0
    best_geom_score = -1.0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        convnext_net.train()
        geom_mlp.train()
        running_loss = 0.0

        for imgs, phys, geom, state_t, dir_t, rot_t in train_loader:
            imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
            state_t, dir_t, rot_t = state_t.to(DEVICE), dir_t.to(DEVICE), rot_t.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                st_log, dir_log, rot_out = convnext_net(imgs, phys)
                g_st_log, g_dir_log = geom_mlp(geom)

                loss_cx = criterion(st_log, dir_log, rot_out, state_t, dir_t, rot_t, phys[:, 5])
                loss_g = bce_geom(g_st_log, state_t) + 0.4 * ce_geom_dir(g_dir_log, dir_t)
                loss = loss_cx + 0.5 * loss_g

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(convnext_net.parameters()) + list(geom_mlp.parameters()), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(state_t)

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader.dataset))

        # Independent Validation Check for ConvNeXt Branch A & Geometry MLP Branch B
        convnext_net.eval()
        geom_mlp.eval()
        vcx_p, vg_p, v_dir_cx, v_dir_g, v_rot_p = [], [], [], [], []
        v_st_t, v_dir_t, v_rot_t = [], [], []

        with torch.no_grad():
            for imgs, phys, geom, state_t, dir_t, rot_t in val_loader:
                imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
                st_log, dir_log, rot_out = convnext_net(imgs, phys)
                g_st_log, g_dir_log = geom_mlp(geom)

                vcx_p.append(torch.sigmoid(st_log).cpu().numpy())
                vg_p.append(torch.sigmoid(g_st_log).cpu().numpy())
                v_dir_cx.append(torch.softmax(dir_log, dim=1).cpu().numpy())
                v_dir_g.append(torch.softmax(g_dir_log, dim=1).cpu().numpy())
                v_rot_p.append(rot_out.cpu().numpy())
                v_st_t.append(state_t.numpy())
                v_dir_t.append(dir_t.numpy())
                v_rot_t.append(rot_t.numpy())

        vcx_p_np = np.concatenate(vcx_p)
        vg_p_np = np.concatenate(vg_p)
        v_dir_cx_np = np.concatenate(v_dir_cx)
        v_dir_g_np = np.concatenate(v_dir_g)
        v_rot_np = np.concatenate(v_rot_p)
        v_st_np = np.concatenate(v_st_t)
        v_dir_np = np.concatenate(v_dir_t)
        v_rot_t_np = np.concatenate(v_rot_t)

        cx_res = evaluate_predictions(vcx_p_np, v_dir_cx_np, v_rot_np, v_st_np, v_dir_np, v_rot_t_np, tau=0.50)
        g_res = evaluate_predictions(vg_p_np, v_dir_g_np, v_rot_np, v_st_np, v_dir_np, v_rot_t_np, tau=0.50)

        cx_score = 0.7 * cx_res["balanced_accuracy"] + 0.3 * cx_res["dir_3class_balanced_acc"]
        g_score = 0.7 * g_res["balanced_accuracy"] + 0.3 * g_res["dir_3class_balanced_acc"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | "
            f"ConvNeXt BalAcc: {cx_res['balanced_accuracy']*100:.1f}% (Dir:{cx_res['dir_3class_balanced_acc']*100:.1f}%) | "
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
    # Calibrate Optimal Soft-Voting Weight w & Threshold tau Strictly on Validation Split!
    # ----------------------------------------------------------------------------------------------
    convnext_net.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "convnext_multitask_best.pt"), map_location=DEVICE))
    geom_mlp.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "geometry_mlp_best.pt"), map_location=DEVICE))
    convnext_net.eval()
    geom_mlp.eval()

    vcx_p, vg_p, v_st_t = [], [], []
    with torch.no_grad():
        for imgs, phys, geom, state_t, _, _ in val_loader:
            st_log, _, _ = convnext_net(imgs.to(DEVICE), phys.to(DEVICE))
            g_st_log, _ = geom_mlp(geom.to(DEVICE))
            vcx_p.append(torch.sigmoid(st_log).cpu().numpy())
            vg_p.append(torch.sigmoid(g_st_log).cpu().numpy())
            v_st_t.append(state_t.numpy())
    vcx_p, vg_p, v_st_t = np.concatenate(vcx_p), np.concatenate(vg_p), np.concatenate(v_st_t)

    best_w, best_tau, best_val_bal = 0.60, 0.50, 0.0
    for w in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        v_ens = w * vcx_p + (1.0 - w) * vg_p
        for tau in np.arange(0.30, 0.76, 0.02):
            b = (v_ens >= tau).astype(int)
            bal = 0.5 * (np.mean(b[v_st_t == 0] == 0) + np.mean(b[v_st_t == 1] == 1))
            if bal > best_val_bal:
                best_val_bal = bal
                best_w = float(w)
                best_tau = float(tau)

    print(f"[CALIBRATION ON VAL] Optimal Soft-Vote Weight w_ConvNeXt={best_w:.2f}, tau={best_tau:.2f} -> Val BalAcc={best_val_bal*100:.2f}%")

    # ----------------------------------------------------------------------------------------------
    # Final Evaluation on Unseen Test Split (Full & Frozen UEIKAP Subset)
    # ----------------------------------------------------------------------------------------------
    t_st_p, t_dir_p, t_rot_p, times = [], [], [], []
    with torch.no_grad():
        for imgs, phys, geom, _, _, _ in test_loader:
            imgs, phys, geom = imgs.to(DEVICE), phys.to(DEVICE), geom.to(DEVICE)
            t0 = time.time()
            st_log, dir_log, rot_out = convnext_net(imgs, phys)
            g_st_log, g_dir_log = geom_mlp(geom)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.time() - t0) / len(imgs))

            t_st_p.append((best_w * torch.sigmoid(st_log) + (1.0 - best_w) * torch.sigmoid(g_st_log)).cpu().numpy())
            t_dir_p.append((best_w * torch.softmax(dir_log, dim=1) + (1.0 - best_w) * torch.softmax(g_dir_log, dim=1)).cpu().numpy())
            t_rot_p.append(rot_out.cpu().numpy())

    t_st_p = np.concatenate(t_st_p)
    t_dir_p = np.concatenate(t_dir_p)
    t_rot_p = np.concatenate(t_rot_p)

    full_metrics = evaluate_predictions(
        t_st_p, t_dir_p, t_rot_p,
        test_ds.state_labels, test_ds.dir_labels, test_ds.rot_values,
        tau=best_tau
    )
    full_metrics["calibrated_w_convnext"] = best_w
    full_metrics["latency_ms_per_ship"] = float(np.mean(times) * 1000.0)
    full_metrics["throughput_fps"] = float(1.0 / (np.mean(times) + 1e-6))

    results = {"expanded_test": full_metrics}

    # Evaluate separately on Original Frozen UEIKAP Test Set if present
    if "source_dataset" in test_df.columns:
        ueikap_mask = (test_df["source_dataset"] == "UEIKAP").values
        if np.any(ueikap_mask):
            ueikap_metrics = evaluate_predictions(
                t_st_p[ueikap_mask], t_dir_p[ueikap_mask], t_rot_p[ueikap_mask],
                test_ds.state_labels[ueikap_mask], test_ds.dir_labels[ueikap_mask], test_ds.rot_values[ueikap_mask],
                tau=best_tau
            )
            results["ueikap_frozen_test"] = ueikap_metrics

    # ----------------------------------------------------------------------------------------------
    # Epistemic Uncertainty Quantification (N=30 MC-Dropout Passes) & Selective Rejection Curve
    # ----------------------------------------------------------------------------------------------
    print("\n[UNCERTAINTY] Running N=30 MC-Dropout passes across unseen test set...")
    unc_metrics, mc_mean, mc_std, mc_entropy = run_mc_dropout_uncertainty(
        convnext_net, geom_mlp, test_loader, test_ds.state_labels, tau=best_tau, w_cx=best_w, num_passes=30
    )
    results["mc_dropout_uncertainty"] = unc_metrics

    # ----------------------------------------------------------------------------------------------
    # Hydrodynamic Explainability: Physics/Radon Feature Permutation Importance
    # ----------------------------------------------------------------------------------------------
    print("[EXPLAINABILITY] Computing permutation importance across 12 Wake/Radon features...")
    imp_df = compute_physics_permutation_importance(
        convnext_net, geom_mlp, test_ds, full_metrics["balanced_accuracy"], tau=best_tau, w_cx=best_w, batch_size=batch_size
    )
    results["top_physics_features"] = imp_df.head(5).to_dict("records")

    print("\n" + "=" * 80)
    print("FINAL PUBLICATION BENCHMARK SUMMARY (CALIBRATED ON VAL, FROZEN ON TEST)")
    print("=" * 80)
    print(json.dumps(results, indent=2))

    with open(os.path.join(CHECKPOINTS_DIR, "master_evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save test predictions parquet with per-vessel Uncertainty metrics
    pred_out = test_df[["sample_id", "source_dataset", "state", "direction"]].copy() if "source_dataset" in test_df.columns else test_df[["sample_id", "state", "direction"]].copy()
    pred_out["prob_turn_ensemble"] = t_st_p
    pred_out["mc_mean_prob"] = mc_mean
    pred_out["epistemic_std"] = mc_std
    pred_out["predictive_entropy"] = mc_entropy
    pred_out["pred_state"] = np.where(t_st_p >= best_tau, "Turning", "Straight")
    inv_dir = {0: "None", 1: "Port", 2: "Starboard"}
    pred_out["pred_direction"] = [inv_dir[int(c)] for c in np.argmax(t_dir_p, axis=1)]
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradcam_per_class", type=int, default=-1, help="-1 generates Grad-CAM for ALL test images")
    args = parser.parse_args()
    run_master_suite(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, gradcam_per_class=args.gradcam_per_class)
