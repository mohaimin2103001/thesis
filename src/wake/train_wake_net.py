"""
Hybrid Wake-Net: Attention U-Net with scSE & Fast Sparse Radon Transform Kinematic Extraction
Implements Paper 1 (Wake-Net: Zhang et al., IEEE TGRS 2023) with Spatial & Channel
Squeeze-and-Excitation (scSE) attention + Paper 3 (UEIKAP: Graziano et al.) Radon Transform
Centerline Fitting and Multi-Window Angular Curvature Analysis.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
cv2.setNumThreads(0)
from skimage.morphology import remove_small_objects
try:
    import segmentation_models_pytorch as smp
except ImportError:
    if os.path.exists("/kaggle/working"):
        print("[WAKE-NET] Installing missing 'segmentation-models-pytorch' on Kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"])
        import segmentation_models_pytorch as smp
    else:
        raise

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "wake_net")
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "wake_segmentation_kaggle")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def extract_unified_sar_gray(img_bgr_or_gray):
    """
    Normalizes UEIKAP false-color VV/VH PNGs (where G is sparse and B carries ratio/signal)
    and MMDEC grayscale/VV-VH chips onto a single consistent luminance range.
    """
    if img_bgr_or_gray is None:
        return None
    if img_bgr_or_gray.ndim == 2:
        return img_bgr_or_gray
    b_ch, g_ch, r_ch = img_bgr_or_gray[:, :, 0], img_bgr_or_gray[:, :, 1], img_bgr_or_gray[:, :, 2]
    if float(np.mean(g_ch)) < 0.40 * (float(np.mean(b_ch)) + 1e-5):
        return np.maximum(b_ch, r_ch)
    return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)


def preprocess_sar_logdb_clahe(img_gray, clahe=None, out_size=(384, 384)):
    """Applies clean CLAHE normalization (avoiding per-image min-max quantization stretch)."""
    img_gray = extract_unified_sar_gray(img_gray)
    if out_size is not None and (img_gray.shape[1], img_gray.shape[0]) != tuple(out_size):
        img_gray = cv2.resize(img_gray, out_size)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def build_wakenet_3ch(img_bgr_or_gray, clahe=None, out_size=(384, 384)):
    """
    Constructs a domain-consistent 3-channel SAR tensor (H, W, 3) uint8:
      [0] unified raw SAR backscatter (fixes UEIKAP sparse-G drop)
      [1] CLAHE local contrast stretched SAR
      [2] local background-subtracted anomaly map (sharpens faint wake arms across wind gradients)
    """
    base = extract_unified_sar_gray(img_bgr_or_gray)
    if out_size is not None and (base.shape[1], base.shape[0]) != tuple(out_size):
        base = cv2.resize(base, out_size)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_u8 = clahe.apply(base)
    bg = cv2.GaussianBlur(clahe_u8.astype(np.float32), (31, 31), 0)
    hp_u8 = np.clip(clahe_u8.astype(np.float32) - bg + 128.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([base, clahe_u8, hp_u8], axis=-1)


def hysteresis_wake_mask(prob_map, seed_thresh=0.35, grow_thresh=0.20, min_size=30):
    """
    Connects faint distal wake tails to high-confidence wake seeds via hysteresis thresholding,
    suppressing disconnected isolated sea-clutter speckles.
    """
    grow_bin = (prob_map >= grow_thresh).astype(np.uint8)
    seed_bin = prob_map >= seed_thresh
    num_labels, labels = cv2.connectedComponents(grow_bin, connectivity=8)
    kept = np.zeros_like(grow_bin, dtype=bool)
    if num_labels > 1 and np.any(seed_bin):
        hit_labels = np.unique(labels[seed_bin])
        for lbl in hit_labels:
            if lbl > 0:
                kept |= (labels == lbl)
    if np.any(kept):
        kept = remove_small_objects(kept, min_size=min_size)
    out = (kept.astype(np.uint8)) * 255
    if np.any(out):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


class WakeDataset(Dataset):
    def __init__(self, df, img_dir=None, mask_dir=None, img_size=(384, 384), augment=False, in_channels=3):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.in_channels = in_channels
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if "image_path" in row and pd.notna(row["image_path"]):
            img_path = str(row["image_path"])
        else:
            img_path = os.path.join(self.img_dir, row["image_filename"])
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = str(row["mask_path"])
        else:
            mask_path = os.path.join(self.mask_dir, row["mask_filename"])

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing SAR image: {img_path}")

        if self.in_channels == 3:
            img = build_wakenet_3ch(bgr, clahe=self.clahe, out_size=self.img_size)  # (H, W, 3)
        else:
            img = preprocess_sar_logdb_clahe(bgr, clahe=self.clahe, out_size=self.img_size)  # (H, W)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k).copy()
            if np.random.rand() < 0.35:
                alpha_c = np.random.uniform(0.88, 1.12)
                beta_c = np.random.uniform(-10.0, 10.0)
                img = np.clip(img.astype(np.float32) * alpha_c + beta_c, 0, 255).astype(np.uint8)

        if self.in_channels == 3:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        else:
            img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        w = float(row["sample_weight"]) if "sample_weight" in row and pd.notna(row["sample_weight"]) else 1.0

        return img_tensor, mask_tensor, torch.tensor(w, dtype=torch.float32)


class TverskyDiceBCELoss(nn.Module):
    """
    Stabilized batch-global Focal/BCE + Tversky + Dice loss for extreme ocean:wake imbalance.
    Supports per-sample weights for semi-supervised pseudo-label adaptation.
    """
    def __init__(self, pos_weight=2.0, alpha=0.45, beta=0.55, smooth=1.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets, sample_weights=None):
        pw = torch.tensor([self.pos_weight_val], device=logits.device, dtype=logits.dtype)
        bce_map = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        if sample_weights is not None:
            sw = sample_weights.view(-1, 1, 1, 1).to(logits.dtype)
            bce_loss = (bce_map * sw).sum() / (sw.sum() * logits.shape[2] * logits.shape[3] + 1e-6)
        else:
            bce_loss = bce_map.mean()
            sw = 1.0

        probs = torch.sigmoid(logits)
        tp = (probs * targets * sw).sum()
        fp = (probs * (1.0 - targets) * sw).sum()
        fn = ((1.0 - probs) * targets * sw).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        return 0.30 * bce_loss + 0.35 * (1.0 - tversky) + 0.35 * (1.0 - dice)


def build_wake_net_model(encoder_name="resnet50", in_channels=3, classes=1, pretrained=True):
    """
    Constructs Wake-Net: ResNet-50 encoder with concurrent Spatial & Channel Squeeze-and-Excitation (scSE)
    decoder attention blocks (Paper 1: Zhang et al., 2023).
    """
    weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=weights,
        decoder_attention_type="scse",
        in_channels=in_channels,
        classes=classes
    )
    return model


_THETA_DEG = np.arange(0, 180, 1.0)
_THETA_RAD = np.radians(_THETA_DEG)
_COS_T = np.cos(_THETA_RAD)[None, :]
_SIN_T = np.sin(_THETA_RAD)[None, :]


def _fast_radon_dominant_angle(binary_region, bin_width=2.5):
    """
    Vectorized sparse Radon peak detector (~100x faster than skimage.transform.radon on binary masks).
    """
    ys, xs = np.where(binary_region > 0)
    if len(xs) < 20:
        return 0.0
    h, w = binary_region.shape
    cx, cy = w * 0.5, h * 0.5
    if len(xs) > 3000:
        step = max(1, len(xs) // 3000)
        xs = xs[::step]
        ys = ys[::step]

    proj = ((xs - cx)[:, None] * _COS_T - (ys - cy)[:, None] * _SIN_T) / bin_width
    bins = np.round(proj).astype(np.int32)
    bins -= bins.min()

    best_score = -1
    best_idx = 0
    for ti in range(len(_THETA_DEG)):
        peak = int(np.bincount(bins[:, ti]).max())
        if peak > best_score:
            best_score = peak
            best_idx = ti
    return float(_THETA_DEG[best_idx])


def compute_radon_wake_kinematics(binary_mask, sog_knots=None, pixel_res_meters=10.0, ship_point=None):
    """
    Orientation-invariant Hybridization: Applies morphological denoising + fast sparse Radon Transform
    along the true principal wake axis (splitting proximal near-ship vs distal wake tail rather than fixed y//2)
    to compute dominant orientation theta and signed angular deflection Delta theta.
    """
    bool_clean = remove_small_objects(binary_mask > 127, min_size=30)
    clean_mask = (bool_clean.astype(np.uint8)) * 255
    h, w = clean_mask.shape

    ys, xs = np.where(clean_mask > 0)
    if len(xs) < 40:
        return {
            "dominant_angle_deg": 0.0,
            "angular_deflection_deg": 0.0,
            "estimated_kappa": 0.0,
            "estimated_rot": 0.0,
            "is_valid": False
        }

    dominant_angle_deg = _fast_radon_dominant_angle(clean_mask)

    # Orient wake axis from ship center toward tail (invariant to horizontal/vertical wake heading)
    if ship_point is None:
        ship_point = (w * 0.5, h * 0.5)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean_pt = pts.mean(axis=0)
    centered = pts - mean_pt
    cov = np.cov(centered.T) if len(pts) > 2 else np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]

    # Ensure principal vector points away from ship_point into the wake tail
    s_proj = centered @ principal
    s_ship = (np.array([ship_point[0], ship_point[1]]) - mean_pt) @ principal
    if np.median(s_proj) < s_ship:
        principal = -principal
        s_proj = -s_proj

    s_med = float(np.median(s_proj))
    prox_mask = np.zeros_like(clean_mask)
    dist_mask = np.zeros_like(clean_mask)
    prox_idx = s_proj <= s_med
    dist_idx = s_proj > s_med
    prox_mask[ys[prox_idx], xs[prox_idx]] = 255
    dist_mask[ys[dist_idx], xs[dist_idx]] = 255

    if np.sum(prox_mask > 0) > 20 and np.sum(dist_mask > 0) > 20:
        theta_prox = _fast_radon_dominant_angle(prox_mask)
        theta_dist = _fast_radon_dominant_angle(dist_mask)

        deflection = (theta_dist - theta_prox + 180.0) % 180.0
        if deflection > 90.0:
            deflection -= 180.0
        angular_deflection_deg = float(deflection)
    else:
        angular_deflection_deg = 0.0

    wake_span_px = float(np.percentile(s_proj, 95) - np.percentile(s_proj, 5))
    delta_s_meters = max(100.0, 0.5 * wake_span_px * pixel_res_meters)
    kappa_est = np.radians(angular_deflection_deg) / delta_s_meters

    rot_est = 0.0
    if sog_knots is not None and sog_knots > 1.0:
        speed_ms = sog_knots * 0.514444
        rot_est = float(speed_ms * kappa_est * (180.0 / np.pi) * 60.0)

    return {
        "dominant_angle_deg": dominant_angle_deg,
        "angular_deflection_deg": angular_deflection_deg,
        "estimated_kappa": float(kappa_est),
        "estimated_rot": rot_est,
        "is_valid": True
    }


def calculate_metrics(preds_prob, targets, threshold=0.5):
    preds_bin = (preds_prob > threshold).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    intersection = np.sum((preds_bin == 1) & (targets_bin == 1))
    union = np.sum((preds_bin == 1) | (targets_bin == 1))
    pred_sum = np.sum(preds_bin == 1)
    target_sum = np.sum(targets_bin == 1)

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    precision = (intersection + 1e-6) / (pred_sum + 1e-6)
    recall = (intersection + 1e-6) / (target_sum + 1e-6)

    per_sample_iou = []
    for i in range(len(preds_bin)):
        inter_i = np.sum((preds_bin[i] == 1) & (targets_bin[i] == 1))
        union_i = np.sum((preds_bin[i] == 1) | (targets_bin[i] == 1))
        if union_i > 0:
            per_sample_iou.append(inter_i / union_i)
    det_rate = float(np.mean(np.array(per_sample_iou) >= 0.30)) if per_sample_iou else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "detection_rate_iou30": det_rate
    }


def load_segmentation_splits():
    train_df = pd.read_parquet(os.path.join(PACKAGE_DIR, "train.parquet"))

    val_path = os.path.join(PACKAGE_DIR, "validation.parquet")
    if not os.path.exists(val_path):
        val_path = os.path.join(PACKAGE_DIR, "val.parquet")
    holdout_df = pd.read_parquet(val_path)

    test_path = os.path.join(PACKAGE_DIR, "test.parquet")
    if os.path.exists(test_path):
        return train_df, holdout_df, pd.read_parquet(test_path)

    if "acquisition_group" in holdout_df.columns and holdout_df["acquisition_group"].nunique() >= 2:
        groups = sorted(holdout_df["acquisition_group"].unique())
        mid = len(groups) // 2
        val_groups = set(groups[:mid])
        val_df = holdout_df[holdout_df["acquisition_group"].isin(val_groups)].copy()
        test_df = holdout_df[~holdout_df["acquisition_group"].isin(val_groups)].copy()
    else:
        val_df = holdout_df.iloc[:len(holdout_df) // 2].copy()
        test_df = holdout_df.iloc[len(holdout_df) // 2:].copy()

    return train_df, val_df, test_df


def load_wake_net_checkpoint(ckpt_path, device):
    """
    Loads a Wake-Net checkpoint (3-channel preferred or legacy 1-channel) by inspecting
    encoder.conv1.weight input channel dimension automatically.
    """
    state = torch.load(ckpt_path, map_location=device)
    conv1_w = state.get("encoder.conv1.weight", None)
    in_ch = int(conv1_w.shape[1]) if conv1_w is not None else 3
    model = build_wake_net_model(encoder_name="resnet50", in_channels=in_ch, classes=1, pretrained=False).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, in_ch


def select_high_conf_mmdec_pseudolabels(crop_paths, mask_paths, min_area=140, max_area=6500, min_main_ratio=0.65, min_elongation=4.0, max_keep=500):
    """
    Filters Pass-1 MMDEC predicted masks (strictly from MMDEC train split) for single,
    elongated, high-confidence wake plumes suitable for semi-supervised self-training.
    """
    scored = []
    for cpath, mpath in zip(crop_paths, mask_paths):
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        fg = (m > 127).astype(np.uint8)
        total_area = int(fg.sum())
        if total_area < min_area or total_area > max_area:
            continue
        num_cc, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_cc <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_ratio = float(areas.max()) / float(total_area)
        if main_ratio < min_main_ratio:
            continue
        ys, xs = np.where(fg > 0)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        evals = np.linalg.eigvalsh(np.cov(pts.T))
        elong = float(evals.max()) / (float(evals.min()) + 1.0)
        if elong < min_elongation:
            continue
        scored.append((main_ratio * np.log1p(total_area), cpath, mpath))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(c, m) for _, c, m in scored[:max_keep]]


def adapt_wakenet_on_mmdec_pseudolabels(ckpt_path, pseudo_pairs, epochs=8, batch_size=8, lr=1e-4, device=None):
    """
    Semi-supervised domain adaptation: mixes UEIKAP human-annotated wakes (weight=1.0)
    with strictly train-split high-confidence MMDEC pseudo-masks (weight=0.45) to close
    the Sentinel-1 domain gap on MMDEC wakes without val/test leakage.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, in_ch = load_wake_net_checkpoint(ckpt_path, device)
    if in_ch != 3 or len(pseudo_pairs) < 40:
        return model, ckpt_path

    train_df, val_df, _ = load_segmentation_splits()
    img_dir = os.path.join(PACKAGE_DIR, "images")
    mask_dir = os.path.join(PACKAGE_DIR, "masks")

    uei_rows = [{
        "image_path": os.path.join(img_dir, r["image_filename"]),
        "mask_path": os.path.join(mask_dir, r["mask_filename"]),
        "sample_weight": 1.0,
    } for _, r in train_df.iterrows()]
    mm_rows = [{
        "image_path": cpath,
        "mask_path": mpath,
        "sample_weight": 0.45,
    } for cpath, mpath in pseudo_pairs]
    mix_df = pd.DataFrame(uei_rows + mm_rows)

    mix_loader = DataLoader(WakeDataset(mix_df, augment=True, in_channels=3), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(WakeDataset(val_df, img_dir, mask_dir, augment=False, in_channels=3), batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = TverskyDiceBCELoss(pos_weight=2.0, alpha=0.45, beta=0.55, smooth=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def _eval_val():
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                vp.append(torch.sigmoid(model(imgs.to(device))).cpu().numpy())
                vt.append(masks.numpy())
        vp = np.concatenate(vp, axis=0)
        vt = np.concatenate(vt, axis=0)
        return max(calculate_metrics(vp, vt, threshold=t)["dice"] for t in np.arange(0.30, 0.70, 0.05))

    base_val_dice = _eval_val()
    best_val_dice = base_val_dice * 0.94  # allow slight domain shift without forgetting UEIKAP
    adapted_ckpt = os.path.join(CHECKPOINTS_DIR, "wake_net_3ch_adapted.pt")
    torch.save(model.state_dict(), adapted_ckpt)
    print(f"[WAKE-NET ADAPT] Pseudo-label domain adaptation ({len(uei_rows)} UEIKAP GT + {len(mm_rows)} MMDEC train pseudo, base Val Dice={base_val_dice:.4f})...")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for imgs, masks, sw in mix_loader:
            imgs, masks, sw = imgs.to(device), masks.to(device), sw.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks, sample_weights=sw)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item() * len(imgs)
        scheduler.step()
        v_dice = _eval_val()
        print(f"  Adapt Ep {ep:02d}/{epochs:02d} | Loss: {tot_loss / max(1, len(mix_df)):.4f} | UEIKAP Val Sweep Dice: {v_dice:.4f}")
        if v_dice >= best_val_dice:
            best_val_dice = v_dice
            torch.save(model.state_dict(), adapted_ckpt)

    model.load_state_dict(torch.load(adapted_ckpt, map_location=device))
    model.eval()
    return model, adapted_ckpt


def train_wake_net(epochs=35, batch_size=8, lr=2e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WAKE-NET] Training 3-Ch ResNet-50 + scSE Attention (Unified SAR + CLAHE + HighPass) on device: {device}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_df, val_df, test_df = load_segmentation_splits()
    print(f"[WAKE-NET] Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    img_dir = os.path.join(PACKAGE_DIR, "images")
    mask_dir = os.path.join(PACKAGE_DIR, "masks")

    train_loader = DataLoader(WakeDataset(train_df, img_dir, mask_dir, augment=True, in_channels=3), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(WakeDataset(val_df, img_dir, mask_dir, augment=False, in_channels=3), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(WakeDataset(test_df, img_dir, mask_dir, augment=False, in_channels=3), batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_wake_net_model(encoder_name="resnet50", in_channels=3, classes=1, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = TverskyDiceBCELoss(pos_weight=2.0, alpha=0.45, beta=0.55, smooth=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = 0.0
    history = []
    ckpt_3ch = os.path.join(CHECKPOINTS_DIR, "wake_net_3ch_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks, sw in train_loader:
            imgs, masks, sw = imgs.to(device), masks.to(device), sw.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks, sample_weights=sw)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(imgs)

        train_loss /= max(1, len(train_loader.dataset))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, masks, sw in val_loader:
                imgs, masks, sw = imgs.to(device), masks.to(device), sw.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks, sample_weights=sw)
                val_loss += loss.item() * len(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(masks.cpu().numpy())

        val_loss /= max(1, len(val_loader.dataset))
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        metrics = calculate_metrics(val_preds, val_targets, threshold=0.5)
        best_ep_dice = max(calculate_metrics(val_preds, val_targets, threshold=t)["dice"] for t in np.arange(0.30, 0.70, 0.05))

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice@0.5: {metrics['dice']:.4f} | BestSweep: {best_ep_dice:.4f} | IoU: {metrics['iou']:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice_sweep": float(best_ep_dice), **metrics})

        if best_ep_dice > best_val_dice:
            best_val_dice = best_ep_dice
            torch.save(model.state_dict(), ckpt_3ch)
            print(f" -> Best 3-Ch Wake-Net checkpoint saved (Sweep Dice: {best_val_dice:.4f})")

    print("\n[WAKE-NET EVALUATION ON UNSEEN UEIKAP TEST SET]")
    model.load_state_dict(torch.load(ckpt_3ch, map_location=device))
    model.eval()

    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(masks.numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    best_thresh = 0.5
    best_v_dice = 0.0
    for t in np.arange(0.15, 0.75, 0.05):
        m = calculate_metrics(val_preds, val_targets, threshold=t)
        if m["dice"] > best_v_dice:
            best_v_dice = m["dice"]
            best_thresh = float(t)

    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(masks.numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    final_test_metrics = calculate_metrics(test_preds, test_targets, threshold=best_thresh)
    final_test_metrics["optimal_val_threshold"] = best_thresh

    print(f"Calibrated Val Threshold: {best_thresh:.2f}")
    print(f"Test IoU:                 {final_test_metrics['iou']:.4f}")
    print(f"Test Dice (F1):           {final_test_metrics['dice']:.4f}")
    print(f"Test Detection Rate:      {final_test_metrics['detection_rate_iou30']*100:.1f}%")
    print(f"Test Precision / Recall:  {final_test_metrics['precision']:.4f} / {final_test_metrics['recall']:.4f}")

    # Fast Sparse Radon Kinematic validation (< 1s total!)
    angular_errors = []
    for i in range(len(test_preds)):
        pred_mask = (test_preds[i, 0] > best_thresh).astype(np.uint8) * 255
        gt_mask = (test_targets[i, 0] > 0.5).astype(np.uint8) * 255

        kin_pred = compute_radon_wake_kinematics(pred_mask)
        kin_gt = compute_radon_wake_kinematics(gt_mask)

        if kin_pred["is_valid"] and kin_gt["is_valid"]:
            err = abs(kin_pred["dominant_angle_deg"] - kin_gt["dominant_angle_deg"])
            err = min(err, 180 - err)
            angular_errors.append(err)

    if angular_errors:
        mean_ang_err = float(np.mean(angular_errors))
        print(f"Mean Radon Wake Orientation Error: {mean_ang_err:.2f} deg")
        final_test_metrics["mean_radon_angular_error_deg"] = mean_ang_err

    with open(os.path.join(LOGS_DIR, "wake_net_evaluation.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    return final_test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Wake-Net (ResNet-50 + scSE + Fast Radon)")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train_wake_net(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
