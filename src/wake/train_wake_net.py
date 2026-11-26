"""
Hybrid Wake-Net: Attention U-Net with scSE & Fast Sparse Radon Transform Kinematic Extraction
Implements Paper 1 (Wake-Net: Zhang et al., IEEE TGRS 2023) with Spatial & Channel
Squeeze-and-Excitation (scSE) attention + Paper 3 (UEIKAP: Graziano et al.) Radon Transform
Centerline Fitting and Multi-Window Angular Curvature Analysis.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
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


def preprocess_sar_logdb_clahe(img_gray, clahe=None, out_size=(384, 384)):
    """Log-dB percentile SAR contrast stretch [p1, p99.5] + CLAHE normalizing dark UEIKAP & MMDEC chips."""
    if out_size is not None and (img_gray.shape[1], img_gray.shape[0]) != tuple(out_size):
        img_gray = cv2.resize(img_gray, out_size)
    log_g = np.log1p(img_gray.astype(np.float32))
    lo, hi = np.percentile(log_g[::4, ::4], [1.0, 99.5])
    if hi - lo > 1e-4:
        stretched = np.clip((log_g - lo) / (hi - lo), 0.0, 1.0)
    else:
        stretched = img_gray.astype(np.float32) / 255.0
    u8 = (stretched * 255.0).astype(np.uint8)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(u8)


class WakeDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, img_size=(384, 384), augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_filename"])
        mask_path = os.path.join(self.mask_dir, row["mask_filename"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing SAR image: {img_path}")
        img = preprocess_sar_logdb_clahe(img, clahe=self.clahe, out_size=self.img_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
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
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            if np.random.rand() < 0.35:
                alpha_c = np.random.uniform(0.88, 1.12)
                beta_c = np.random.uniform(-10.0, 10.0)
                img = np.clip(img.astype(np.float32) * alpha_c + beta_c, 0, 255).astype(np.uint8)

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor


class TverskyDiceBCELoss(nn.Module):
    """
    Stabilized batch-global Focal/BCE + Tversky + Dice loss for extreme ocean:wake imbalance.
    Batch-global reduction avoids gradient spikes from sparse/tiny wake masks while balanced
    pos_weight keeps predicted probabilities well-calibrated around 0.5.
    """
    def __init__(self, pos_weight=2.0, alpha=0.45, beta=0.55, smooth=1.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        pw = torch.tensor([self.pos_weight_val], device=logits.device, dtype=logits.dtype)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

        probs = torch.sigmoid(logits)
        # Batch-global reduction prevents tiny per-sample target denominator spikes
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        return 0.30 * bce_loss + 0.35 * (1.0 - tversky) + 0.35 * (1.0 - dice)


def build_wake_net_model(encoder_name="resnet50", in_channels=1, classes=1, pretrained=True):
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


def train_wake_net(epochs=35, batch_size=8, lr=2e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WAKE-NET] Training ResNet-50 + scSE Attention (CLAHE + Stabilized Batch-Global Tversky/Dice) on device: {device}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_df, val_df, test_df = load_segmentation_splits()
    print(f"[WAKE-NET] Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    img_dir = os.path.join(PACKAGE_DIR, "images")
    mask_dir = os.path.join(PACKAGE_DIR, "masks")

    train_loader = DataLoader(WakeDataset(train_df, img_dir, mask_dir, augment=True), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(WakeDataset(val_df, img_dir, mask_dir, augment=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WakeDataset(test_df, img_dir, mask_dir, augment=False), batch_size=batch_size, shuffle=False)

    model = build_wake_net_model(encoder_name="resnet50", in_channels=1, classes=1, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = TverskyDiceBCELoss(pos_weight=2.0, alpha=0.45, beta=0.55, smooth=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)
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
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
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
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "wake_net_best.pt"))
            print(f" -> Best Wake-Net checkpoint saved (Sweep Dice: {best_val_dice:.4f})")

    print("\n[WAKE-NET EVALUATION ON UNSEEN UEIKAP TEST SET]")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "wake_net_best.pt"), map_location=device))
    model.eval()

    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
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
        for imgs, masks in test_loader:
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
