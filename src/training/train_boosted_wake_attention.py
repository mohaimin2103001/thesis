"""
================================================================================
BOOSTED WAKE-ATTENTION & PHYSICS FUSION TRAINING PIPELINE
================================================================================
Eliminates Modality Collapse and Background Clutter Spurious Correlations.
1. Employs Spatial Wake Mask Guidance (anchoring CNN receptive field to the wake).
2. Uses WeightedRandomSampler (50/50 Straight vs Turning balanced mini-batches).
3. Adds Spatial Attention Alignment Loss (forcing Layer 4 activations to match wake).
4. Multi-Task Focal + Kinematic Physics Loss.
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score, recall_score

# Set single thread for OpenBLAS to prevent memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "boosted_wake_attention")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GEOMETRY_COLS = [
    "wake_length", "wake_orientation", "mean_curvature", "median_curvature",
    "max_abs_curvature", "signed_mean_curvature", "curvature_std",
    "wake_asymmetry", "mask_area", "geometry_confidence"
]


def resolve_path(p):
    if not p or not isinstance(p, str):
        return p
    if os.path.exists(p):
        return p
    fname = os.path.basename(p.replace("\\", "/"))
    for base in [
        os.path.join(PROJECT_ROOT, "data", "raw", "UEIKAP_SAR", "images"),
        os.path.join(PROJECT_ROOT, "data", "interim", "ueikap_masks"),
        os.path.join(PROJECT_ROOT, "data", "raw"),
        os.path.join(PROJECT_ROOT, "data", "interim")
    ]:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if fname in files:
                    return os.path.join(root, fname)
    return p


class BoostedWakeDataset(Dataset):
    def __init__(self, df, geom_mean, geom_std, img_size=(256, 256), is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train

        geom_vals = self.df[GEOMETRY_COLS].values.astype(np.float32)
        self.geom_norm = (geom_vals - geom_mean) / geom_std
        self.geom_norm = np.nan_to_num(self.geom_norm, nan=0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Image
        img_p = resolve_path(row["original_image_path"])
        img = cv2.imread(img_p)
        if img is None:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, self.img_size)

        # 2. Wake Mask
        mask_p = resolve_path(row["wake_mask_path"])
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask > 127).astype(np.float32)
        else:
            mask_bin = np.zeros(self.img_size, dtype=np.float32)

        # SPATIAL WAKE ATTENTION GUIDANCE:
        # Suppress open ocean background noise by 85%, keep 100% of wake
        wake_weight = 0.85 * mask_bin[:, :, np.newaxis] + 0.15
        img_guided = (img.astype(np.float32) * wake_weight).astype(np.uint8)

        # Normalization
        img_t = torch.from_numpy(img_guided.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0) # 1, H, W
        geom_t = torch.from_numpy(self.geom_norm[idx])

        state_lbl = 1 if row["state"] == "Turning" else 0
        dir_lbl = 0 if row["direction"] == "Port" else (1 if row["direction"] == "Starboard" else -1)
        rot_tgt = float(row["rot_target"]) if pd.notnull(row["rot_target"]) else 0.0

        return {
            "image": img_t,
            "mask": mask_t,
            "geometry": geom_t,
            "state": torch.tensor(state_lbl, dtype=torch.long),
            "direction": torch.tensor(dir_lbl, dtype=torch.long),
            "rot": torch.tensor(rot_tgt, dtype=torch.float32)
        }


class BoostedWakeAttentionFusion(nn.Module):
    def __init__(self, visual_backbone="resnet18", geom_dim=10, fused_dim=128):
        super().__init__()
        base = getattr(models, visual_backbone)(weights=models.ResNet18_Weights.DEFAULT)
        vis_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        # Hook layer 4 for spatial attention alignment
        self.layer4 = base.layer4

        # Geometry MLP
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Gated Multi-Modal Cross-Attention Fusion
        self.fusion_linear = nn.Linear(vis_dim + 64, fused_dim)
        self.fusion_bn = nn.BatchNorm1d(fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(vis_dim + 64, fused_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.25)

        # Multi-task heads
        self.head_state = nn.Linear(fused_dim, 2)
        self.head_dir = nn.Linear(fused_dim, 2)
        self.head_rot = nn.Linear(fused_dim, 1)

    def forward(self, img, geom):
        # Extract visual feature map from layer 4 before global avg pool
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feat_map = self.backbone.layer4(x) # (B, 512, H_feat, W_feat)

        f_vis = self.backbone.avgpool(feat_map)
        f_vis = torch.flatten(f_vis, 1) # (B, 512)

        f_geom = self.geom_mlp(geom) # (B, 64)

        cat_feat = torch.cat([f_vis, f_geom], dim=1)
        z = F.relu(self.fusion_bn(self.fusion_linear(cat_feat)))
        g = self.gate(cat_feat)
        fused = self.dropout(z * g)

        state_logits = self.head_state(fused)
        dir_logits = self.head_dir(fused)
        rot_pred = self.head_rot(fused)

        return state_logits, dir_logits, rot_pred, feat_map


def train_boosted_model(epochs=25, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Boosted Wake-Attention Training on {device} ===")

    train_df = pd.read_parquet(os.path.join(SPLITS_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(SPLITS_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(SPLITS_DIR, "test.parquet"))

    geom_train = train_df[GEOMETRY_COLS].values.astype(np.float32)
    geom_mean = np.nanmean(geom_train, axis=0)
    geom_std = np.nanstd(geom_train, axis=0) + 1e-6

    train_dataset = BoostedWakeDataset(train_df, geom_mean, geom_std, is_train=True)
    val_dataset = BoostedWakeDataset(val_df, geom_mean, geom_std, is_train=False)
    test_dataset = BoostedWakeDataset(test_df, geom_mean, geom_std, is_train=False)

    # 50/50 Balanced Sampling to eliminate class imbalance
    state_labels = (train_df["state"] == "Turning").values.astype(int)
    n_straight = np.sum(state_labels == 0)
    n_turning = np.sum(state_labels == 1)
    class_weights = [1.0 / n_straight, 1.0 / n_turning]
    sample_weights = [class_weights[lbl] for lbl in state_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = BoostedWakeAttentionFusion(visual_backbone="resnet18", geom_dim=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Focal loss for state classification
    def focal_loss(logits, targets, gamma=2.0, alpha=0.5):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        f_loss = alpha * ((1.0 - pt) ** gamma) * ce_loss
        return f_loss.mean()

    criterion_dir = nn.CrossEntropyLoss()
    criterion_rot = nn.SmoothL1Loss()

    best_val_score = -1.0
    best_path = os.path.join(CHECKPOINT_DIR, "boosted_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            geoms = batch["geometry"].to(device)
            states = batch["state"].to(device)
            dirs = batch["direction"].to(device)
            rots = batch["rot"].to(device).unsqueeze(1)

            optimizer.zero_grad()

            state_logits, dir_logits, rot_pred, feat_map = model(imgs, geoms)

            # 1. State Focal Loss
            loss_state = focal_loss(state_logits, states, gamma=2.0)

            # 2. Direction Loss on Turning Ships
            turn_mask = (states == 1) & (dirs >= 0)
            if turn_mask.sum() > 0:
                loss_dir = criterion_dir(dir_logits[turn_mask], dirs[turn_mask])
            else:
                loss_dir = 0.0

            # 3. ROT Regression Loss
            loss_rot = criterion_rot(rot_pred, rots)

            # 4. Spatial Attention Alignment Loss (Force CNN to activate on Wake Mask)
            # Downsample wake mask to feature map size
            h_f, w_f = feat_map.shape[2], feat_map.shape[3]
            mask_down = F.adaptive_avg_pool2d(masks, (h_f, w_f)) # (B, 1, h_f, w_f)
            # Feature map activation across channels
            feat_act = torch.mean(feat_map, dim=1, keepdim=True) # (B, 1, h_f, w_f)
            feat_act = torch.sigmoid(feat_act)
            loss_attn = F.mse_loss(feat_act, mask_down)

            loss = loss_state + 0.6 * loss_dir + 0.4 * loss_rot + 0.5 * loss_attn
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_preds_state = []
        val_targets_state = []
        val_probs_turn = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                geoms = batch["geometry"].to(device)
                states = batch["state"].to(device)
                state_logits, _, _, _ = model(imgs, geoms)
                probs = torch.softmax(state_logits, dim=1)[:, 1]
                val_probs_turn.extend(probs.cpu().numpy())
                val_targets_state.extend(states.cpu().numpy())

        val_targets_state = np.array(val_targets_state)
        val_probs_turn = np.array(val_probs_turn)
        # Find best threshold on validation set
        taus = np.linspace(0.1, 0.7, 31)
        best_t_epoch = 0.5
        best_bal_epoch = 0.0
        for t in taus:
            b_acc = balanced_accuracy_score(val_targets_state, (val_probs_turn >= t).astype(int))
            if b_acc > best_bal_epoch:
                best_bal_epoch = b_acc
                best_t_epoch = t

        print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Val BalAcc: {best_bal_epoch:.4f} (best tau={best_t_epoch:.2f})")

        if best_bal_epoch > best_val_score:
            best_val_score = best_bal_epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_bal_acc": best_val_score,
                "best_tau": best_t_epoch,
                "geom_mean": geom_mean,
                "geom_std": geom_std
            }, best_path)

    print(f"\nTraining Complete! Best model saved to: {best_path} (Val BalAcc = {best_val_score:.4f})")
    return best_path


if __name__ == "__main__":
    t0 = time.time()
    best_ckpt = train_boosted_model(epochs=22, lr=3e-4)
    print(f"Total Training Time: {round(time.time() - t0, 1)} seconds.")

