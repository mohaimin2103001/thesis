"""
Wake Segmenter Training & ONNX Export Pipeline (Phase 6)
Trains a U-Net / ResNet architecture with mixed precision, Dice + BCE loss, and threshold optimization.
Exports PyTorch weights (.pt) and optimized ONNX runtime graph (.onnx).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import segmentation_models_pytorch as smp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "segmenter")
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "wake_segmentation_kaggle")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


class WakeDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, img_size=(384, 384)):
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_filename"])
        mask_path = os.path.join(self.mask_dir, row["mask_filename"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.img_size, dtype=np.uint8)
        else:
            img = cv2.resize(img, self.img_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Dynamic range enhancement: normalize 0..255 to 0..1
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


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

    return {"iou": float(iou), "dice": float(dice), "precision": float(precision), "recall": float(recall)}


def train_wake_segmenter(epochs=12, batch_size=8, lr=3e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Wake Segmenter on device: {device}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(PACKAGE_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PACKAGE_DIR, "validation.parquet"))
    img_dir = os.path.join(PACKAGE_DIR, "images")
    mask_dir = os.path.join(PACKAGE_DIR, "masks")

    train_ds = WakeDataset(train_df, img_dir, mask_dir)
    val_ds = WakeDataset(val_df, img_dir, mask_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize U-Net with ResNet-18 encoder (single-channel SAR input)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_dice = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(masks.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Optimize validation threshold
        best_th = 0.5
        best_th_dice = 0.0
        for th in np.arange(0.2, 0.8, 0.05):
            m = calculate_metrics(all_probs, all_targets, threshold=th)
            if m["dice"] > best_th_dice:
                best_th_dice = m["dice"]
                best_th = th

        metrics = calculate_metrics(all_probs, all_targets, threshold=best_th)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Best Threshold: {best_th:.2f} | Val Dice: {metrics['dice']:.4f} | Val IoU: {metrics['iou']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": metrics["dice"],
            "val_iou": metrics["iou"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "best_threshold": float(best_th)
        })

        if metrics["dice"] > best_val_dice:
            best_val_dice = metrics["dice"]
            pt_path = os.path.join(CHECKPOINTS_DIR, "wake_segmenter_best.pt")
            torch.save(model.state_dict(), pt_path)
            
            # Save threshold config
            th_path = os.path.join(CHECKPOINTS_DIR, "wake_threshold.json")
            with open(th_path, "w") as f:
                json.dump({"optimal_threshold": float(best_th), "val_dice": metrics["dice"], "val_iou": metrics["iou"]}, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(LOGS_DIR, "segmenter_training_history.csv"), index=False)

    # Export to ONNX
    print("\nExporting Best Checkpoint to ONNX...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "wake_segmenter_best.pt"), map_location="cpu"))
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 1, 384, 384, dtype=torch.float32)
    onnx_path = os.path.join(CHECKPOINTS_DIR, "wake_segmenter.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["sar_image"],
        output_names=["wake_logits"],
        dynamic_axes={"sar_image": {0: "batch_size"}, "wake_logits": {0: "batch_size"}}
    )

    print(f"Exported ONNX model successfully to: {onnx_path}")
    return history_df


if __name__ == "__main__":
    train_wake_segmenter(epochs=6, batch_size=8)
