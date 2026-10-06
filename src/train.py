"""
Deep Learning Training Pipeline with Mixed Precision, Cosine Annealing, and Focal Loss
Trains ResNet-18, ResNet-34, and ConvNeXt-Tiny backbones for SAR Ship Wake Analysis.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .dataset import get_dataloaders
from .models import SARWakeBackbone
from .losses import FocalLoss, MultiTaskKinematicsLoss

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, multi_task=False, target_key="binary_label"):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch[target_key].to(device)
        kinematics = batch["kinematics"].to(device)

        optimizer.zero_grad()

        use_cuda = device.type == "cuda"
        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
            if multi_task:
                logits, pred_kin = model(images)
                loss, _, _, _ = criterion(logits, pred_kin, targets, kinematics)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        if scaler and use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, multi_task=False, target_key="binary_label", num_classes=2):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch[target_key].to(device)
            kinematics = batch["kinematics"].to(device)

            if multi_task:
                logits, pred_kin = model(images)
                loss, _, _, _ = criterion(logits, pred_kin, targets, kinematics)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_bal_acc = balanced_accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    val_auc = 0.5
    if num_classes == 2:
        try:
            val_auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
        except Exception:
            pass

    return {
        "loss": val_loss,
        "acc": val_acc,
        "bal_acc": val_bal_acc,
        "macro_f1": val_macro_f1,
        "auc": val_auc,
        "preds": all_preds,
        "targets": all_targets,
        "probs": all_probs
    }


def plot_training_curves(history, save_path, title_prefix="ResNet-18"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0].set_title(f"{title_prefix} - Focal Loss Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Balanced Accuracy
    axes[1].plot(history["val_bal_acc"], label="Val Balanced Acc", color="#2ca02c", lw=2)
    axes[1].set_title(f"{title_prefix} - Balanced Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Acc")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Macro F1
    axes[2].plot(history["val_macro_f1"], label="Val Macro F1", color="#d62728", lw=2)
    axes[2].set_title(f"{title_prefix} - Macro F1-Score Progression", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def train_model(
    architecture="resnet18",
    mode="binary",
    epochs=25,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-2,
    img_size=384,
    multi_task=False,
    index_csv=None
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    if index_csv is None:
        index_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_index.csv"))

    num_classes = 2 if mode == "binary" else 3
    target_key = "binary_label" if mode == "binary" else "class3_label"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training for {architecture.upper()} [{mode.upper()}] on device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        index_csv, img_size=img_size, batch_size=batch_size, mode="polarimetric_multichannel"
    )

    # Class-weighted Focal Loss parameters
    # For binary: w0=0.657, w1=2.091 -> alpha=[0.25, 0.75]
    # For 3-class: w=[1.0, 8.14, 5.97]
    if mode == "binary":
        alpha = torch.tensor([0.25, 0.75]).to(device)
    else:
        alpha = torch.tensor([0.15, 0.45, 0.40]).to(device)

    if multi_task:
        criterion = MultiTaskKinematicsLoss(alpha=alpha, gamma=2.0)
    else:
        criterion = FocalLoss(alpha=alpha, gamma=2.0)

    model = SARWakeBackbone(
        architecture=architecture,
        num_classes=num_classes,
        dropout_p=0.3,
        multi_task=multi_task
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal_acc": [],
        "val_macro_f1": [],
        "val_auc": []
    }

    best_val_f1 = -1.0
    best_model_path = os.path.join(MODELS_DIR, f"best_{architecture}_{mode}_model.pt")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, multi_task, target_key
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, multi_task, target_key, num_classes
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_bal_acc"].append(val_metrics["bal_acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_auc"].append(val_metrics["auc"])

        ep_duration = time.time() - ep_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.2f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']*100:.2f}% "
              f"BalAcc: {val_metrics['bal_acc']*100:.2f}% F1: {val_metrics['macro_f1']:.4f}")

        # Save checkpoint if Macro F1 improves
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "architecture": architecture,
                "mode": mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics
            }, best_model_path)
            print(f"  --> Saved new best checkpoint with Val F1: {best_val_f1:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

    # Save training curves
    curve_path = os.path.join(FIGURES_DIR, f"{architecture}_{mode}_training_curves.png")
    plot_training_curves(history, curve_path, title_prefix=f"{architecture.upper()} ({mode})")

    # Evaluate on authentic unseen Test Set
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, multi_task, target_key, num_classes)

    print("=" * 60)
    print(f"TEST BENCHMARK RESULTS [{architecture.upper()} - {mode.upper()}]:")
    print(f"  Test Accuracy:        {test_metrics['acc']*100:.2f}%")
    print(f"  Test Balanced Acc:    {test_metrics['bal_acc']*100:.2f}%")
    print(f"  Test Macro F1:        {test_metrics['macro_f1']:.4f}")
    if num_classes == 2:
        print(f"  Test ROC-AUC:         {test_metrics['auc']:.4f}")
    print("=" * 60)

    metrics_json_path = os.path.join(METRICS_DIR, f"{architecture}_{mode}_test_metrics.json")
    pd.Series({
        "architecture": architecture,
        "mode": mode,
        "test_acc": test_metrics["acc"],
        "test_bal_acc": test_metrics["bal_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_auc": test_metrics["auc"],
        "best_val_f1": best_val_f1
    }).to_json(metrics_json_path, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "convnext_tiny"])
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "3class"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--multi_task", action="store_true")
    args = parser.parse_args()

    train_model(
        architecture=args.arch,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        multi_task=args.multi_task
    )
