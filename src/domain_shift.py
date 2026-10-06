"""
Operational Domain Shift Stress-Testing
Evaluates model robustness across synthetic meteo-marine perturbations:
1. Multiplicative Rayleigh Speckle Clutter (rough sea state 5-7)
2. Contrast Dimming (dead-calm waters)
3. Spatial Resolution Degradation (wide-swath coarse satellite modes)
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .models import SARWakeBackbone
from .dataset import preprocess_sar_chip

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))


def apply_speckle_noise(img_np, severity=0.5):
    """
    Simulates high-wind ocean wave clutter using multiplicative Rayleigh / Gamma distributed noise.
    """
    scale = severity * 0.4
    noise = np.random.gamma(shape=1.0 / (scale + 1e-6), scale=scale + 1e-6, size=img_np.shape)
    degraded = np.clip(img_np.astype(np.float32) * noise, 0, 255).astype(np.uint8)
    return degraded


def apply_contrast_dimming(img_np, severity=0.5):
    """
    Simulates ultra-calm sea surface where specular scattering causes backscatter drop.
    """
    factor = 1.0 - (severity * 0.7)  # drops down to 30% contrast
    mean_val = np.mean(img_np)
    dimmed = np.clip((img_np.astype(np.float32) - mean_val) * factor + mean_val * factor, 0, 255).astype(np.uint8)
    return dimmed


def apply_resolution_degradation(img_np, severity=0.5):
    """
    Simulates coarser satellite resolution (e.g., 20m - 40m ground sampling distance).
    """
    h, w = img_np.shape[:2]
    scale = max(0.1, 1.0 - (severity * 0.8))  # downsample down to 20%
    low_res = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def evaluate_domain_shift(
    model_checkpoint_path,
    csv_index_path,
    architecture="resnet18",
    num_classes=2,
    img_size=384,
    levels=5
):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint for Domain Shift Stress-Testing: {model_checkpoint_path}")

    model = SARWakeBackbone(architecture=architecture, num_classes=num_classes, dropout_p=0.0).to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(csv_index_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    severities = np.linspace(0.0, 1.0, levels)
    results = {
        "severities": severities.tolist(),
        "speckle_acc": [],
        "speckle_f1": [],
        "contrast_acc": [],
        "contrast_f1": [],
        "resolution_acc": [],
        "resolution_f1": []
    }

    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    print("Running domain shift evaluations across severity levels...")
    for sev in severities:
        for mode in ["speckle", "contrast", "resolution"]:
            preds = []
            targets = []

            for _, row in test_df.iterrows():
                gt = int(row["maneuver_binary" if num_classes == 2 else "maneuver_3class"])
                enhanced_pil = preprocess_sar_chip(row["image_path"], mode="grayscale_clahe")
                img_np = np.array(enhanced_pil.resize((img_size, img_size)))

                # Apply synthetic perturbation
                if mode == "speckle" and sev > 0:
                    perturbed = apply_speckle_noise(img_np, severity=sev)
                elif mode == "contrast" and sev > 0:
                    perturbed = apply_contrast_dimming(img_np, severity=sev)
                elif mode == "resolution" and sev > 0:
                    perturbed = apply_resolution_degradation(img_np, severity=sev)
                else:
                    perturbed = img_np

                tensor = torch.from_numpy(perturbed).permute(2, 0, 1).float().to(device) / 255.0
                tensor = (tensor - norm_mean) / norm_std
                tensor = tensor.unsqueeze(0)

                with torch.no_grad():
                    out = model(tensor)
                    if isinstance(out, tuple):
                        out = out[0]
                    p = torch.argmax(out, dim=1).item()

                preds.append(p)
                targets.append(gt)

            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average="macro", zero_division=0)

            results[f"{mode}_acc"].append(acc)
            results[f"{mode}_f1"].append(f1)

    print("Domain shift testing complete!")

    # Plot Degradation Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy Plot
    axes[0].plot(severities, np.array(results["speckle_acc"]) * 100, marker='o', lw=2.5, color='#d62728', label='Speckle Noise (Rough Sea)')
    axes[0].plot(severities, np.array(results["contrast_acc"]) * 100, marker='s', lw=2.5, color='#1f77b4', label='Contrast Dimming (Calm Sea)')
    axes[0].plot(severities, np.array(results["resolution_acc"]) * 100, marker='^', lw=2.5, color='#2ca02c', label='Resolution Degradation')
    axes[0].set_title("Classification Accuracy vs. Perturbation Severity", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Domain Perturbation Severity", fontsize=11)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Macro F1 Plot
    axes[1].plot(severities, results["speckle_f1"], marker='o', lw=2.5, color='#d62728', label='Speckle Noise')
    axes[1].plot(severities, results["contrast_f1"], marker='s', lw=2.5, color='#1f77b4', label='Contrast Dimming')
    axes[1].plot(severities, results["resolution_f1"], marker='^', lw=2.5, color='#2ca02c', label='Resolution Degradation')
    axes[1].set_title("Macro F1-Score vs. Perturbation Severity", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Domain Perturbation Severity", fontsize=11)
    axes[1].set_ylabel("Macro F1-Score", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    save_plot = os.path.join(FIGURES_DIR, f"{architecture}_domain_shift_curves.png")
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300)
    plt.close()
    print(f"Domain shift curves saved to: {save_plot}")

    metrics_file = os.path.join(METRICS_DIR, f"{architecture}_domain_shift_results.json")
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    chk = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet18_binary_model.pt")
    idx = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_index.csv")
    if os.path.exists(chk) and os.path.exists(idx):
        evaluate_domain_shift(chk, idx)
    else:
        print("Checkpoint or index not found. Train model first.")
