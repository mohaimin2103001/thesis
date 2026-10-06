"""
Dual-Architecture Ensemble and Test-Time Augmentation (TTA) Benchmark
Combines ResNet-18 (high specificity) and ResNet-34 (high sensitivity) with 4-fold geometric TTA,
reproducing Table I from the research paper (Accuracy: 77.66%, Macro F1: 0.6141).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

from .models import SARWakeBackbone, DualEnsembleWakeClassifier
from .dataset import SARWakeDataset
from torch.utils.data import DataLoader

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))


def evaluate_dual_ensemble(
    chk_r18_path,
    chk_r34_path,
    csv_index_path,
    img_size=384,
    batch_size=16,
    weight_r18=0.45,
    weight_r34=0.55
):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading ResNet-18 and ResNet-34 checkpoints for Dual Ensemble...")

    model_r18 = SARWakeBackbone(architecture="resnet18", num_classes=2, dropout_p=0.0).to(device)
    c18 = torch.load(chk_r18_path, map_location=device, weights_only=False)
    model_r18.load_state_dict(c18["model_state_dict"])
    model_r18.eval()

    model_r34 = SARWakeBackbone(architecture="resnet34", num_classes=2, dropout_p=0.0).to(device)
    c34 = torch.load(chk_r34_path, map_location=device, weights_only=False)
    model_r34.load_state_dict(c34["model_state_dict"])
    model_r34.eval()

    ensemble = DualEnsembleWakeClassifier(model_r18, model_r34, weight_r18=weight_r18, weight_r34=weight_r34)

    df = pd.read_csv(csv_index_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_ds = SARWakeDataset(test_df, img_size=img_size, is_train=False)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_targets = []

    print(f"Running Dual Ensemble with 4-fold TTA on {len(test_df)} test samples...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["binary_label"].numpy()

            blended_probs = ensemble.forward_tta(images)
            preds = torch.argmax(blended_probs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(blended_probs.cpu().numpy())
            all_targets.extend(targets)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    # Specificity = TN / (TN + FP), Sensitivity = TP / (TP + FN)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    auc = roc_auc_score(all_targets, [p[1] for p in all_probs])

    print("=" * 60)
    print("DUAL-ARCHITECTURE ENSEMBLE (+ TTA) TEST PERFORMANCE:")
    print(f"  Overall Accuracy:      {acc*100:.2f}%  (Baseline ResNet-18: 73.40%)")
    print(f"  Balanced Accuracy:     {bal_acc*100:.2f}%  (Baseline: 57.17%)")
    print(f"  Macro F1-Score:        {macro_f1:.4f}  (Baseline: 0.5612)")
    print(f"  Specificity (TN Rate): {spec*100:.2f}%  (84.8% target)")
    print(f"  Sensitivity (TP Rate): {sens*100:.2f}%  (40.0% target)")
    print(f"  ROC-AUC:               {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print("=" * 60)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Pred Straight', 'Pred Turning'],
        yticklabels=['Actual Straight', 'Actual Turning'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title("Dual Ensemble (+ TTA) Confusion Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Ensemble confusion matrix saved to: {cm_path}")

    metrics = {
        "architecture": "Dual-Ensemble-ResNet18-ResNet34-TTA",
        "test_acc": float(acc),
        "test_bal_acc": float(bal_acc),
        "test_macro_f1": float(macro_f1),
        "specificity": float(spec),
        "sensitivity": float(sens),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }

    metrics_file = os.path.join(METRICS_DIR, "dual_ensemble_test_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    chk18 = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet18_binary_model.pt")
    chk34 = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet34_binary_model.pt")
    idx = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_index.csv")
    if os.path.exists(chk18) and os.path.exists(chk34) and os.path.exists(idx):
        evaluate_dual_ensemble(chk18, chk34, idx)
    else:
        print("Required model checkpoints or index file not found. Train models first.")
