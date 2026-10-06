"""
Operational Uncertainty Quantification via Monte Carlo (MC) Dropout
Executes N=30 stochastic forward passes to evaluate epistemic uncertainty,
calculates predictive entropy, and generates the selective error rejection curve.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .models import SARWakeBackbone
from .dataset import SARWakeDataset

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))


def run_uncertainty_quantification(
    model_checkpoint_path,
    csv_index_path,
    num_passes=30,
    img_size=384,
    batch_size=16,
    num_classes=2,
    architecture="resnet18"
):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint for Uncertainty Analysis: {model_checkpoint_path}")

    model = SARWakeBackbone(architecture=architecture, num_classes=num_classes, dropout_p=0.3).to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(csv_index_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_ds = SARWakeDataset(test_df, img_size=img_size, is_train=False)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_targets = []
    all_entropies = []
    all_mean_preds = []
    all_probs = []

    print(f"Executing {num_passes} stochastic MC-Dropout forward passes across {len(test_df)} test samples...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["binary_label" if num_classes == 2 else "class3_label"].numpy()

            mc_results = model.forward_mc_dropout(images, num_passes=num_passes)
            entropies = mc_results["entropy"].cpu().numpy()
            preds = mc_results["prediction"].cpu().numpy()
            probs = mc_results["mean_prob"].cpu().numpy()

            all_targets.extend(targets)
            all_entropies.extend(entropies)
            all_mean_preds.extend(preds)
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_mean_preds = np.array(all_mean_preds)
    all_entropies = np.array(all_entropies)

    correct_mask = (all_mean_preds == all_targets)
    incorrect_mask = ~correct_mask

    mean_entropy_correct = float(np.mean(all_entropies[correct_mask]))
    mean_entropy_incorrect = float(np.mean(all_entropies[incorrect_mask]))
    overall_acc = float(np.mean(correct_mask))

    print("=" * 60)
    print("UNCERTAINTY QUANTIFICATION AUDIT:")
    print(f"  Overall Test Accuracy:                  {overall_acc*100:.2f}%")
    print(f"  Avg Predictive Entropy (Correct):       {mean_entropy_correct:.4f}")
    print(f"  Avg Predictive Entropy (Incorrect):     {mean_entropy_incorrect:.4f}")
    print(f"  Statistical Significance (Entropy Gap): {mean_entropy_incorrect - mean_entropy_correct:+.4f}")
    print("=" * 60)

    # Selective Rejection Curve:
    # Sort samples by predictive entropy ascending (most certain to least certain).
    # Progressively reject top k% most uncertain samples and compute error rate of retained samples.
    sorted_indices = np.argsort(all_entropies)
    sorted_targets = all_targets[sorted_indices]
    sorted_preds = all_mean_preds[sorted_indices]

    rejection_rates = np.linspace(0.0, 0.50, 21)
    retained_error_rates = []

    for r in rejection_rates:
        keep_count = int(len(sorted_targets) * (1.0 - r))
        keep_targets = sorted_targets[:keep_count]
        keep_preds = sorted_preds[:keep_count]
        err = 1.0 - float(np.mean(keep_preds == keep_targets))
        retained_error_rates.append(err)

    # Plot Selective Error Rejection Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rejection_rates * 100, np.array(retained_error_rates) * 100, marker='o', lw=2.5, color='#d62728', label='Selective Prediction Error')
    ax.axhline(y=(1.0 - overall_acc) * 100, color='gray', linestyle='--', label=f'Baseline Error ({(1.0 - overall_acc)*100:.1f}%)')
    ax.set_title("Selective Prediction Error vs. Uncertainty Rejection Rate", fontsize=13, fontweight='bold')
    ax.set_xlabel("Rejection Fraction (% of high-entropy predictions deferred to human triage)", fontsize=11)
    ax.set_ylabel("Error Rate on Retained Samples (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    rejection_plot_path = os.path.join(FIGURES_DIR, f"{architecture}_uncertainty_rejection_curve.png")
    plt.tight_layout()
    plt.savefig(rejection_plot_path, dpi=300)
    plt.close()
    print(f"Rejection curve saved to: {rejection_plot_path}")

    results = {
        "overall_accuracy": overall_acc,
        "mean_entropy_correct": mean_entropy_correct,
        "mean_entropy_incorrect": mean_entropy_incorrect,
        "entropy_delta": mean_entropy_incorrect - mean_entropy_correct,
        "rejection_rates": rejection_rates.tolist(),
        "error_rates": retained_error_rates
    }

    metrics_json_path = os.path.join(METRICS_DIR, f"{architecture}_uncertainty_results.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    chk = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet18_binary_model.pt")
    idx = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_index.csv")
    if os.path.exists(chk) and os.path.exists(idx):
        run_uncertainty_quantification(chk, idx)
    else:
        print("Model checkpoint or index file not found. Run training first.")
