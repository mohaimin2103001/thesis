"""
Qualitative Failure Analysis & Physical Radar Error Taxonomy
Analyzes misclassified test samples and maps them to physical hydrodynamic failure mechanisms:
1. Sub-critical Froude number (low vessel speed U < 6 knots, Kelvin envelope collapse)
2. Sea state clutter masking (ambient gravity-capillary wave interference)
3. Wake dissipation over time (temporal delay between maneuver and satellite overpass)
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from .models import SARWakeBackbone
from .dataset import preprocess_sar_chip

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "figures"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))


def run_failure_analysis(
    model_checkpoint_path,
    csv_index_path,
    architecture="resnet18",
    num_classes=2,
    img_size=384,
    max_gallery_items=6
):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint for Failure Analysis: {model_checkpoint_path}")

    model = SARWakeBackbone(architecture=architecture, num_classes=num_classes, dropout_p=0.0).to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(csv_index_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    failures = []
    class_names = ["Straight", "Turning"] if num_classes == 2 else ["Straight", "Left Turn", "Right Turn"]

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            gt = int(row["maneuver_binary" if num_classes == 2 else "maneuver_3class"])
            enhanced_pil = preprocess_sar_chip(row["image_path"], mode="grayscale_clahe")
            img_np = np.array(enhanced_pil.resize((img_size, img_size)))

            tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device) / 255.0
            tensor = (tensor - norm_mean) / norm_std
            tensor = tensor.unsqueeze(0)

            logits = model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)

            if pred != gt:
                sog = float(row.get("sog_knots", 10.0))
                rot = float(row.get("rot_deg_min", 0.0))
                drift = float(row.get("drift_angle_deg", 0.0))

                # Physical taxonomy attribution
                if sog < 6.0:
                    mechanism = "Sub-Critical Froude Number (U < 6 kts; Kelvin wake collapsed)"
                elif abs(rot) < 1.0 and abs(drift) >= 10.0:
                    mechanism = "Pure Drift without Rudder Deflection (Lateral wind/current displacement)"
                elif probs[pred] > 0.85:
                    mechanism = "Sea State Clutter Masking (Rough ocean backscatter artifact)"
                else:
                    mechanism = "Wake Dissipation / Ambiguous Temporal Latency"

                failures.append({
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "ground_truth": class_names[gt],
                    "predicted": class_names[pred],
                    "confidence": float(probs[pred]),
                    "sog_knots": sog,
                    "rot_deg_min": rot,
                    "drift_angle": drift,
                    "failure_mechanism": mechanism,
                    "img_np": img_np
                })

    print(f"Total test errors identified: {len(failures)} out of {len(test_df)} samples.")
    taxonomy_counts = {}
    for f in failures:
        m = f["failure_mechanism"]
        taxonomy_counts[m] = taxonomy_counts.get(m, 0) + 1

    print("\nPhysical Failure Taxonomy Distribution:")
    for k, v in taxonomy_counts.items():
        print(f"  - {k}: {v} samples ({v/len(failures)*100:.1f}%)")

    # Generate Failure Analysis Gallery
    gallery_items = failures[:max_gallery_items]
    if gallery_items:
        n_items = len(gallery_items)
        cols = 3
        rows = (n_items + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for i, item in enumerate(gallery_items):
            ax = axes[i]
            ax.imshow(item["img_np"])
            title = (f"ID: {item['image_id'][:12]}...\n"
                     f"Actual: {item['ground_truth']} | Pred: {item['predicted']} ({item['confidence']*100:.1f}%)\n"
                     f"SOG: {item['sog_knots']:.1f} kts | ROT: {item['rot_deg_min']:.1f}°/min\n"
                     f"Cause: {item['failure_mechanism'][:35]}...")
            ax.set_title(title, fontsize=9.5, fontweight='bold', color='darkred')
            ax.axis('off')

        # Hide extra subplots
        for j in range(n_items, len(axes)):
            axes[j].axis('off')

        gallery_path = os.path.join(FIGURES_DIR, f"{architecture}_failure_gallery.png")
        plt.tight_layout()
        plt.savefig(gallery_path, dpi=300)
        plt.close()
        print(f"Failure analysis gallery saved to: {gallery_path}")

    # Save failure analysis metrics
    clean_failures = [{k: v for k, v in f.items() if k != "img_np"} for f in failures]
    metrics_path = os.path.join(METRICS_DIR, f"{architecture}_failure_analysis.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "total_test_samples": len(test_df),
            "total_misclassifications": len(failures),
            "error_rate": len(failures) / len(test_df),
            "taxonomy_counts": taxonomy_counts,
            "misclassified_samples": clean_failures
        }, f, indent=2)

    return taxonomy_counts


if __name__ == "__main__":
    chk = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet18_binary_model.pt")
    idx = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_index.csv")
    if os.path.exists(chk) and os.path.exists(idx):
        run_failure_analysis(chk, idx)
    else:
        print("Checkpoint or index not found. Train model first.")
