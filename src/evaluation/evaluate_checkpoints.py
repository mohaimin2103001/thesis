"""
================================================================================
FROZEN TEST BENCHMARK EVALUATION SUITE (Phases 33-42 / Steps 7-8)
Loads all 5 trained model checkpoints from checkpoints/
Evaluates on the 89 frozen unseen test ships in data/processed/splits/test.parquet
Computes Target A, Target B, and Target C metrics.
Exports official tables to tables/ and figures to figures/.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
import torchvision.models as models

# Enforce single-thread execution to prevent Windows OpenBLAS/memory allocation issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

GEOMETRY_COLS = [
    "wake_length", "wake_orientation", "mean_curvature", "median_curvature",
    "max_abs_curvature", "signed_mean_curvature", "curvature_std",
    "wake_asymmetry", "mask_area", "geometry_confidence"
]

# ------------------------------------------------------------------------------
# Model Architectures matching the trained weights
# ------------------------------------------------------------------------------
class GeometryMLP(nn.Module):
    def __init__(self, in_dim=10, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        curr = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(curr, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)])
            curr = h
        self.net = nn.Sequential(*layers)
        self.head_state = nn.Linear(curr, 2)
        self.head_dir = nn.Linear(curr, 2)
        self.head_rot = nn.Linear(curr, 1)

    def forward(self, x):
        feat = self.net(x)
        return self.head_state(feat), self.head_dir(feat), self.head_rot(feat)

class VisualManeuverModel(nn.Module):
    def __init__(self, backbone="resnet34"):
        super().__init__()
        base = getattr(models, backbone)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head_state = nn.Linear(feat_dim, 2)
        self.head_dir = nn.Linear(feat_dim, 2)
        self.head_rot = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_state(feat), self.head_dir(feat), self.head_rot(feat)

class VisualGeometryFusionModel(nn.Module):
    def __init__(self, visual_backbone="resnet34", geom_dim=10, fused_dim=256):
        super().__init__()
        base = getattr(models, visual_backbone)(weights=None)
        vis_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.visual_encoder = base

        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(vis_dim + 64, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.head_state = nn.Linear(fused_dim, 2)
        self.head_dir = nn.Linear(fused_dim, 2)
        self.head_rot = nn.Linear(fused_dim, 1)

    def forward(self, img, geom):
        f_vis = self.visual_encoder(img)
        f_geom = self.geom_encoder(geom)
        f_fused = self.fusion_mlp(torch.cat([f_vis, f_geom], dim=1))
        return self.head_state(f_fused), self.head_dir(f_fused), self.head_rot(f_fused)


# ------------------------------------------------------------------------------
# Helper functions for resolving local paths
# ------------------------------------------------------------------------------
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


def load_test_data(img_size=(384, 384)):
    train_path = os.path.join(SPLITS_DIR, "train.parquet")
    test_path = os.path.join(SPLITS_DIR, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path).reset_index(drop=True)

    # Compute normalizer strictly on train split
    geom_train = train_df[GEOMETRY_COLS].values.astype(np.float32)
    geom_mean = np.nanmean(geom_train, axis=0)
    geom_std = np.nanstd(geom_train, axis=0) + 1e-6

    geom_test = test_df[GEOMETRY_COLS].values.astype(np.float32)
    geom_test_norm = (geom_test - geom_mean) / geom_std
    geom_test_norm = np.nan_to_num(geom_test_norm, nan=0.0)

    # Preload all test images and masks
    images = []
    wake_images = []
    print(f"Loading {len(test_df)} test images and masks...")
    for idx, row in test_df.iterrows():
        img_p = resolve_path(row["original_image_path"])
        img = cv2.imread(img_p)
        if img is None:
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, img_size)

        mask_p = resolve_path(row["wake_mask_path"])
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            wake_img = cv2.bitwise_and(img, img, mask=(mask > 127).astype(np.uint8))
        else:
            wake_img = img.copy()

        images.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        wake_images.append(wake_img.transpose(2, 0, 1).astype(np.float32) / 255.0)

    images_tensor = torch.from_numpy(np.stack(images))
    wake_images_tensor = torch.from_numpy(np.stack(wake_images))
    geom_tensor = torch.from_numpy(geom_test_norm)

    return test_df, images_tensor, wake_images_tensor, geom_tensor


def run_benchmark():
    print("=" * 70)
    print("STARTING FROZEN TEST BENCHMARK EVALUATION (89 UNSEEN VESSELS)")
    print("=" * 70)

    test_df, images, wake_images, geom = load_test_data()

    # Ground truth labels
    y_state_true = (test_df["state"] == "Turning").astype(int).values
    # Direction: 0 = Port, 1 = Starboard
    y_dir_true = np.array([0 if d == "Port" else (1 if d == "Starboard" else -1) for d in test_df["direction"]])
    y_rot_true = test_df["rot_target"].values.astype(np.float32)

    turning_mask = (y_state_true == 1)
    print(f"Test Set Ground Truth: Total={len(test_df)}, Straight={sum(y_state_true==0)}, Turning={sum(y_state_true==1)} (Port={sum(y_dir_true==0)}, Starboard={sum(y_dir_true==1)})")

    model_configs = [
        {
            "name": "Model 1: Raw SAR Image-Only",
            "type": "image",
            "path": os.path.join(CHECKPOINTS_DIR, "image_baseline", "image_baseline_best.pt")
        },
        {
            "name": "Model 2: Segmented Wake-Only",
            "type": "wake",
            "path": os.path.join(CHECKPOINTS_DIR, "wake_only", "wake_only_best.pt")
        },
        {
            "name": "Model 3: Wake Geometry-Only (MLP)",
            "type": "geom",
            "path": os.path.join(CHECKPOINTS_DIR, "geometry_baseline", "geometry_baseline_best.pt")
        },
        {
            "name": "Model 4: Visual-Geometry Fusion",
            "type": "fusion",
            "path": os.path.join(CHECKPOINTS_DIR, "fusion", "fusion_best.pt")
        },
        {
            "name": "Model 5: Fusion + Physics Guidance (Ours)",
            "type": "fusion",
            "path": os.path.join(CHECKPOINTS_DIR, "fusion_physics", "fusion_physics_best.pt")
        }
    ]

    all_metrics = []
    predictions_dict = {
        "sample_id": test_df["sample_id"],
        "mmsi": test_df["mmsi"],
        "state_true": test_df["state"],
        "direction_true": test_df["direction"],
        "rot_true": y_rot_true
    }

    for cfg in model_configs:
        m_name = cfg["name"]
        m_type = cfg["type"]
        m_path = cfg["path"]
        print(f"\nEvaluating {m_name}...")

        if not os.path.exists(m_path):
            print(f"Warning: Checkpoint {m_path} not found. Skipping.")
            continue

        state_dict = torch.load(m_path, map_location="cpu")

        if m_type in ["image", "wake"]:
            model = VisualManeuverModel(backbone="resnet34")
            model.load_state_dict(state_dict)
            model.eval()
            inp = images if m_type == "image" else wake_images
            with torch.no_grad():
                out_state, out_dir, out_rot = model(inp)

        elif m_type == "geom":
            model = GeometryMLP(in_dim=10)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                out_state, out_dir, out_rot = model(geom)

        elif m_type == "fusion":
            model = VisualGeometryFusionModel(visual_backbone="resnet34", geom_dim=10)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                out_state, out_dir, out_rot = model(images, geom)

        # Probabilities & Predictions
        prob_state = torch.softmax(out_state, dim=1)[:, 1].numpy()
        pred_state = (prob_state >= 0.5).astype(int)

        prob_dir = torch.softmax(out_dir, dim=1)[:, 1].numpy()
        pred_dir = (prob_dir >= 0.5).astype(int)

        pred_rot = out_rot.squeeze(1).numpy()

        # Save predictions
        prefix = m_type if m_type != "fusion" else ("fusion_physics" if "Physics" in m_name else "fusion")
        predictions_dict[f"{prefix}_pred_state"] = pred_state
        predictions_dict[f"{prefix}_prob_turning"] = prob_state
        predictions_dict[f"{prefix}_pred_rot"] = pred_rot

        # Metric 1: Target A (Maneuver State: Straight vs Turning)
        acc_a = accuracy_score(y_state_true, pred_state)
        bal_acc_a = balanced_accuracy_score(y_state_true, pred_state)
        f1_a = f1_score(y_state_true, pred_state, average="macro", zero_division=0)
        prec_a = precision_score(y_state_true, pred_state, zero_division=0)
        rec_a = recall_score(y_state_true, pred_state, zero_division=0)
        try:
            auc_a = roc_auc_score(y_state_true, prob_state)
        except Exception:
            auc_a = 0.5

        # Metric 2: Target B (Turn Direction - Port vs Starboard on confirmed turning vessels)
        if sum(turning_mask) > 0:
            dir_true_turn = y_dir_true[turning_mask]
            dir_pred_turn = pred_dir[turning_mask]
            acc_b = accuracy_score(dir_true_turn, dir_pred_turn)
            f1_b = f1_score(dir_true_turn, dir_pred_turn, average="macro", zero_division=0)
        else:
            acc_b, f1_b = 0.0, 0.0

        # Metric 3: Target C (Continuous Signed Rate of Turn)
        mae_c = mean_absolute_error(y_rot_true, pred_rot)
        rmse_c = np.sqrt(mean_squared_error(y_rot_true, pred_rot))
        r2_c = r2_score(y_rot_true, pred_rot)
        try:
            corr_c, _ = pearsonr(y_rot_true, pred_rot)
            if np.isnan(corr_c):
                corr_c = 0.0
        except Exception:
            corr_c = 0.0

        # Sign consistency on turning vessels
        rot_true_turn = y_rot_true[turning_mask]
        rot_pred_turn = pred_rot[turning_mask]
        sign_match = np.sign(rot_true_turn) == np.sign(rot_pred_turn)
        sign_consistency = np.mean(sign_match) if len(sign_match) > 0 else 0.0

        all_metrics.append({
            "Model": m_name,
            "State_Acc": round(acc_a, 4),
            "State_BalAcc": round(bal_acc_a, 4),
            "State_MacroF1": round(f1_a, 4),
            "State_Precision": round(prec_a, 4),
            "State_Recall": round(rec_a, 4),
            "State_ROC_AUC": round(auc_a, 4),
            "Dir_Acc": round(acc_b, 4),
            "Dir_MacroF1": round(f1_b, 4),
            "ROT_MAE": round(mae_c, 3),
            "ROT_RMSE": round(rmse_c, 3),
            "ROT_Pearson_r": round(corr_c, 3),
            "ROT_R2": round(r2_c, 3),
            "ROT_SignConsistency": round(sign_consistency, 3)
        })

    # Export unified test predictions
    pred_df = pd.DataFrame(predictions_dict)
    pred_parquet_path = os.path.join(PROCESSED_DIR, "test_predictions.parquet")
    pred_df.to_parquet(pred_parquet_path, index=False)
    print(f"\nSaved unified test predictions to: {pred_parquet_path}")

    # Export main ablation results table
    results_df = pd.DataFrame(all_metrics)
    main_table_path = os.path.join(TABLES_DIR, "main_ablation_results.csv")
    results_df.to_csv(main_table_path, index=False)
    print(f"Saved Main Ablation Results Table to: {main_table_path}")
    print("\n" + "=" * 90)
    print(results_df[["Model", "State_MacroF1", "State_BalAcc", "Dir_MacroF1", "ROT_MAE", "ROT_R2", "ROT_Pearson_r"]].to_string())
    print("=" * 90)

    # Export specialized Target tables for publication
    target_a_df = results_df[["Model", "State_Acc", "State_BalAcc", "State_Precision", "State_Recall", "State_MacroF1", "State_ROC_AUC"]]
    target_a_df.to_csv(os.path.join(TABLES_DIR, "state_metrics.csv"), index=False)

    target_b_df = results_df[["Model", "Dir_Acc", "Dir_MacroF1"]]
    target_b_df.to_csv(os.path.join(TABLES_DIR, "direction_metrics.csv"), index=False)

    target_c_df = results_df[["Model", "ROT_MAE", "ROT_RMSE", "ROT_Pearson_r", "ROT_R2", "ROT_SignConsistency"]]
    target_c_df.to_csv(os.path.join(TABLES_DIR, "rot_metrics.csv"), index=False)

    # Generate Publication-Grade Comparison Figure (Fig 15)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(results_df))
    width = 0.25

    # Subplot 1: Classification Performance
    rects1 = ax1.bar(x - width, results_df["State_MacroF1"], width, label="Maneuver State F1 (Target A)", color="#1f77b4")
    rects2 = ax1.bar(x, results_df["State_BalAcc"], width, label="Balanced Accuracy (Target A)", color="#4ca0d0")
    rects3 = ax1.bar(x + width, results_df["Dir_MacroF1"], width, label="Direction F1 (Target B)", color="#2ca02c")
    ax1.set_title("(a) Maneuver State & Turn Direction Classification", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split(":")[0] for m in results_df["Model"]], rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("Metric Score [0.0 - 1.0]", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Continuous ROT Regression Performance
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x - 0.18, results_df["ROT_MAE"], 0.35, label="ROT Mean Absolute Error (MAE)", color="#d62728")
    l1 = ax2_twin.plot(x, results_df["ROT_Pearson_r"], "o-", label="Pearson Correlation (r)", color="#9467bd", linewidth=2.5, markersize=8)

    ax2.set_title("(b) Continuous Signed Rate-of-Turn Regression Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split(":")[0] for m in results_df["Model"]], rotation=20, ha="right", fontsize=10)
    ax2.set_ylabel("ROT MAE [deg/min] (Lower is Better)", fontsize=11, color="#d62728")
    ax2_twin.set_ylabel("Pearson Correlation (Higher is Better)", fontsize=11, color="#9467bd")
    ax2.set_ylim(0, max(results_df["ROT_MAE"]) * 1.3)
    ax2_twin.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True)

    plt.tight_layout()
    fig15_path = os.path.join(FIGURES_DIR, "fig15_main_ablation_comparison.png")
    plt.savefig(fig15_path, dpi=300)
    plt.close()
    print(f"Saved Figure 15 (Main Ablation Comparison) to: {fig15_path}")

    # Generate ROT Scatter Comparison Plot (Fig 16)
    plt.figure(figsize=(7, 6))
    if "fusion_physics_pred_rot" in pred_df.columns:
        pred_rot_ours = pred_df["fusion_physics_pred_rot"].values
        plt.scatter(y_rot_true, pred_rot_ours, alpha=0.75, c="#2ca02c", edgecolors="k", s=50, label="Proposed Fusion+Physics")
    elif "fusion_pred_rot" in pred_df.columns:
        pred_rot_ours = pred_df["fusion_pred_rot"].values
        plt.scatter(y_rot_true, pred_rot_ours, alpha=0.75, c="#1f77b4", edgecolors="k", s=50, label="Fusion Model")

    # Reference ideal line
    min_val = min(y_rot_true.min(), -10.0)
    max_val = max(y_rot_true.max(), 10.0)
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="Perfect Agreement (y = x)")
    plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
    plt.axvline(0, color="gray", linestyle=":", alpha=0.7)

    plt.title("Rate of Turn Estimation: True AIS vs Physics-Guided SAR Prediction", fontsize=12, fontweight="bold")
    plt.xlabel("True Synchronized AIS ROT [deg/min]", fontsize=11)
    plt.ylabel("SAR Wake-Inferred ROT [deg/min]", fontsize=11)
    plt.xlim(min_val - 1, max_val + 1)
    plt.ylim(min_val - 1, max_val + 1)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig16_path = os.path.join(FIGURES_DIR, "fig16_rot_scatter_comparison.png")
    plt.savefig(fig16_path, dpi=300)
    plt.close()
    print(f"Saved Figure 16 (ROT Scatter Plot) to: {fig16_path}")

    print("\nBENCHMARK EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_benchmark()
