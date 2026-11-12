"""
Calibrates operational decision thresholds on the validation set
and evaluates on the unseen frozen test set.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import torchvision.models as models

from src.evaluation.evaluate_checkpoints import (
    VisualGeometryFusionModel, VisualManeuverModel, GeometryMLP,
    resolve_path, GEOMETRY_COLS, SPLITS_DIR, CHECKPOINTS_DIR, TABLES_DIR
)


def run_threshold_calibration():
    train_df = pd.read_parquet(os.path.join(SPLITS_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(SPLITS_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(SPLITS_DIR, "test.parquet"))

    # Fit normalizer on train split only
    geom_train = train_df[GEOMETRY_COLS].values.astype(np.float32)
    geom_mean = np.nanmean(geom_train, axis=0)
    geom_std = np.nanstd(geom_train, axis=0) + 1e-6

    def prep_data(df):
        geom = (df[GEOMETRY_COLS].values.astype(np.float32) - geom_mean) / geom_std
        geom = np.nan_to_num(geom, nan=0.0)
        imgs = []
        wake_imgs = []
        for _, r in df.iterrows():
            im = cv2.imread(resolve_path(r["original_image_path"]))
            im = cv2.resize(im, (384, 384)) if im is not None else np.zeros((384, 384, 3), dtype=np.uint8)
            mask_p = resolve_path(r["wake_mask_path"])
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
                w_im = cv2.bitwise_and(im, im, mask=(mask > 127).astype(np.uint8))
            else:
                w_im = im.copy()
            imgs.append(im.transpose(2, 0, 1).astype(np.float32) / 255.0)
            wake_imgs.append(w_im.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return torch.tensor(np.stack(imgs)), torch.tensor(np.stack(wake_imgs)), torch.tensor(geom), (df["state"] == "Turning").astype(int).values

    print("Preparing validation and test data...")
    val_imgs, val_wakes, val_geom, y_val = prep_data(val_df)
    test_imgs, test_wakes, test_geom, y_test = prep_data(test_df)

    models_info = [
        ("Model 1: Raw SAR Image-Only", "image", os.path.join(CHECKPOINTS_DIR, "image_baseline", "image_baseline_best.pt")),
        ("Model 2: Segmented Wake-Only", "wake", os.path.join(CHECKPOINTS_DIR, "wake_only", "wake_only_best.pt")),
        ("Model 3: Wake Geometry-Only (MLP)", "geom", os.path.join(CHECKPOINTS_DIR, "geometry_baseline", "geometry_baseline_best.pt")),
        ("Model 4: Visual-Geometry Fusion", "fusion", os.path.join(CHECKPOINTS_DIR, "fusion", "fusion_best.pt")),
        ("Model 5: Fusion + Physics Guidance (Ours)", "fusion", os.path.join(CHECKPOINTS_DIR, "fusion_physics", "fusion_physics_best.pt"))
    ]

    calibrated_results = []
    print("\n" + "=" * 80)
    print("VALIDATION THRESHOLD TUNING & FROZEN TEST EVALUATION")
    print("=" * 80)

    for name, m_type, path in models_info:
        state_dict = torch.load(path, map_location="cpu")
        if m_type in ["image", "wake"]:
            m = VisualManeuverModel(backbone="resnet34")
            m.load_state_dict(state_dict)
            m.eval()
            inp_val = val_imgs if m_type == "image" else val_wakes
            inp_test = test_imgs if m_type == "image" else test_wakes
            with torch.no_grad():
                val_probs = torch.softmax(m(inp_val)[0], dim=1)[:, 1].numpy()
                test_probs = torch.softmax(m(inp_test)[0], dim=1)[:, 1].numpy()
        elif m_type == "geom":
            m = GeometryMLP(in_dim=10)
            m.load_state_dict(state_dict)
            m.eval()
            with torch.no_grad():
                val_probs = torch.softmax(m(val_geom)[0], dim=1)[:, 1].numpy()
                test_probs = torch.softmax(m(test_geom)[0], dim=1)[:, 1].numpy()
        elif m_type == "fusion":
            m = VisualGeometryFusionModel(visual_backbone="resnet34", geom_dim=10)
            m.load_state_dict(state_dict)
            m.eval()
            with torch.no_grad():
                val_probs = torch.softmax(m(val_imgs, val_geom)[0], dim=1)[:, 1].numpy()
                test_probs = torch.softmax(m(test_imgs, test_geom)[0], dim=1)[:, 1].numpy()

        # Find best threshold on validation set (maximizing Balanced Accuracy)
        taus = np.linspace(0.05, 0.60, 56)
        best_tau = 0.5
        best_val_score = -1.0
        for tau in taus:
            preds_val = (val_probs >= tau).astype(int)
            score = balanced_accuracy_score(y_val, preds_val)
            if score > best_val_score:
                best_val_score = score
                best_tau = tau

        # Apply best_tau to test set
        preds_test = (test_probs >= best_tau).astype(int)
        acc = accuracy_score(y_test, preds_test)
        bal_acc = balanced_accuracy_score(y_test, preds_test)
        f1 = f1_score(y_test, preds_test, average="macro")
        prec = precision_score(y_test, preds_test, zero_division=0)
        rec = recall_score(y_test, preds_test, zero_division=0)
        auc = roc_auc_score(y_test, test_probs)

        calibrated_results.append({
            "Model": name,
            "Optimal_Tau": round(best_tau, 3),
            "Val_BalAcc": round(best_val_score, 4),
            "Test_Acc": round(acc, 4),
            "Test_BalAcc": round(bal_acc, 4),
            "Test_MacroF1": round(f1, 4),
            "Test_Precision": round(prec, 4),
            "Test_Recall": round(rec, 4),
            "Test_ROC_AUC": round(auc, 4)
        })

    calib_df = pd.DataFrame(calibrated_results)
    out_csv = os.path.join(TABLES_DIR, "calibrated_state_metrics.csv")
    calib_df.to_csv(out_csv, index=False)
    print(calib_df.to_string())
    print(f"\nSaved calibrated metrics to: {out_csv}")


if __name__ == "__main__":
    run_threshold_calibration()
