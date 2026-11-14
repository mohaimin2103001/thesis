"""
Generates and inspects Grad-CAM heatmaps for Model 5 (Fusion + Physics Guidance)
and quantifies the overlap with the authentic wake masks.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.evaluation.evaluate_checkpoints import (
    VisualGeometryFusionModel, resolve_path, GEOMETRY_COLS,
    SPLITS_DIR, CHECKPOINTS_DIR, FIGURES_DIR
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "gradcam_model5")
os.makedirs(OUT_DIR, exist_ok=True)


class GradCAMFusion:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor, geom_tensor, target_class=1):
        self.model.eval()
        self.model.zero_grad()

        out_state, out_dir, out_rot = self.model(img_tensor, geom_tensor)
        score = out_state[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        cam_resized = cv2.resize(cam_np, (img_tensor.shape[3], img_tensor.shape[2]))
        probs = torch.softmax(out_state, dim=1).detach().cpu().numpy()[0]
        pred_rot = out_rot.item()
        return cam_resized, probs, pred_rot


def inspect_model5_attention():
    train_df = pd.read_parquet(os.path.join(SPLITS_DIR, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(SPLITS_DIR, "test.parquet"))

    geom_train = train_df[GEOMETRY_COLS].values.astype(np.float32)
    geom_mean = np.nanmean(geom_train, axis=0)
    geom_std = np.nanstd(geom_train, axis=0) + 1e-6

    # Load Model 5
    m_path = os.path.join(CHECKPOINTS_DIR, "fusion_physics", "fusion_physics_best.pt")
    model = VisualGeometryFusionModel(visual_backbone="resnet34", geom_dim=10)
    model.load_state_dict(torch.load(m_path, map_location="cpu"))
    model.eval()

    grad_cam = GradCAMFusion(model, model.visual_encoder.layer4[-1].conv2)

    # Let's test on turning samples and straight samples
    turning_df = test_df[test_df["state"] == "Turning"].reset_index(drop=True)
    straight_df = test_df[test_df["state"] == "Straight"].reset_index(drop=True)

    test_samples = list(turning_df.iterrows()) + list(straight_df.head(5).iterrows())
    print(f"Generating Grad-CAM for {len(test_samples)} test ships on Model 5...")

    results = []
    for idx, row in test_samples:
        s_id = row["sample_id"]
        img_p = resolve_path(row["original_image_path"])
        mask_p = resolve_path(row["wake_mask_path"])
        gt_state = row["state"]
        gt_rot = row["rot_target"]

        img = cv2.imread(img_p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(cv2.resize(img, (384, 384)), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)

        geom_val = (row[GEOMETRY_COLS].values.astype(np.float32) - geom_mean) / geom_std
        geom_val = np.nan_to_num(geom_val, nan=0.0)
        geom_tensor = torch.from_numpy(geom_val).unsqueeze(0)

        # Target class: 1 for Turning, 0 for Straight
        target_cls = 1 if gt_state == "Turning" else 0
        cam, probs, pred_rot = grad_cam.generate(img_tensor, geom_tensor, target_class=target_cls)

        # Load mask and compute wake overlap
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_bin = (cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        else:
            mask_bin = np.zeros((384, 384), dtype=np.float32)

        # High attention zone (top 30% intensity of CAM)
        high_cam = (cam > 0.6).astype(np.float32)

        # Compute overlap with wake mask
        intersection = np.sum(high_cam * mask_bin)
        cam_area = np.sum(high_cam) + 1e-6
        mask_area = np.sum(mask_bin) + 1e-6

        wake_capture_ratio = intersection / mask_area # How much of wake is highlighted
        cam_precision = intersection / cam_area # How much of CAM is actually on the wake

        # Where is peak attention? (centroid of CAM vs centroid of wake)
        cam_y, cam_x = np.unravel_index(np.argmax(cam), cam.shape)
        if np.sum(mask_bin) > 0:
            mask_y, mask_x = np.mean(np.where(mask_bin > 0)[0]), np.mean(np.where(mask_bin > 0)[1])
            dist = np.sqrt((cam_x - mask_x)**2 + (cam_y - mask_y)**2)
        else:
            dist = -1.0

        results.append({
            "sample_id": s_id[:30],
            "gt_state": gt_state,
            "gt_rot": round(gt_rot, 2),
            "prob_turning": round(float(probs[1]), 3),
            "pred_rot": round(pred_rot, 2),
            "wake_capture": round(float(wake_capture_ratio), 3),
            "cam_precision": round(float(cam_precision), 3),
            "peak_dist_to_wake": round(float(dist), 1),
            "cam_peak_coord": (int(cam_x), int(cam_y))
        })

        # Save visualization overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * img_rgb + 0.4 * heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Sentinel-1 SAR\nGT: {gt_state} (ROT={gt_rot:.1f})", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Activation Map", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].contour(mask_bin, colors="white", linewidths=1.5, levels=[0.5])
        axes[2].set_title(f"Model 5 Overlay (White Line = True Wake)\nPred P(Turn)={probs[1]:.2f}, Pred ROT={pred_rot:.1f}", fontsize=10, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        save_name = f"model5_gradcam_{idx:02d}_{gt_state}_{s_id[:20].replace(' ', '_')}.png"
        plt.savefig(os.path.join(OUT_DIR, save_name), dpi=150)
        plt.close()

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("GRAD-CAM ATTENTION VS TRUE WAKE LOCATION ANALYSIS (MODEL 5):")
    print("=" * 90)
    print(res_df.to_string())
    res_df.to_csv(os.path.join(PROJECT_ROOT, "tables", "model5_gradcam_analysis.csv"), index=False)
    print(f"\nSaved results to tables/model5_gradcam_analysis.csv and visualizations to {OUT_DIR}")


if __name__ == "__main__":
    inspect_model5_attention()

