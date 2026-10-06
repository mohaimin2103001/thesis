"""
Hydrodynamic Explainability via Gradient-Weighted Class Activation Mapping (Grad-CAM)
Empirically verifies that convolutional feature maps activate along the trailing Kelvin wake cusp
and asymmetric rudder turbulence rather than memorizing random ocean clutter or vessel hull artifacts.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .models import SARWakeBackbone
from .dataset import preprocess_sar_chip

GRADCAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "gradcam"))


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Channel-wise global average pooling of gradients
        # alpha_k = (1/Z) * sum(grad)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, torch.softmax(output, dim=1).detach().cpu().numpy()[0]


def generate_gradcam_visualizations(
    model_checkpoint_path,
    csv_index_path,
    num_samples=10,
    architecture="resnet18",
    num_classes=2,
    img_size=384
):
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SARWakeBackbone(architecture=architecture, num_classes=num_classes, dropout_p=0.0).to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Find target layer in ResNet
    if architecture in ["resnet18", "resnet34"]:
        target_layer = model.backbone.layer4[-1].conv2
    elif architecture == "convnext_tiny":
        target_layer = model.backbone.stages[-1].blocks[-1]
    else:
        target_layer = list(model.modules())[-4]

    grad_cam = GradCAM(model, target_layer)

    df = pd.read_csv(csv_index_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Balance samples across classes
    samples_to_show = []
    straight_samples = test_df[test_df["maneuver_binary"] == 0].head(num_samples // 2)
    turning_samples = test_df[test_df["maneuver_binary"] == 1].head(num_samples // 2)
    samples_to_show.extend(straight_samples.to_dict("records"))
    samples_to_show.extend(turning_samples.to_dict("records"))

    class_names = ["Straight", "Turning"] if num_classes == 2 else ["Straight", "Left Turn", "Right Turn"]

    print(f"Generating Grad-CAM hydrodynamic heatmaps for {len(samples_to_show)} test samples...")
    for idx, row in enumerate(samples_to_show):
        img_path = row["image_path"]
        ground_truth = int(row["maneuver_binary" if num_classes == 2 else "maneuver_3class"])
        img_id = row["image_id"]

        # Preprocess
        enhanced_pil = preprocess_sar_chip(img_path, mode="grayscale_clahe")
        resized_pil = enhanced_pil.resize((img_size, img_size))
        rgb_np = np.array(resized_pil)

        # Tensor
        tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        # Normalization
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - norm_mean) / norm_std
        tensor = tensor.unsqueeze(0).to(device)

        cam, pred_class, probs = grad_cam.generate(tensor, target_class=ground_truth)

        # Overlay heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * rgb_np + 0.4 * heatmap)

        # Plot 3-panel comparison: Enhanced SAR, Grad-CAM Heatmap, Hydrodynamic Overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_np)
        axes[0].set_title(f"Enhanced Sentinel-1 SAR Chip\nGT: {class_names[ground_truth]}", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title("Grad-CAM Activation Map\n(Conv Feature Attention)", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        is_correct = (pred_class == ground_truth)
        status_text = "CORRECT" if is_correct else "MISCLASSIFIED"
        color = "green" if is_correct else "red"
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Hydrodynamic Verification [{status_text}]\nPred: {class_names[pred_class]} ({probs[pred_class]*100:.1f}%)",
            fontsize=11, fontweight='bold', color=color
        )
        axes[2].axis('off')

        save_file = os.path.join(GRADCAM_DIR, f"gradcam_{idx:02d}_{class_names[ground_truth]}_{img_id[:16]}.png")
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        plt.close()

    print(f"Grad-CAM visual attribution maps successfully saved to: {GRADCAM_DIR}")


if __name__ == "__main__":
    chk = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet18_binary_model.pt")
    idx = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "dataset_index.csv")
    if os.path.exists(chk) and os.path.exists(idx):
        generate_gradcam_visualizations(chk, idx)
    else:
        print("Checkpoint or index not found. Train model first.")
