"""
Master Execution Pipeline for Spaceborne SAR Ship Wake Analysis
Orchestrates end-to-end execution:
1. Dataset Verification / Synthesis
2. Ground-Truth Telemetry Derivation (AIS)
3. Model Training (ResNet-18 Binary, ResNet-18 3-Class, ResNet-34 Polarimetric)
4. Dual-Architecture Ensemble Evaluation with 4-fold TTA
5. Monte Carlo Dropout Epistemic Uncertainty Quantification
6. Operational Domain Shift Robustness Stress-Testing
7. Hydrodynamic Explainability via Grad-CAM
8. Systematic Qualitative Failure Taxonomy Analysis
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, THESIS_DIR)

from src.label_generator import generate_master_index
from src.train import train_model
from src.ensemble_eval import evaluate_dual_ensemble
from src.uncertainty import run_uncertainty_quantification
from src.domain_shift import evaluate_domain_shift
from src.explainability import generate_gradcam_visualizations
from src.failure_analysis import run_failure_analysis
from generate_synthetic_data import generate_benchmark_dataset


def run_pipeline(use_synthetic_if_missing=True, epochs=25, batch_size=16):
    print("=" * 70)
    print("      SPACEBORNE SAR SHIP WAKE MANEUVER RECOGNITION PIPELINE")
    print("         Physics-Grounded Contrast, Uncertainty & Explainability")
    print("=" * 70)

    raw_dir = os.path.join(THESIS_DIR, "data", "raw", "UEIKAP_SAR")
    images_dir = os.path.join(raw_dir, "images")
    index_csv = os.path.join(THESIS_DIR, "data", "processed", "dataset_index.csv")

    # Step 1: Dataset Check
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print(f"\n[Step 1/8] Authentic SAR images not found in {images_dir}")
        if use_synthetic_if_missing:
            print("Generating authentic 515-sample synthetic benchmark based on Michell thin-ship hydrodynamics...")
            generate_benchmark_dataset(output_dir=raw_dir)
        else:
            print("Please run 'python src/download_ueikap.py' to download the 1.46 GB official UEIKAP SAR zip from Zenodo.")
            return

    # Step 2: Ground-Truth Derivation from AIS
    print("\n[Step 2/8] Deriving ground-truth kinematics (ROT, drift angle, deadband filtering)...")
    generate_master_index(raw_dir=raw_dir, output_csv=index_csv)

    # Step 3: Train ResNet-18 Binary Model (Straight vs Turning)
    print("\n[Step 3/8] Training ResNet-18 Binary Model with Focal Loss & CLAHE Preprocessing...")
    train_model(
        architecture="resnet18",
        mode="binary",
        epochs=epochs,
        batch_size=batch_size,
        index_csv=index_csv
    )

    # Step 4: Train ResNet-34 Polarimetric Model
    print("\n[Step 4/8] Training ResNet-34 Polarimetric Model (deeper wave curvature extraction)...")
    train_model(
        architecture="resnet34",
        mode="binary",
        epochs=epochs,
        batch_size=batch_size,
        index_csv=index_csv
    )

    # Step 5: Train ResNet-18 3-Class Directional Model (Straight vs Left vs Right)
    print("\n[Step 5/8] Training ResNet-18 3-Class Directional Model...")
    train_model(
        architecture="resnet18",
        mode="3class",
        epochs=epochs,
        batch_size=batch_size,
        index_csv=index_csv
    )

    # Step 6: Dual Ensemble + Test-Time Augmentation
    print("\n[Step 6/8] Evaluating Dual-Architecture Ensemble with 4-fold Test-Time Augmentation (TTA)...")
    chk18 = os.path.join(THESIS_DIR, "models", "best_resnet18_binary_model.pt")
    chk34 = os.path.join(THESIS_DIR, "models", "best_resnet34_binary_model.pt")
    evaluate_dual_ensemble(chk18, chk34, index_csv)

    # Step 7: Uncertainty Quantification & Domain Shift
    print("\n[Step 7/8] Running Monte Carlo Dropout Uncertainty Quantification & Domain Shift Stress-Testing...")
    run_uncertainty_quantification(chk18, index_csv, num_passes=30)
    evaluate_domain_shift(chk18, index_csv, levels=5)

    # Step 8: Explainability (Grad-CAM) & Failure Taxonomy
    print("\n[Step 8/8] Generating Hydrodynamic Grad-CAM Heatmaps & Qualitative Failure Taxonomy...")
    generate_gradcam_visualizations(chk18, index_csv, num_samples=10)
    run_failure_analysis(chk18, index_csv)

    print("\n" + "=" * 70)
    print("      RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Figures:  {os.path.join(THESIS_DIR, 'outputs', 'figures')}")
    print(f"  Metrics:  {os.path.join(THESIS_DIR, 'outputs', 'metrics')}")
    print(f"  Grad-CAM: {os.path.join(THESIS_DIR, 'outputs', 'gradcam')}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
