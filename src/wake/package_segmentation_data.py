"""
Wake Segmentation Dataset Packager (Phase 5)
Packages UEIKAP authentic SAR chips and rasterized binary masks with product-level leakage-safe splitting.
Outputs Kaggle-ready directory structure and dataset card.
"""

import os
import re
import shutil
import pandas as pd
import numpy as np

PROCESSED_DIR = r"e:\thesis\data\processed"
KAGGLE_PKG_DIR = os.path.join(PROCESSED_DIR, "wake_segmentation_kaggle")


def extract_acquisition_group(filename):
    """
    Extracts acquisition timestamp (e.g., 20240401T165351) to group chips by radar pass.
    Ensures zero leakage across train and validation splits.
    """
    match = re.search(r'S1[AB]_(\d{8}T\d{6})', filename)
    if match:
        return match.group(1)
    return "UNKNOWN_PASS"


def package_wake_segmentation_dataset(val_ratio=0.20, seed=42):
    np.random.seed(seed)
    
    mask_index_path = os.path.join(PROCESSED_DIR, "ueikap_mask_index.parquet")
    if not os.path.exists(mask_index_path):
        raise FileNotFoundError(f"Missing {mask_index_path}. Run ueikap_parser.py first.")

    df = pd.read_parquet(mask_index_path)
    print(f"Loaded {len(df)} validated UEIKAP mask records.")

    # Assign product acquisition group
    df["acquisition_group"] = df["image_path"].apply(extract_acquisition_group)
    groups = df["acquisition_group"].unique()
    print(f"Discovered {len(groups)} distinct Sentinel-1 acquisition passes: {groups[:5]}...")

    # Grouped split: assign whole acquisition passes to train or val
    np.random.shuffle(groups)
    val_groups_count = max(1, int(len(groups) * val_ratio))
    val_groups = set(groups[:val_groups_count])
    train_groups = set(groups[val_groups_count:])

    df["split"] = df["acquisition_group"].apply(lambda g: "val" if g in val_groups else "train")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    print(f"Grouped Split: {len(train_df)} Train samples ({len(train_groups)} passes) | "
          f"{len(val_df)} Val samples ({len(val_groups)} passes)")

    # Prepare package directories
    pkg_images_dir = os.path.join(KAGGLE_PKG_DIR, "images")
    pkg_masks_dir = os.path.join(KAGGLE_PKG_DIR, "masks")
    os.makedirs(pkg_images_dir, exist_ok=True)
    os.makedirs(pkg_masks_dir, exist_ok=True)

    train_records = []
    val_records = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        src_img = row["image_path"]
        src_mask = row["mask_path"]

        dst_img_name = f"{sample_id}.png"
        dst_mask_name = f"{sample_id}_mask.png"

        dst_img_path = os.path.join(pkg_images_dir, dst_img_name)
        dst_mask_path = os.path.join(pkg_masks_dir, dst_mask_name)

        shutil.copy2(src_img, dst_img_path)
        shutil.copy2(src_mask, dst_mask_path)

        rec = {
            "sample_id": sample_id,
            "image_filename": dst_img_name,
            "mask_filename": dst_mask_name,
            "acquisition_group": row["acquisition_group"],
            "mask_area": row["mask_area"],
            "mask_area_ratio": row["mask_area_ratio"]
        }

        if row["split"] == "train":
            train_records.append(rec)
        else:
            val_records.append(rec)

    train_pkg_df = pd.DataFrame(train_records)
    val_pkg_df = pd.DataFrame(val_records)

    train_pkg_df.to_parquet(os.path.join(KAGGLE_PKG_DIR, "train.parquet"), index=False)
    val_pkg_df.to_parquet(os.path.join(KAGGLE_PKG_DIR, "validation.parquet"), index=False)

    # Generate Dataset Card
    card_content = f"""# UEIKAP Authentic Spaceborne SAR Wake Segmentation Benchmark
## Dataset Card & Provenance Documentation

- **Dataset Name:** UEIKAP Sentinel-1 SAR Wake Segmentation Package
- **Source Benchmark:** Universita degli studi di Napoli Federico II / ESA Sentinel-1
- **Permanent DOI:** [10.5281/zenodo.18391027](https://doi.org/10.5281/zenodo.18391027)
- **Sensor:** Sentinel-1 C-Band SAR (5.405 GHz), VV/VH Polarizations
- **Image Size:** 512 x 512 pixels
- **Total Validated Samples:** {len(df)}
- **Training Samples:** {len(train_pkg_df)} ({len(train_groups)} unique acquisition passes)
- **Validation Samples:** {len(val_pkg_df)} ({len(val_groups)} unique acquisition passes)
- **Leakage Prevention:** Grouped strictly by Sentinel-1 satellite pass timestamp (`acquisition_group`). No chip from a validation orbit pass appears in training.
- **Labels:** Binary Wake Masks (0 = Background ocean, 255 = Authentic hydrodynamic ship wake).
"""
    with open(os.path.join(KAGGLE_PKG_DIR, "dataset_card.md"), "w") as f:
        f.write(card_content)

    print(f"\nWake segmentation package created successfully in: {KAGGLE_PKG_DIR}")
    return train_pkg_df, val_pkg_df


if __name__ == "__main__":
    package_wake_segmentation_dataset()
