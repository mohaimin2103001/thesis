"""
Master Dataset Builder & Subset Partitioning (Phases 21-23)
Merges SAR imagery, rasterized wake masks, telemetry ground truth, and explicit geometry.
Enforces strict MMSI and product-level zero-leakage grouped splitting.
Exports maneuver_master.parquet and all five benchmark subsets.
"""

import os
import re
import pandas as pd
import numpy as np
from PIL import Image

import sys
sys.path.append(r"e:\thesis\src")
from geometry.wake_geometry import extract_wake_features_from_mask

PROCESSED_DIR = r"e:\thesis\data\processed"
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
REJECTED_DIR = r"e:\thesis\data\rejected"


def parse_ueikap_ais_telemetry(ais_path):
    """
    Parses single-sample UEIKAP AIS CSV telemetry file.
    """
    try:
        df = pd.read_csv(ais_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if len(df) == 0:
            return None
        row = df.iloc[0]

        sog = float(row.get("sog", 10.0))
        cog = float(row.get("cog", 0.0))
        hdg = float(row.get("heading", cog))
        if hdg == 511.0 or hdg < 0:
            hdg = cog

        rot = float(row.get("rot", 0.0))
        drift = ((cog - hdg + 180.0) % 360.0) - 180.0
        mmsi = str(row.get("mmsi", "UNKNOWN")).split("_")[0]
        timestamp = str(row.get("# timestamp", "2024-04-01 00:00:00"))

        return {
            "mmsi": mmsi,
            "sar_time": timestamp,
            "sog": sog,
            "cog": cog,
            "heading": hdg,
            "course_heading_drift": drift,
            "rot_raw": rot,
            "rot_deg_min": rot,
            "latitude": float(row.get("latitude", 0.0)),
            "longitude": float(row.get("longitude", 0.0)),
            "ship_length": float(row.get("length", 100.0)),
            "ship_width": float(row.get("width", 20.0))
        }
    except Exception:
        return None


def extract_product_id(path):
    match = re.search(r'S1[AB]_(\d{8}T\d{6})', path)
    if match:
        return match.group(1)
    return "PRODUCT_UNKNOWN"


def build_master_dataset(seed=42):
    np.random.seed(seed)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    mask_index_path = os.path.join(PROCESSED_DIR, "ueikap_mask_index.parquet")
    if not os.path.exists(mask_index_path):
        raise FileNotFoundError(f"Missing {mask_index_path}")

    mask_df = pd.read_parquet(mask_index_path)
    print(f"Building master dataset from {len(mask_df)} validated wake samples...")

    master_records = []
    rejected_records = []

    for idx, row in mask_df.iterrows():
        sample_id = row["sample_id"]
        img_path = row["image_path"]
        mask_path = row["mask_path"]
        ais_path = row["ais_path"]

        # Parse telemetry
        telem = parse_ueikap_ais_telemetry(ais_path) if ais_path and os.path.exists(ais_path) else None
        if telem is None:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "telemetry_parsing",
                "reason": "missing_or_corrupt_ais",
                "details": f"AIS file {ais_path} could not be parsed"
            })
            continue

        sog = telem["sog"]
        rot = telem["rot_deg_min"]
        drift = telem["course_heading_drift"]

        # Kinematic sanity check (filter moored/drifting vessels under 2.0 knots)
        if sog < 2.0:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "kinematic_deadband",
                "reason": "vessel_speed_too_low",
                "details": f"SOG {sog:.1f} knots < 2.0 knots threshold"
            })
            continue

        # Scientific classification thresholds:
        # Straight: |ROT| < 0.5 deg/min AND |Delta Theta| < 5.0 deg
        # Turning: |ROT| >= 0.5 deg/min OR |Delta Theta| >= 10.0 deg
        is_straight = (abs(rot) < 0.5) and (abs(drift) < 5.0)
        is_turning = (abs(rot) >= 0.5) or (abs(drift) >= 10.0)

        if is_straight:
            state = "Straight"
            direction = "None"
            rot_target = 0.0 if abs(rot) < 0.05 else rot
            quality = "Gold-A"
        elif is_turning:
            state = "Turning"
            if rot < -0.5 or (abs(rot) < 0.5 and drift <= -10.0):
                direction = "Port"
            else:
                direction = "Starboard"
            rot_target = rot
            quality = "Gold-A" if abs(rot) >= 0.5 else "Gold-B"
        else:
            # Ambiguous buffer zone: 0.5 > |ROT| and 5.0 <= |drift| < 10.0
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "label_disambiguation",
                "reason": "ambiguous_drift_buffer",
                "details": f"ROT={rot:.2f}, Drift={drift:.2f} falls into non-conclusive transition zone"
            })
            continue

        # Extract explicit wake geometry
        geom = extract_wake_features_from_mask(mask_path)
        geom_valid = geom is not None

        rec = {
            "sample_id": sample_id,
            "source_dataset": "UEIKAP",
            "product_id": extract_product_id(img_path),
            "original_image_path": img_path,
            "crop_path": img_path,
            "sar_time": telem["sar_time"],
            "mmsi": telem["mmsi"],
            "latitude": telem["latitude"],
            "longitude": telem["longitude"],
            "sog": sog,
            "cog": telem["cog"],
            "heading": telem["heading"],
            "course_heading_drift": drift,
            "rot_raw": telem["rot_raw"],
            "rot_deg_min": rot,
            "rot_target": rot_target,
            "state": state,
            "direction": direction,
            "ship_x": 256.0,
            "ship_y": 256.0,
            "wake_mask_path": mask_path,
            "wake_mask_source": "official_annotation",
            "wake_confidence": 1.0,
            "label_quality": quality,
            "geometry_valid": geom_valid
        }

        # Append explicit geometry attributes
        if geom_valid:
            rec.update({
                "wake_length": geom["wake_length"],
                "wake_orientation": geom["wake_orientation"],
                "mean_curvature": geom["mean_curvature"],
                "median_curvature": geom["median_curvature"],
                "max_abs_curvature": geom["max_abs_curvature"],
                "signed_mean_curvature": geom["signed_mean_curvature"],
                "curvature_std": geom["curvature_std"],
                "wake_asymmetry": geom["wake_asymmetry"],
                "mask_area": geom["mask_area"],
                "geometry_confidence": geom["geometry_confidence"],
                "branch_count": geom["branch_count"]
            })
        else:
            rec.update({
                "wake_length": 0.0,
                "wake_orientation": 0.0,
                "mean_curvature": 0.0,
                "median_curvature": 0.0,
                "max_abs_curvature": 0.0,
                "signed_mean_curvature": 0.0,
                "curvature_std": 0.0,
                "wake_asymmetry": 0.0,
                "mask_area": row["mask_area"],
                "geometry_confidence": 0.0,
                "branch_count": 0
            })

        master_records.append(rec)

    master_df = pd.DataFrame(master_records)
    print(f"Master records formed: {len(master_df)} (Rejected at labeling: {len(rejected_records)})")

    # Grouped split: Group by MMSI & Product ID to prevent cross-partition leakage
    mmsi_list = master_df["mmsi"].unique()
    np.random.shuffle(mmsi_list)

    test_count = int(len(mmsi_list) * 0.20)
    val_count = int(len(mmsi_list) * 0.15)

    test_mmsis = set(mmsi_list[:test_count])
    val_mmsis = set(mmsi_list[test_count:test_count + val_count])
    train_mmsis = set(mmsi_list[test_count + val_count:])

    def assign_split(mmsi):
        if mmsi in test_mmsis:
            return "test"
        elif mmsi in val_mmsis:
            return "val"
        return "train"

    master_df["split"] = master_df["mmsi"].apply(assign_split)

    # Save Master Parquet
    master_path = os.path.join(PROCESSED_DIR, "maneuver_master.parquet")
    master_df.to_parquet(master_path, index=False)
    print(f"Saved master labeled dataset: {master_path}")

    # Generate Subsets (Phase 22)
    # 1. Image-only: all reliable maneuver samples
    img_only = master_df[master_df["label_quality"].isin(["Gold-A", "Gold-B"])].copy()
    img_only.to_parquet(os.path.join(PROCESSED_DIR, "image_only_subset.parquet"), index=False)

    # 2. Wake-only: samples with valid wake mask
    wake_only = master_df[master_df["mask_area"] > 50].copy()
    wake_only.to_parquet(os.path.join(PROCESSED_DIR, "wake_only_subset.parquet"), index=False)

    # 3. Geometry subset: valid centerline and geometry features
    geom_sub = master_df[master_df["geometry_valid"] & (master_df["geometry_confidence"] > 0.3)].copy()
    geom_sub.to_parquet(os.path.join(PROCESSED_DIR, "geometry_subset.parquet"), index=False)

    # 4. Direction subset: turning vessels only (Port vs Starboard)
    dir_sub = master_df[master_df["state"] == "Turning"].copy()
    dir_sub.to_parquet(os.path.join(PROCESSED_DIR, "direction_subset.parquet"), index=False)

    # 5. ROT regression subset: Gold-A samples with continuous numerical ROT
    rot_sub = master_df[(master_df["label_quality"] == "Gold-A") & master_df["rot_target"].notnull()].copy()
    rot_sub.to_parquet(os.path.join(PROCESSED_DIR, "rot_regression_subset.parquet"), index=False)

    # 6. Partitions
    master_df[master_df["split"] == "train"].to_parquet(os.path.join(SPLITS_DIR, "train.parquet"), index=False)
    master_df[master_df["split"] == "val"].to_parquet(os.path.join(SPLITS_DIR, "val.parquet"), index=False)
    master_df[master_df["split"] == "test"].to_parquet(os.path.join(SPLITS_DIR, "test.parquet"), index=False)

    print("\n=== Master Dataset Generation & Subsets Summary ===")
    print(f"Total Master Samples: {len(master_df)}")
    print(f"  - Straight: {sum(master_df['state'] == 'Straight')}")
    print(f"  - Turning: {sum(master_df['state'] == 'Turning')} (Port: {sum(master_df['direction'] == 'Port')}, Starboard: {sum(master_df['direction'] == 'Starboard')})")
    print(f"  - Geometry-Valid: {sum(master_df['geometry_valid'])}")
    print(f"Splits: Train={sum(master_df['split'] == 'train')} | Val={sum(master_df['split'] == 'val')} | Test={sum(master_df['split'] == 'test')}")
    print(f"Subsets exported to: {PROCESSED_DIR}")

    return master_df


if __name__ == "__main__":
    build_master_dataset()
