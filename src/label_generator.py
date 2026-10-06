"""
Kinematic Ground Truth Derivation from Telemetry (AIS)
Computes Rate of Turn (ROT) and circular course drift angle (Delta Theta) from interpolated AIS tracks.
Applies scientific deadband thresholding to generate clean Binary and 3-Class maneuver labels
on the authentic European Space Agency Sentinel-1 UEIKAP SAR dataset.
"""

import os
import glob
import re
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RAW_SAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "UEIKAP_SAR"))


def normalize_key(fname):
    """
    Normalizes filename to link Sentinel-1 SAR imagery chips with corresponding AIS telemetry CSVs.
    E.g.,
      DAN_ S1A_20240401T165351_TC_Sub1_crop21_FalseColorVVVH_235108529.png
    matches:
      DAN_ S1A_20240401T165351_TC_Sub1_crop21_VV_235108529.tif.csv
    """
    name = os.path.basename(fname)
    name = re.sub(r'\.(png|jpg|tif|tiff|csv)+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_(FalseColorVVVH|VV|VH)_', '_POL_', name)
    return name


def compute_circular_drift(cog, heading):
    """
    Computes circular angular difference between Course Over Ground (COG)
    and True Heading (HDG) in degrees within [-180, +180].
    """
    diff = (cog - heading + 180.0) % 360.0 - 180.0
    return diff


def parse_ais_telemetry(ais_path):
    """
    Reads an authentic AIS CSV file and extracts kinematic features: SOG, HDG, COG, ROT, and drift.
    """
    try:
        df = pd.read_csv(ais_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if len(df) == 0:
            return None

        row = df.iloc[0]

        sog_col = next((c for c in df.columns if 'sog' in c or 'speed' in c), None)
        hdg_col = next((c for c in df.columns if 'heading' in c or 'hdg' in c), None)
        cog_col = next((c for c in df.columns if 'cog' in c or 'course' in c), None)
        rot_col = next((c for c in df.columns if 'rot' in c or 'turn' in c), None)

        sog = float(row[sog_col]) if sog_col and pd.notnull(row[sog_col]) else 10.0
        cog = float(row[cog_col]) if cog_col and pd.notnull(row[cog_col]) else 0.0
        
        hdg = float(row[hdg_col]) if hdg_col and pd.notnull(row[hdg_col]) else cog
        if hdg == 511.0 or hdg < 0:
            hdg = cog

        drift = compute_circular_drift(cog, hdg)

        rot = float(row[rot_col]) if rot_col and pd.notnull(row[rot_col]) else 0.0

        return {
            "sog": sog,
            "heading": hdg,
            "cog": cog,
            "rot": rot,
            "drift_angle": drift
        }
    except Exception as e:
        return None


def generate_master_index(raw_dir=RAW_SAR_DIR, output_csv=None, seed=42):
    """
    Scans the images and AIS directories, computes ground truth, filters ambiguous samples,
    and partitions the dataset into official train, val, and test splits.
    """
    if output_csv is None:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        output_csv = os.path.join(PROCESSED_DIR, "dataset_index.csv")

    images_dir = os.path.join(raw_dir, "images")
    ais_dir = os.path.join(raw_dir, "AIS_interpolated_SAR_UEIKAP")

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.tif", "*.tiff"):
        image_paths.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))

    print(f"Discovered {len(image_paths)} SAR image chips in {images_dir}")

    # Index AIS files by normalized key
    ais_map = {}
    if os.path.exists(ais_dir):
        for f in os.listdir(ais_dir):
            if f.endswith(".csv"):
                k = normalize_key(f)
                ais_map[k] = os.path.join(ais_dir, f)

    records = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        k = normalize_key(img_path)
        ais_candidate = ais_map.get(k, None)

        telemetry = parse_ais_telemetry(ais_candidate) if ais_candidate else None
        if telemetry is None:
            continue

        sog = telemetry["sog"]
        rot = telemetry["rot"]
        drift = telemetry["drift_angle"]
        hdg = telemetry["heading"]
        cog = telemetry["cog"]

        # Kinematic pre-filter: SOG >= 2.0 knots (eliminate moored/drifting vessels)
        if sog < 2.0:
            continue

        # Scientific classification thresholds:
        # Straight (0): |ROT| < 0.5 deg/min AND |Delta Theta| < 5.0 deg
        # Turning (1):  |ROT| >= 0.5 deg/min OR |Delta Theta| >= 10.0 deg
        is_straight = (abs(rot) < 0.5) and (abs(drift) < 5.0)
        is_turning = (abs(rot) >= 0.5) or (abs(drift) >= 10.0)

        if is_straight:
            binary_label = 0
            class_3 = 0  # Straight
        elif is_turning:
            binary_label = 1
            if rot < -0.5 or (abs(rot) < 0.5 and drift <= -10.0):
                class_3 = 1  # Left Turn (Port)
            else:
                class_3 = 2  # Right Turn (Starboard)
        else:
            # Ambiguous buffer zone: Pruned to eliminate label noise
            continue

        # Preserve official partition (train / val / test) from directory structure if available
        parent_dir = os.path.basename(os.path.dirname(img_path)).lower()
        if parent_dir in ["train", "val", "test"]:
            partition = parent_dir
        else:
            partition = "train"

        records.append({
            "image_id": base_name,
            "image_path": img_path,
            "sog_knots": sog,
            "heading_deg": hdg,
            "cog_deg": cog,
            "rot_deg_min": rot,
            "drift_angle_deg": drift,
            "maneuver_binary": binary_label,
            "maneuver_3class": class_3,
            "split": partition
        })

    df = pd.DataFrame(records)
    print(f"Clean samples after kinematic filtering & buffer pruning: {len(df)}")
    print(f"Binary distribution: Straight={sum(df['maneuver_binary'] == 0)}, Turning={sum(df['maneuver_binary'] == 1)}")
    print(f"3-Class distribution: Straight={sum(df['maneuver_3class'] == 0)}, Left={sum(df['maneuver_3class'] == 1)}, Right={sum(df['maneuver_3class'] == 2)}")

    print("\nPartition breakdown:")
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        st = sum(sub["maneuver_binary"] == 0)
        tr = sum(sub["maneuver_binary"] == 1)
        print(f"  {split_name:5s}: Total={len(sub):3d} | Straight={st:3d} | Turning={tr:2d}")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved master labeled dataset index to: {output_csv}")
    return df


if __name__ == "__main__":
    generate_master_index()
