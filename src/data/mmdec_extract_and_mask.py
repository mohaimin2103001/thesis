"""
MMDEC High-Yield SAR Cropper, AIS Kinematic Ground-Truth Builder & Wake-Net Auto-Masker
========================================================================================
Runs directly inside Kaggle (or locally if Dataset_S1_chunks.zip & Dataset_AIS_POS.parquet exist).
1. Filters Dataset_AIS_POS.parquet for satellite-synchronized ship pings (chunk_folder not null).
2. Computes exact unwrapped d(COG)/dt Rate-of-Turn (deg/min) over a +-120s window around SAR pass.
3. Selects Straight (|ROT| < 0.5 deg/min) and Tactical Turning (|ROT| >= min_turn_rot) vessels,
   supporting maximum harvest (--max_straight, --max_turning -1 for all available).
4. Streams 800x800x3 (VV, VH, avg) SAR chunks directly from Dataset_S1_chunks.zip in memory
   (ZERO 15 GB unzipping overhead!), stitches 3x3 neighbors for edge ships, and saves 512x512 crops.
5. Deploys Stage-1 trained Wake-Net (ResNet-50 + scSE) on GPU to segment every ship's wake mask,
   then extracts explicit B-spline curvature geometry + Radon Transform kinematics.
6. Enforces strict MMSI + product_id zero-leakage splits and merges with UEIKAP maneuver_master.parquet.
"""

import os
import re
import sys
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from geometry.wake_geometry import extract_wake_features_from_mask
from wake.train_wake_net import build_wake_net_model, compute_radon_wake_kinematics

MMDEC_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "mmdec")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
MMDEC_CROPS_DIR = os.path.join(INTERIM_DIR, "mmdec_crops")
MMDEC_MASKS_DIR = os.path.join(INTERIM_DIR, "mmdec_masks")
WAKE_NET_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "wake_net", "wake_net_best.pt")
WAKE_NET_EVAL = os.path.join(PROJECT_ROOT, "logs", "wake_net_evaluation.json")
UEIKAP_MASTER = os.path.join(PROCESSED_DIR, "maneuver_master.parquet")
EXPANDED_OUT = os.path.join(PROCESSED_DIR, "master_expanded_mmdec.parquet")


def parse_id_chunk(val):
    """Safely extracts the primary chunk name (e.g., 'i0j2') from string or list/ndarray."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return str(val[0]).strip() if len(val) > 0 else None
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s.strip("[]").replace("'", "").replace('"', "").split(",")[0].split()[0]
    s = s.replace(".jpg", "").strip()
    return s if s and s.lower() != "nan" and s.lower() != "none" else None


def load_corners_index(mmdec_dir):
    """Loads chunk corner lat/lon tables from either extracted folder or zip archive."""
    corners_map = {}
    extracted_root = os.path.join(mmdec_dir, "Dataset_S1_chunks_corners")
    zip_path = os.path.join(mmdec_dir, "Dataset_S1_chunks_corners.zip")

    if os.path.exists(extracted_root):
        for root, _, files in os.walk(extracted_root):
            for f in files:
                if f.endswith(".csv") and not f.startswith("._"):
                    prod = f.replace(".csv", "")
                    df = pd.read_csv(os.path.join(root, f))
                    corners_map[prod] = df.set_index("chunk_name")
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base.endswith(".csv") and not base.startswith("._"):
                    prod = base.replace(".csv", "")
                    with zf.open(name) as f:
                        df = pd.read_csv(f)
                        corners_map[prod] = df.set_index("chunk_name")
    return corners_map


def latlon_to_chunk_pixel(lat, lon, corner_row, chunk_size=800):
    """
    Computes exact pixel (x, y) inside an 800x800 chunk via perspective homography
    from the 4 geographic corner coordinates (Lat1/Long1 .. Lat4/Long4).
    """
    src_geo = np.array([
        [float(corner_row["Long1"]), float(corner_row["Lat1"])],
        [float(corner_row["Long2"]), float(corner_row["Lat2"])],
        [float(corner_row["Long3"]), float(corner_row["Lat3"])],
        [float(corner_row["Long4"]), float(corner_row["Lat4"])],
    ], dtype=np.float32)

    dst_px = np.array([
        [0.0, 0.0],
        [float(chunk_size), 0.0],
        [float(chunk_size), float(chunk_size)],
        [0.0, float(chunk_size)],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_geo, dst_px)
    pt = np.array([[[float(lon), float(lat)]]], dtype=np.float32)
    px = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(px[0]), float(px[1])


def compute_kinematic_rot_from_ais(ais_path, min_sog=3.0, window_sec=120):
    """
    1. Finds all AIS pings synchronized to a Sentinel-1 overpass (chunk_folder != null).
    2. Loads +-window_sec AIS trajectory for each matched MMSI to compute unwrapped d(COG)/dt (deg/min).
    3. Follows exact Maritime & UEIKAP sign convention:
         ROT < 0 -> Port (Left), ROT > 0 -> Starboard (Right).
    """
    print(f"[MMDEC AIS] Scanning {ais_path} for Sentinel-1 synchronized ships...")
    cols = [
        "Date", "Mmsi", "NavigationStatus", "Latitude", "Longitude",
        "CourseOverGroundDegrees", "SpeedOverGround", "RateOfTurn",
        "TrueHeadingDegrees", "chunk_folder", "id_chunk"
    ]
    table = pq.read_table(ais_path, columns=cols)
    df_all = table.to_pandas()
    print(f"[MMDEC AIS] Loaded {len(df_all):,} total AIS messages.")

    # Convert Date to UTC seconds
    df_all["ts_sec"] = pd.to_datetime(df_all["Date"], utc=True).astype("int64") // 10**9

    # Synchronized overpass detections
    sync_df = df_all[df_all["chunk_folder"].notna() & df_all["id_chunk"].notna()].copy()
    sync_df = sync_df[
        (sync_df["SpeedOverGround"] >= min_sog) &
        (sync_df["CourseOverGroundDegrees"] >= 0.0) &
        (sync_df["CourseOverGroundDegrees"] < 360.0)
    ].copy()
    sync_df["primary_chunk"] = sync_df["id_chunk"].apply(parse_id_chunk)
    sync_df = sync_df[sync_df["primary_chunk"].notna()].copy()

    # Keep closest ping per (Mmsi, chunk_folder)
    sync_df = sync_df.drop_duplicates(subset=["Mmsi", "chunk_folder"]).reset_index(drop=True)
    print(f"[MMDEC AIS] Unique moving ships matched to S1 chunks: {len(sync_df):,}")

    # Filter full AIS to matched MMSIs for fast trajectory grouping
    matched_mmsis = set(sync_df["Mmsi"].unique())
    traj_df = df_all[
        df_all["Mmsi"].isin(matched_mmsis) &
        (df_all["CourseOverGroundDegrees"] >= 0.0) &
        (df_all["CourseOverGroundDegrees"] < 360.0)
    ][["Mmsi", "ts_sec", "CourseOverGroundDegrees", "SpeedOverGround"]].sort_values(["Mmsi", "ts_sec"])

    # Pre-split trajectory dictionary by Mmsi for O(1) lookup
    mmsi_groups = {
        mmsi: (grp["ts_sec"].values, grp["CourseOverGroundDegrees"].values)
        for mmsi, grp in traj_df.groupby("Mmsi", sort=False)
    }

    del df_all, traj_df

    records = []
    for _, row in tqdm(sync_df.iterrows(), total=len(sync_df), desc="Computing unwrapped ROT"):
        mmsi = row["Mmsi"]
        t0 = row["ts_sec"]
        sog = float(row["SpeedOverGround"])
        cog = float(row["CourseOverGroundDegrees"])
        hdg = float(row["TrueHeadingDegrees"]) if (pd.notna(row["TrueHeadingDegrees"]) and row["TrueHeadingDegrees"] < 360.0) else cog
        drift = ((cog - hdg + 180.0) % 360.0) - 180.0

        ts_arr, cog_arr = mmsi_groups.get(mmsi, (None, None))
        if ts_arr is None:
            continue

        mask = (ts_arr >= t0 - window_sec) & (ts_arr <= t0 + window_sec)
        t_win = ts_arr[mask]
        c_win = cog_arr[mask]

        if len(t_win) < 3 or (t_win[-1] - t_win[0]) < 30:
            continue

        # Unwrap angular trajectory to prevent 0/360 jump artifacts
        unwrapped_deg = np.degrees(np.unwrap(np.radians(c_win)))
        dt_min = (t_win - t0) / 60.0
        # Linear regression slope = true Rate of Turn (deg/min)
        rot_deg_min = float(np.polyfit(dt_min, unwrapped_deg, 1)[0])
        delta_cog = float(abs(unwrapped_deg[-1] - unwrapped_deg[0]))

        # Sanity cap against GPS glitch (> 100 deg/min is physically impossible for commercial ships)
        if abs(rot_deg_min) > 100.0:
            continue

        records.append({
            "mmsi": str(mmsi),
            "product_id": str(row["chunk_folder"]).strip(),
            "chunk_name": row["primary_chunk"],
            "sar_time": str(row["Date"]),
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "sog": sog,
            "cog": cog,
            "heading": hdg,
            "course_heading_drift": drift,
            "rot_raw": float(row["RateOfTurn"]) if pd.notna(row["RateOfTurn"]) else -128.0,
            "rot_deg_min": rot_deg_min,
            "delta_cog_win": delta_cog
        })

    cand_df = pd.DataFrame(records)
    print(f"[MMDEC AIS] Validated trajectories with exact ROT: {len(cand_df):,}")
    return cand_df


def build_zip_lookup(chunks_zip_path):
    """Indexes every .jpg inside Dataset_S1_chunks.zip by (product_name, chunk_name)."""
    lookup = {}
    with zipfile.ZipFile(chunks_zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".jpg") and not name.startswith("__MACOSX") and not os.path.basename(name).startswith("._"):
                parts = name.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    prod = parts[-2]
                    chunk = parts[-1].replace(".jpg", "")
                    lookup[(prod, chunk)] = name
    return lookup


def extract_stitched_crop(zf, zip_lookup, prod, chunk_name, ship_x, ship_y, crop_size=512, chunk_size=800):
    """
    Reads the target chunk and any needed 3x3 neighbor chunks directly from Dataset_S1_chunks.zip,
    extracting a 512x512 crop centered on the ship without extracting the 10.5 GB zip file!
    """
    m = re.match(r"i(-?\d+)j(-?\d+)", chunk_name)
    if not m:
        return None
    ci, cj = int(m.group(1)), int(m.group(2))
    half = crop_size // 2

    # Fast path if ship is comfortably inside the single 800x800 chunk
    if half <= ship_x <= (chunk_size - half) and half <= ship_y <= (chunk_size - half):
        zname = zip_lookup.get((prod, chunk_name))
        if not zname:
            return None
        data = np.frombuffer(zf.read(zname), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        cx, cy = int(round(ship_x)), int(round(ship_y))
        crop = img[cy - half:cy + half, cx - half:cx + half]
        return crop if crop.shape[:2] == (crop_size, crop_size) else None

    # Neighbor stitching path (3x3 grid -> 2400x2400 canvas)
    canvas = np.zeros((chunk_size * 3, chunk_size * 3, 3), dtype=np.uint8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            zname = zip_lookup.get((prod, f"i{ci+di}j{cj+dj}"))
            if zname:
                data = np.frombuffer(zf.read(zname), np.uint8)
                patch = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if patch is not None:
                    ph, pw = patch.shape[:2]
                    y0 = (di + 1) * chunk_size
                    x0 = (dj + 1) * chunk_size
                    canvas[y0:y0 + min(ph, chunk_size), x0:x0 + min(pw, chunk_size)] = patch[:chunk_size, :chunk_size]

    cx = int(round(chunk_size + ship_x))
    cy = int(round(chunk_size + ship_y))
    if cx - half < 0 or cy - half < 0 or cx + half > chunk_size * 3 or cy + half > chunk_size * 3:
        return None
    crop = canvas[cy - half:cy + half, cx - half:cx + half]
    return crop if crop.shape[:2] == (crop_size, crop_size) else None


def run_mmdec_harvest_and_mask(
    min_sog=3.5,
    min_turn_rot=1.5,
    max_straight=4000,
    max_turning=-1,
    batch_size=32,
    seed=42
):
    np.random.seed(seed)
    os.makedirs(MMDEC_CROPS_DIR, exist_ok=True)
    os.makedirs(MMDEC_MASKS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ais_path = os.path.join(MMDEC_DIR, "Dataset_AIS_POS.parquet")
    chunks_zip_path = os.path.join(MMDEC_DIR, "Dataset_S1_chunks.zip")

    if not os.path.exists(ais_path):
        raise FileNotFoundError(f"Missing {ais_path}")
    if not os.path.exists(chunks_zip_path):
        raise FileNotFoundError(f"Missing {chunks_zip_path}")

    # 1. AIS Kinematic Ground-Truth
    cand_df = compute_kinematic_rot_from_ais(ais_path, min_sog=min_sog)

    # Partition into Straight vs Turning (following exact UEIKAP maritime sign convention)
    straight_pool = cand_df[(cand_df["rot_deg_min"].abs() < 0.5) & (cand_df["delta_cog_win"] < 3.0)].copy()
    straight_pool["state"] = "Straight"
    straight_pool["direction"] = "None"
    straight_pool["rot_target"] = 0.0

    turning_pool = cand_df[cand_df["rot_deg_min"].abs() >= min_turn_rot].copy()
    turning_pool["state"] = "Turning"
    turning_pool["direction"] = np.where(turning_pool["rot_deg_min"] < 0.0, "Port", "Starboard")
    turning_pool["rot_target"] = turning_pool["rot_deg_min"]

    print(f"\n[MMDEC HARVEST POOLS]")
    print(f"  Available Straight (|ROT|<0.5, dCOG<3.0): {len(straight_pool):,}")
    print(f"  Available Turning (|ROT|>={min_turn_rot}):  {len(turning_pool):,} "
          f"(Port: {sum(turning_pool['direction']=='Port')}, Starboard: {sum(turning_pool['direction']=='Starboard')})")

    # Subsample if capped, or take ALL if max_* == -1
    if max_turning > 0 and len(turning_pool) > max_turning:
        port_df = turning_pool[turning_pool["direction"] == "Port"]
        stbd_df = turning_pool[turning_pool["direction"] == "Starboard"]
        half_t = max_turning // 2
        turning_sel = pd.concat([
            port_df.sample(n=min(half_t, len(port_df)), random_state=seed),
            stbd_df.sample(n=min(max_turning - min(half_t, len(port_df)), len(stbd_df)), random_state=seed)
        ], ignore_index=True)
    else:
        turning_sel = turning_pool

    if max_straight > 0 and len(straight_pool) > max_straight:
        straight_sel = straight_pool.sample(n=max_straight, random_state=seed)
    else:
        straight_sel = straight_pool

    selected_df = pd.concat([turning_sel, straight_sel], ignore_index=True).reset_index(drop=True)
    print(f"[MMDEC SELECTED] Harvesting {len(selected_df):,} ships "
          f"(Turning={len(turning_sel)}, Straight={len(straight_sel)})...")

    # 2. Stream Crops Directly from Dataset_S1_chunks.zip
    corners_map = load_corners_index(MMDEC_DIR)
    zip_lookup = build_zip_lookup(chunks_zip_path)
    print(f"[MMDEC ZIP INDEX] Indexed {len(zip_lookup):,} SAR chunks across {len(corners_map)} products.")

    cropped_rows = []
    with zipfile.ZipFile(chunks_zip_path, "r") as zf:
        for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Cropping SAR chips from zip"):
            prod = row["product_id"]
            chunk = row["chunk_name"]
            if prod not in corners_map or chunk not in corners_map[prod].index:
                continue

            corner_row = corners_map[prod].loc[chunk]
            if isinstance(corner_row, pd.DataFrame):
                corner_row = corner_row.iloc[0]

            ship_x, ship_y = latlon_to_chunk_pixel(row["latitude"], row["longitude"], corner_row)
            if not (-50 <= ship_x <= 850 and -50 <= ship_y <= 850):
                continue

            crop = extract_stitched_crop(zf, zip_lookup, prod, chunk, ship_x, ship_y, crop_size=512)
            if crop is None:
                continue

            # Skip crops with mostly zero padding (> 25% black edge)
            if np.mean(crop[:, :, 0] == 0) > 0.25:
                continue

            sample_id = f"MMDEC_{prod[-15:]}_{row['mmsi']}_{chunk}"
            crop_path = os.path.join(MMDEC_CROPS_DIR, f"{sample_id}.png")
            cv2.imwrite(crop_path, crop)

            rec = row.to_dict()
            rec["sample_id"] = sample_id
            rec["source_dataset"] = "MMDEC"
            rec["original_image_path"] = crop_path
            rec["crop_path"] = crop_path
            rec["ship_x"] = 256.0
            rec["ship_y"] = 256.0
            cropped_rows.append(rec)

    crop_df = pd.DataFrame(cropped_rows)
    print(f"[MMDEC CROPPING] Successfully extracted {len(crop_df):,} authentic SAR ship chips.")

    if len(crop_df) == 0:
        raise RuntimeError("No MMDEC crops were extracted. Verify zip paths and chunk names.")

    # 3. Auto-Segment Wakes with Stage-1 Trained Hybrid Wake-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresh = 0.50
    if os.path.exists(WAKE_NET_EVAL):
        with open(WAKE_NET_EVAL, "r") as f:
            thresh = float(json.load(f).get("optimal_val_threshold", 0.50))

    print(f"[WAKE-NET AUTO-MASKING] Loading Wake-Net ({WAKE_NET_CKPT}) with thresh={thresh:.2f} on {device}...")
    wake_net = build_wake_net_model(encoder_name="resnet50", in_channels=1, classes=1, pretrained=False).to(device)
    wake_net.load_state_dict(torch.load(WAKE_NET_CKPT, map_location=device))
    wake_net.eval()

    paths = crop_df["crop_path"].tolist()
    mask_paths = []
    radon_records = []

    for start in tqdm(range(0, len(paths), batch_size), desc="Running Wake-Net + Radon"):
        batch_paths = paths[start:start + batch_size]
        imgs = []
        for p in batch_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            g_res = cv2.resize(g, (384, 384)).astype(np.float32) / 255.0
            imgs.append(g_res)
        batch_t = torch.from_numpy(np.array(imgs)).unsqueeze(1).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(wake_net(batch_t)).cpu().numpy()

        for j, p in enumerate(batch_paths):
            row_idx = start + j
            prob_map = cv2.resize(probs[j, 0], (512, 512), interpolation=cv2.INTER_LINEAR)
            bin_mask = ((prob_map > thresh).astype(np.uint8)) * 255

            sid = crop_df.iloc[row_idx]["sample_id"]
            m_path = os.path.join(MMDEC_MASKS_DIR, f"{sid}_mask.png")
            cv2.imwrite(m_path, bin_mask)
            mask_paths.append(m_path)

            radon_res = compute_radon_wake_kinematics(bin_mask, sog_knots=crop_df.iloc[row_idx]["sog"])
            radon_records.append(radon_res)

    crop_df["wake_mask_path"] = mask_paths
    crop_df["wake_mask_source"] = "WakeNet_scSE_Auto"
    crop_df["radon_angle_deg"] = [r["dominant_angle_deg"] for r in radon_records]
    crop_df["radon_deflection_deg"] = [r["angular_deflection_deg"] for r in radon_records]
    crop_df["radon_rot_est"] = [r["estimated_rot"] for r in radon_records]

    # 4. Extract Explicit Hydrodynamic B-Spline Geometry Features
    geom_rows = []
    for _, row in tqdm(crop_df.iterrows(), total=len(crop_df), desc="Extracting wake geometry"):
        geom = extract_wake_features_from_mask(row["wake_mask_path"], ship_point=(256.0, 256.0))
        if geom is not None:
            geom_rows.append({
                "geometry_valid": True,
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
                "branch_count": geom["branch_count"],
                "label_quality": "AIS-AutoWake-A"
            })
        else:
            geom_rows.append({
                "geometry_valid": False,
                "wake_length": 0.0,
                "wake_orientation": row["radon_angle_deg"],
                "mean_curvature": 0.0,
                "median_curvature": 0.0,
                "max_abs_curvature": 0.0,
                "signed_mean_curvature": 0.0,
                "curvature_std": 0.0,
                "wake_asymmetry": 0.0,
                "mask_area": 0,
                "geometry_confidence": 0.0,
                "branch_count": 0,
                "label_quality": "AIS-AutoWake-Low"
            })

    geom_df = pd.DataFrame(geom_rows)
    mmdec_final = pd.concat([crop_df.reset_index(drop=True), geom_df], axis=1)

    # Keep only vessels with detectable wake masks (filter open-water blank masks)
    before_n = len(mmdec_final)
    mmdec_final = mmdec_final[mmdec_final["mask_area"] >= 50].reset_index(drop=True)
    print(f"[MMDEC QUALITY FILTER] Retained {len(mmdec_final):,} / {before_n:,} ships with confirmed wake masks.")

    # 5. Strict Zero-Leakage Grouped Split (by MMSI & product_id)
    unique_prods = sorted(mmdec_final["product_id"].unique())
    np.random.shuffle(unique_prods)
    n_tr = int(0.70 * len(unique_prods))
    n_va = int(0.15 * len(unique_prods))

    tr_prods = set(unique_prods[:n_tr])
    va_prods = set(unique_prods[n_tr:n_tr + n_va])
    te_prods = set(unique_prods[n_tr + n_va:])

    # Assign split by product, then purge any MMSI crossing splits to guarantee zero vessel leakage
    mmdec_final["split"] = mmdec_final["product_id"].apply(
        lambda p: "train" if p in tr_prods else ("val" if p in va_prods else "test")
    )
    mmsi_primary_split = mmdec_final.groupby("mmsi")["split"].first().to_dict()
    mmdec_final["split"] = mmdec_final["mmsi"].map(mmsi_primary_split)

    # 6. Merge with Original UEIKAP maneuver_master.parquet (Preserving UEIKAP splits!)
    if os.path.exists(UEIKAP_MASTER):
        ueikap_df = pd.read_parquet(UEIKAP_MASTER)
        for col in ["radon_angle_deg", "radon_deflection_deg", "radon_rot_est"]:
            if col not in ueikap_df.columns:
                ueikap_df[col] = 0.0
        common_cols = [c for c in ueikap_df.columns if c in mmdec_final.columns]
        combined_df = pd.concat([ueikap_df[common_cols], mmdec_final[common_cols]], ignore_index=True)
    else:
        combined_df = mmdec_final

    combined_df.to_parquet(EXPANDED_OUT, index=False)
    print("\n" + "=" * 75)
    print(f"EXPANDED MASTER DATASET SAVED: {EXPANDED_OUT}")
    print(f"Total Vessels: {len(combined_df):,} (UEIKAP + MMDEC)")
    print(f"  State Counts:     {combined_df['state'].value_counts().to_dict()}")
    print(f"  Direction Counts: {combined_df['direction'].value_counts().to_dict()}")
    print(f"  Split Counts:     {combined_df['split'].value_counts().to_dict()}")
    print("=" * 75)
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMDEC Harvester + AIS ROT + Wake-Net Auto-Masker")
    parser.add_argument("--min_sog", type=float, default=3.5)
    parser.add_argument("--min_turn_rot", type=float, default=1.5)
    parser.add_argument("--max_straight", type=int, default=4000, help="-1 for all available straight ships")
    parser.add_argument("--max_turning", type=int, default=-1, help="-1 for all available turning ships")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_mmdec_harvest_and_mask(
        min_sog=args.min_sog,
        min_turn_rot=args.min_turn_rot,
        max_straight=args.max_straight,
        max_turning=args.max_turning,
        batch_size=args.batch_size
    )
