"""
MMDEC Product Time Index Builder (Phase 8)
Extracts Sentinel-1 product acquisition timestamps and geospatial bounds from extra_info.csv.
Normalizes timestamps to UTC and saves mmdec_product_time_index.parquet.
"""

import os
import pandas as pd

SAT_DIR = r"e:\thesis\data\raw\mmdec\Dataset_S1_SatShipAI_outputs\Dataset_S1_SatShipAI_outputs"
PROCESSED_DIR = r"e:\thesis\data\processed"


def build_product_time_index():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    info_path = os.path.join(SAT_DIR, "extra_info.csv")

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing {info_path}")

    df = pd.read_csv(info_path)
    print(f"Loaded extra_info.csv with {len(df)} products.")

    # begindatetime is formatted like '2023-07-01 06:23:00'
    df["sar_time_utc"] = pd.to_datetime(df["begindatetime"], utc=True)

    index_df = pd.DataFrame({
        "product_name": df["product_name"],
        "product_width": df["product_width"],
        "product_height": df["product_height"],
        "begindatetime_raw": df["begindatetime"],
        "sar_time_utc": df["sar_time_utc"],
        "north_azimuth_deg": df["North_azimuth"],
        "vertical_pixel_distance": df["vertical_pixelDistance"],
        "lat_min": df[["lat11", "lat12", "lat21", "lat22"]].min(axis=1),
        "lat_max": df[["lat11", "lat12", "lat21", "lat22"]].max(axis=1),
        "lon_min": df[["lon11", "lon12", "lon21", "lon22"]].min(axis=1),
        "lon_max": df[["lon11", "lon12", "lon21", "lon22"]].max(axis=1),
    })

    out_path = os.path.join(PROCESSED_DIR, "mmdec_product_time_index.parquet")
    index_df.to_parquet(out_path, index=False)

    print(f"Saved MMDEC Product Time Index ({len(index_df)} products) to: {out_path}")
    print("Sample time index records:")
    print(index_df[["product_name", "sar_time_utc", "lat_min", "lat_max"]].head(3))
    return index_df


if __name__ == "__main__":
    build_product_time_index()
