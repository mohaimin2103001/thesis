"""
High-Speed Resilient Downloader for MMDEC Metadata & AIS (Phases 1 & 7)
Downloads official MMDEC metadata, SatShipAI detections, chunk corners, and AIS positional telemetry.
"""

import os
import requests
from tqdm import tqdm
import zipfile

MMDEC_DIR = r"e:\thesis\data\raw\mmdec"
os.makedirs(MMDEC_DIR, exist_ok=True)

FILES = [
    {
        "name": "Dataset_S1_SatShipAI_outputs.zip",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_S1_SatShipAI_outputs.zip/content",
        "size_mb": 0.71,
        "unzip": True
    },
    {
        "name": "Dataset_AIS_ShipTypes.csv",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_AIS_ShipTypes.csv/content",
        "size_mb": 0.01,
        "unzip": False
    },
    {
        "name": "Dataset_S1_chunks_corners.zip",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_S1_chunks_corners.zip/content",
        "size_mb": 4.68,
        "unzip": True
    },
    {
        "name": "README.pdf",
        "url": "https://zenodo.org/api/records/17491518/files/README.pdf/content",
        "size_mb": 0.88,
        "unzip": False
    },
    {
        "name": "Dataset_AIS_POS.parquet",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_AIS_POS.parquet/content",
        "size_mb": 430.05,
        "unzip": False
    }
]


def download_file(url, target_path, expected_mb):
    if os.path.exists(target_path):
        actual_mb = os.path.getsize(target_path) / (1024 * 1024)
        if actual_mb >= expected_mb * 0.95:
            print(f"[EXISTS] {os.path.basename(target_path)} ({actual_mb:.2f} MB)")
            return True

    print(f"\n[DOWNLOADING] {os.path.basename(target_path)} (~{expected_mb:.2f} MB)...")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunk

    with open(target_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(target_path)) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    actual_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"[VERIFIED] {os.path.basename(target_path)} downloaded successfully ({actual_mb:.2f} MB).")
    return True


def run_mmdec_download():
    for item in FILES:
        target = os.path.join(MMDEC_DIR, item["name"])
        download_file(item["url"], target, item["size_mb"])
        if item["unzip"] and os.path.exists(target):
            extract_dir = os.path.join(MMDEC_DIR, os.path.splitext(item["name"])[0])
            if not os.path.exists(extract_dir):
                print(f"[UNZIPPING] {item['name']} -> {extract_dir}...")
                with zipfile.ZipFile(target, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)


if __name__ == "__main__":
    run_mmdec_download()
