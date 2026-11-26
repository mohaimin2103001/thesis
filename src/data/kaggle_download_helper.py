"""
Resilient Zenodo Downloader for Kaggle (Guards against HTTP 503 maintenance pages)
Deletes any corrupt HTML error files (< min_mb) and uses curl -f with automatic retry/resume.
"""

import os
import sys
import time
import shutil
import zipfile
import subprocess

ASSETS = [
    {
        "name": "UEIKAP_SAR.zip",
        "url": "https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1",
        "path": "data/raw/UEIKAP_SAR.zip",
        "min_bytes": 1_400_000_000,
    },
    {
        "name": "Dataset_AIS_POS.parquet",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_AIS_POS.parquet/content",
        "path": "data/raw/mmdec/Dataset_AIS_POS.parquet",
        "min_bytes": 400_000_000,
    },
    {
        "name": "Dataset_S1_chunks_corners.zip",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_S1_chunks_corners.zip/content",
        "path": "data/raw/mmdec/Dataset_S1_chunks_corners.zip",
        "min_bytes": 4_000_000,
    },
    {
        "name": "Dataset_S1_SatShipAI_outputs.zip",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_S1_SatShipAI_outputs.zip/content",
        "path": "data/raw/mmdec/Dataset_S1_SatShipAI_outputs.zip",
        "min_bytes": 500_000,
    },
    {
        "name": "Dataset_S1_chunks.zip",
        "url": "https://zenodo.org/api/records/17491518/files/Dataset_S1_chunks.zip/content",
        "path": "data/raw/mmdec/Dataset_S1_chunks.zip",
        "min_bytes": 9_500_000_000,
    },
]


def check_zenodo_online(url):
    res = subprocess.run(["curl", "-sI", url], capture_output=True, text=True)
    header = res.stdout.splitlines()[0] if res.stdout else ""
    if "503" in header or "502" in header:
        return False, header
    return True, header


def download_verified(item, max_wait_rounds=6):
    path = item["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    corrupt_thresh = min(100_000, max(10_000, item["min_bytes"] // 10))

    # Remove corrupt HTML error pages (e.g., 1598-byte 503 page) without touching valid <1 MB zips
    if os.path.exists(path) and os.path.getsize(path) < corrupt_thresh:
        print(f"[CLEANUP] Removing corrupt small file ({os.path.getsize(path)} bytes): {path}")
        os.remove(path)

    if os.path.exists(path) and os.path.getsize(path) >= item["min_bytes"]:
        print(f"[VERIFIED] {item['name']} already complete ({os.path.getsize(path)/(1024**2):.1f} MB).")
        return True

    for attempt in range(1, max_wait_rounds + 1):
        online, hdr = check_zenodo_online(item["url"])
        if not online:
            print(f"[ZENODO MAINTENANCE] {item['name']} returned '{hdr}'. Waiting 30s (attempt {attempt}/{max_wait_rounds})...")
            time.sleep(30)
            continue

        print(f"[DOWNLOADING] {item['name']} from Zenodo...")
        cmd = ["curl", "-f", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10", item["url"], "-o", path]
        subprocess.run(cmd)

        if os.path.exists(path) and os.path.getsize(path) >= item["min_bytes"]:
            print(f"[OK] {item['name']} verified ({os.path.getsize(path)/(1024**2):.1f} MB).")
            return True

        if os.path.exists(path) and os.path.getsize(path) < corrupt_thresh:
            os.remove(path)
        time.sleep(15)

    raise RuntimeError(f"Failed to download {item['name']} — Zenodo server may still be in maintenance (HTTP 503). Retry in a few minutes.")


def main():
    for item in ASSETS:
        download_verified(item)

    # Extract lightweight metadata archives (SatShipAI radar detections & corners)
    sat_dir = "data/raw/mmdec/Dataset_S1_SatShipAI_outputs"
    sat_zip = "data/raw/mmdec/Dataset_S1_SatShipAI_outputs.zip"
    if os.path.exists(sat_zip) and not os.path.exists(sat_dir):
        print("[UNZIPPING] Dataset_S1_SatShipAI_outputs.zip...")
        with zipfile.ZipFile(sat_zip, "r") as zf:
            zf.extractall(sat_dir)

    # Extract & normalize UEIKAP if needed
    ueikap_dir = "data/raw/UEIKAP_SAR"
    if not os.path.exists(os.path.join(ueikap_dir, "images")):
        print("[UNZIPPING] UEIKAP_SAR.zip...")
        with zipfile.ZipFile("data/raw/UEIKAP_SAR.zip", "r") as zf:
            zf.extractall(ueikap_dir)
        for root, dirs, _ in os.walk("data/raw"):
            if "images" in dirs and root != ueikap_dir:
                for d in dirs:
                    src_p = os.path.join(root, d)
                    dst_p = os.path.join(ueikap_dir, d)
                    if not os.path.exists(dst_p):
                        shutil.move(src_p, dst_p)


if __name__ == "__main__":
    main()
