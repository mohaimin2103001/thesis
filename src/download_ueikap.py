"""
High-Speed Resilient Downloader for the Official UEIKAP SAR Dataset
Downloads the official UEIKAP_SAR.zip (1,571,288,093 bytes, ~1.46 GB) from Zenodo (Record 18391027)
using native curl.exe multi-process byte-range chunking, automatic resume, MD5 checksum validation, and automated extraction.
"""

import os
import sys
import time
import math
import shutil
import hashlib
import zipfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ZENODO_DOWNLOAD_URL = "https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1"
EXPECTED_MD5 = "356148359a9ae6b7ae3d3cec2fb61304"
EXPECTED_SIZE = 1571288093  # Exact byte size from official Zenodo release

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DIR = os.path.join(BASE_DIR, "data", "raw")
ZIP_PATH = os.path.join(TARGET_DIR, "UEIKAP_SAR.zip")
EXTRACT_DIR = os.path.join(TARGET_DIR, "UEIKAP_SAR")
PARTS_DIR = os.path.join(TARGET_DIR, "_download_parts")


def download_chunk_curl(chunk_idx, start_byte, end_byte, part_file):
    expected_len = end_byte - start_byte + 1
    if os.path.exists(part_file):
        existing = os.path.getsize(part_file)
        if existing == expected_len:
            return chunk_idx, True
        elif existing > expected_len:
            os.remove(part_file)

    max_attempts = 10
    for attempt in range(max_attempts):
        # We download using curl.exe with exact byte range
        cmd = [
            "curl.exe",
            "-s",
            "-L",
            "--retry", "5",
            "--retry-delay", "2",
            "-r", f"{start_byte}-{end_byte}",
            "-o", part_file,
            ZENODO_DOWNLOAD_URL
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0 and os.path.exists(part_file):
            if os.path.getsize(part_file) == expected_len:
                return chunk_idx, True
        time.sleep(2 ** min(attempt, 3))
    return chunk_idx, False


def download_dataset(num_workers=8, num_chunks=16, force=False):
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Check if authentic dataset is already extracted and verified
    canonical_images = os.path.join(EXTRACT_DIR, "images")
    canonical_ais = os.path.join(EXTRACT_DIR, "AIS_interpolated_SAR_UEIKAP")
    if not force and os.path.exists(canonical_images) and os.path.exists(canonical_ais):
        imgs = os.listdir(canonical_images)
        ais = os.listdir(canonical_ais)
        if len(imgs) >= 500 and len(ais) >= 500:
            print(f"Authentic UEIKAP SAR dataset already verified: {len(imgs)} images, {len(ais)} AIS files.")
            return True

    need_download = True
    if os.path.exists(ZIP_PATH):
        if os.path.getsize(ZIP_PATH) == EXPECTED_SIZE:
            print(f"Archive {ZIP_PATH} already exists with correct size ({EXPECTED_SIZE} bytes).")
            print("Verifying MD5 checksum...")
            hasher = hashlib.md5()
            with open(ZIP_PATH, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024 * 4), b""):
                    hasher.update(chunk)
            if hasher.hexdigest().lower() == EXPECTED_MD5.lower():
                print("MD5 Checksum 100% Verified! Skipping download.")
                need_download = False
            else:
                print("MD5 mismatch on existing archive. Re-downloading...")
        else:
            print(f"Existing file size ({os.path.getsize(ZIP_PATH)}) != {EXPECTED_SIZE}. Re-downloading...")

    if need_download:
        print("\n" + "=" * 65)
        print("DOWNLOADING OFFICIAL UEIKAP SAR DATASET FROM ZENODO (RECORD 18391027)")
        print(f"Target size: {EXPECTED_SIZE / (1024 * 1024):.2f} MB | Chunks: {num_chunks} | Parallel streams: {num_workers}")
        print("=" * 65 + "\n")

        os.makedirs(PARTS_DIR, exist_ok=True)
        chunk_size = math.ceil(EXPECTED_SIZE / num_chunks)

        futures = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size - 1, EXPECTED_SIZE - 1)
                part_file = os.path.join(PARTS_DIR, f"chunk_{i:03d}.part")
                f = executor.submit(download_chunk_curl, i, start, end, part_file)
                futures[f] = i

            completed_chunks = 0
            for f in as_completed(futures):
                idx, success = f.result()
                if not success:
                    raise RuntimeError(f"Failed downloading chunk {idx} after maximum retries.")
                completed_chunks += 1
                pct = (completed_chunks / num_chunks) * 100
                bar = "#" * int(pct // 5)
                print(f"[{bar:20s}] {pct:5.1f}% | Completed chunk {idx + 1:02d}/{num_chunks}")

        print("\nMerging chunk files into single archive: UEIKAP_SAR.zip...")
        with open(ZIP_PATH, 'wb') as outfile:
            for i in range(num_chunks):
                part_file = os.path.join(PARTS_DIR, f"chunk_{i:03d}.part")
                with open(part_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile, length=1024 * 1024 * 2)
                os.remove(part_file)
        try:
            os.rmdir(PARTS_DIR)
        except OSError:
            pass

        print("Validating MD5 checksum of assembled archive...")
        hasher = hashlib.md5()
        with open(ZIP_PATH, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024 * 4), b""):
                hasher.update(chunk)
        calc_md5 = hasher.hexdigest()
        print(f"Calculated MD5 : {calc_md5}")
        print(f"Expected MD5   : {EXPECTED_MD5}")
        if calc_md5.lower() != EXPECTED_MD5.lower():
            raise ValueError(f"MD5 mismatch! Downloaded {calc_md5}, expected {EXPECTED_MD5}")
        print("MD5 Checksum 100% Validated!")

    # Clean out any old/synthetic files
    if os.path.exists(EXTRACT_DIR):
        print(f"Removing old contents in {EXTRACT_DIR}...")
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print(f"\nExtracting {ZIP_PATH} to {TARGET_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print("Archive extraction complete.")

    # Check extracted directory structure and normalize
    possible_roots = [
        os.path.join(TARGET_DIR, "UEIKAP_dataset_SAR"),
        os.path.join(TARGET_DIR, "UEIKAP_dataset", "UEIKAP_dataset_SAR"),
        os.path.join(TARGET_DIR, "UEIKAP_SAR"),
    ]

    source_root = None
    for p in possible_roots:
        if os.path.exists(p) and os.path.isdir(p):
            subdirs = os.listdir(p)
            if "images" in subdirs or "AIS_interpolated_SAR_UEIKAP" in subdirs:
                source_root = p
                break

    if source_root and os.path.abspath(source_root) != os.path.abspath(EXTRACT_DIR):
        print(f"Moving files from {source_root} into canonical {EXTRACT_DIR}...")
        for item in os.listdir(source_root):
            src_item = os.path.join(source_root, item)
            dst_item = os.path.join(EXTRACT_DIR, item)
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            shutil.move(src_item, dst_item)

    final_images = os.path.join(EXTRACT_DIR, "images")
    final_ais = os.path.join(EXTRACT_DIR, "AIS_interpolated_SAR_UEIKAP")
    final_labels = os.path.join(EXTRACT_DIR, "labels")

    print("\n" + "=" * 65)
    print("VERIFICATION OF AUTHENTIC DATASET:")
    if os.path.exists(final_images):
        print(f"  SAR Images       : {len(os.listdir(final_images))} files ({final_images})")
    if os.path.exists(final_ais):
        print(f"  AIS Telemetry    : {len(os.listdir(final_ais))} files ({final_ais})")
    if os.path.exists(final_labels):
        print(f"  Wake Labels      : {len(os.listdir(final_labels))} files ({final_labels})")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    download_dataset(num_workers=8, num_chunks=16, force=True)
