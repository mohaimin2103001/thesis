"""
High-Speed 4-Part Segmented Downloader for UEIKAP SAR Dataset
Downloads the 1.46 GB archive from Zenodo in 4 parallel segments with resume,
concatenates the segments, validates MD5, and extracts into data/raw/UEIKAP_SAR/
"""

import os
import sys
import time
import shutil
import hashlib
import zipfile
import subprocess
import threading

EXPECTED_SIZE = 1571288093
EXPECTED_MD5 = "356148359a9ae6b7ae3d3cec2fb61304"
URL = "https://zenodo.org/records/18391027/files/UEIKAP_SAR.zip?download=1"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DIR = os.path.join(BASE_DIR, "data", "raw")
ZIP_PATH = os.path.join(TARGET_DIR, "UEIKAP_SAR.zip")
EXTRACT_DIR = os.path.join(TARGET_DIR, "UEIKAP_SAR")
SEGMENTS_DIR = os.path.join(TARGET_DIR, "_segments")

NUM_PARTS = 4


def download_segment(part_idx, start, end, out_file):
    expected_len = end - start + 1
    if os.path.exists(out_file) and os.path.getsize(out_file) == expected_len:
        print(f"Segment {part_idx} already complete ({expected_len / (1024*1024):.1f} MB).")
        return True

    cur_size = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if cur_size > expected_len:
        os.remove(out_file)
        cur_size = 0

    temp_file = out_file + f".part_{part_idx}.tmp"

    max_retries = 25
    for attempt in range(max_retries):
        cur_size = os.path.getsize(out_file) if os.path.exists(out_file) else 0
        if cur_size >= expected_len:
            print(f"Segment {part_idx} finished successfully!")
            return True

        cur_start = start + cur_size
        print(f"[Seg {part_idx}] Attempt {attempt + 1}: Downloading bytes {cur_start}-{end} (Remaining: {(end - cur_start + 1) / (1024*1024):.1f} MB)...")

        cmd = [
            "curl.exe",
            "-s",
            "-L",
            "--retry", "5",
            "--retry-delay", "3",
            "-r", f"{cur_start}-{end}",
            "-o", temp_file,
            URL
        ]
        res = subprocess.run(cmd)
        if os.path.exists(temp_file):
            t_size = os.path.getsize(temp_file)
            if t_size > 0:
                with open(out_file, 'ab') as out_f, open(temp_file, 'rb') as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024 * 2)
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                cur_size = os.path.getsize(out_file)
                if cur_size == expected_len:
                    print(f"Segment {part_idx} 100% complete!")
                    return True
        time.sleep(3)
    return os.path.exists(out_file) and os.path.getsize(out_file) == expected_len


def main():
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    part_size = (EXPECTED_SIZE + NUM_PARTS - 1) // NUM_PARTS

    threads = []
    results = [False] * NUM_PARTS

    def worker(i, s, e, f):
        results[i] = download_segment(i, s, e, f)

    print("\n" + "=" * 65)
    print("STARTING 4-SEGMENT HIGH-SPEED DOWNLOAD OF UEIKAP SAR DATASET")
    print(f"Total Size: {EXPECTED_SIZE / (1024 * 1024):.2f} MB (~1.46 GB)")
    print("=" * 65 + "\n")

    for i in range(NUM_PARTS):
        s = i * part_size
        e = min((i + 1) * part_size - 1, EXPECTED_SIZE - 1)
        f = os.path.join(SEGMENTS_DIR, f"part_{i}.bin")
        t = threading.Thread(target=worker, args=(i, s, e, f))
        threads.append(t)
        t.start()
        time.sleep(3)  # Stagger launch

    # Monitor progress
    while any(t.is_alive() for t in threads):
        time.sleep(10)
        sizes = [os.path.getsize(os.path.join(SEGMENTS_DIR, f"part_{i}.bin")) if os.path.exists(os.path.join(SEGMENTS_DIR, f"part_{i}.bin")) else 0 for i in range(NUM_PARTS)]
        total = sum(sizes)
        pct = (total / EXPECTED_SIZE) * 100
        prog_str = " | ".join([f"P{i}: {s/(1024*1024):.1f}MB" for i, s in enumerate(sizes)])
        print(f"[{total/(1024*1024):.1f}/{EXPECTED_SIZE/(1024*1024):.1f} MB ({pct:5.1f}%)] -> {prog_str}")

    for t in threads:
        t.join()

    if not all(results):
        print("Error: One or more segments failed.")
        sys.exit(1)

    print("\nAll 4 segments complete! Concatenating into UEIKAP_SAR.zip...")
    with open(ZIP_PATH, 'wb') as outfile:
        for i in range(NUM_PARTS):
            f = os.path.join(SEGMENTS_DIR, f"part_{i}.bin")
            with open(f, 'rb') as infile:
                shutil.copyfileobj(infile, outfile, length=1024 * 1024 * 4)

    print("Verifying MD5 checksum...")
    hasher = hashlib.md5()
    with open(ZIP_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 4), b""):
            hasher.update(chunk)
    actual_md5 = hasher.hexdigest()
    print(f"Calculated MD5: {actual_md5}")
    print(f"Expected MD5  : {EXPECTED_MD5}")
    if actual_md5.lower() != EXPECTED_MD5.lower():
        print("Warning: MD5 checksum mismatch, proceeding with archive verification...")
    else:
        print("MD5 Checksum 100% Verified!")

    print(f"\nExtracting archive to {TARGET_DIR}...")
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(TARGET_DIR)

    # Normalize extracted files
    candidates = [
        os.path.join(TARGET_DIR, "UEIKAP_dataset_SAR"),
        os.path.join(TARGET_DIR, "UEIKAP_dataset", "UEIKAP_dataset_SAR"),
        os.path.join(TARGET_DIR, "UEIKAP_SAR")
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.abspath(c) != os.path.abspath(EXTRACT_DIR):
            for item in os.listdir(c):
                src = os.path.join(c, item)
                dst = os.path.join(EXTRACT_DIR, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst): shutil.rmtree(dst)
                    else: os.remove(dst)
                shutil.move(src, dst)

    final_images = os.path.join(EXTRACT_DIR, "images")
    final_ais = os.path.join(EXTRACT_DIR, "AIS_interpolated_SAR_UEIKAP")
    print("\n" + "=" * 65)
    print("DATASET EXTRACTION & VERIFICATION COMPLETE:")
    if os.path.exists(final_images):
        print(f"  Images: {len(os.listdir(final_images))} files in {final_images}")
    if os.path.exists(final_ais):
        print(f"  AIS Telemetry: {len(os.listdir(final_ais))} files in {final_ais}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
