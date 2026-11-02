"""
UEIKAP Dataset Auditor and Wake Mask Rasterizer (Phases 2 & 3)
Extracts official COCO polygon annotations and rasterizes them into clean binary masks.
Enforces rigorous QA checks and produces metadata reports.
"""

import os
import glob
import re
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2

RAW_DIR = r"e:\thesis\data\raw\UEIKAP_SAR"
INTERIM_MASKS_DIR = r"e:\thesis\data\interim\ueikap_masks"
FIGURES_DIR = r"e:\thesis\figures\ueikap_mask_qa"
PROCESSED_DIR = r"e:\thesis\data\processed"
REJECTED_DIR = r"e:\thesis\data\rejected"


def normalize_key(filename):
    """
    Strips extensions and polarization tags to map images, JSONs, and AIS files.
    """
    name = os.path.basename(filename)
    name = re.sub(r'\.(png|jpg|tif|tiff|csv|json)+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_(FalseColorVVVH|VV|VH|annotations)_', '_POL_', name)
    name = re.sub(r'_annotations_', '_', name)
    name = re.sub(r'_coco$', '', name)
    name = name.strip()
    return name


def audit_ueikap_schema(raw_dir=RAW_DIR):
    """
    Audits JSON schemas and reports statistics across all annotations.
    """
    jsons = glob.glob(os.path.join(raw_dir, "labels_jsons", "*.json"))
    schema_info = {
        "total_annotations": len(jsons),
        "categories_found": set(),
        "annotation_types": set(),
        "image_shapes": set(),
        "polygons_per_sample": []
    }

    for jpath in jsons:
        try:
            with open(jpath, 'r') as f:
                data = json.load(f)
            for c in data.get("categories", []):
                schema_info["categories_found"].add(c.get("name"))
            for img in data.get("images", []):
                schema_info["image_shapes"].add((img.get("width"), img.get("height")))
            anns = data.get("annotations", [])
            schema_info["polygons_per_sample"].append(len(anns))
            for a in anns:
                if "segmentation" in a:
                    schema_info["annotation_types"].add("polygon")
                elif "bbox" in a:
                    schema_info["annotation_types"].add("bbox_only")
        except Exception as e:
            pass

    report = {
        "total_annotations": schema_info["total_annotations"],
        "categories": list(schema_info["categories_found"]),
        "annotation_types": list(schema_info["annotation_types"]),
        "image_shapes": list(schema_info["image_shapes"]),
        "avg_polygons_per_sample": float(np.mean(schema_info["polygons_per_sample"])) if schema_info["polygons_per_sample"] else 0.0,
        "max_polygons_per_sample": int(np.max(schema_info["polygons_per_sample"])) if schema_info["polygons_per_sample"] else 0
    }
    
    report_path = os.path.join(PROCESSED_DIR, "ueikap_schema_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Schema report saved to {report_path}: {report}")
    return report


def rasterize_polygon_mask(segmentation, width=512, height=512):
    """
    Rasterizes a list of polygon vertex lists [[x1, y1, x2, y2, ...], ...] into a binary mask (uint8, 0 or 255).
    """
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for poly in segmentation:
        if len(poly) < 6:  # Need at least 3 vertices (x, y)
            continue
        # Convert flat list [x1, y1, x2, y2, ...] to list of tuples [(x1, y1), (x2, y2), ...]
        coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(coords, outline=255, fill=255)

    return np.array(mask_img, dtype=np.uint8)


def generate_ueikap_masks(raw_dir=RAW_DIR, save_overlays=True):
    """
    Parses all JSON annotations, aligns with images, creates masks, performs QA,
    and saves artifacts to interim and processed directories.
    """
    os.makedirs(INTERIM_MASKS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    # 1. Discover all images
    images = glob.glob(os.path.join(raw_dir, "images", "**", "*.*"), recursive=True)
    images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
    image_map = {normalize_key(img): img for img in images}

    # 2. Discover all JSONs
    jsons = glob.glob(os.path.join(raw_dir, "labels_jsons", "*.json"))
    json_map = {normalize_key(j): j for j in jsons}

    # 3. Discover all AIS files
    ais_files = glob.glob(os.path.join(raw_dir, "AIS_interpolated_SAR_UEIKAP", "*.csv"))
    ais_map = {normalize_key(a): a for a in ais_files}

    valid_records = []
    rejected_records = []

    total_keys = sorted(list(set(image_map.keys()) | set(json_map.keys())))
    print(f"Total unique normalized keys identified: {len(total_keys)}")

    overlay_count = 0
    max_overlays = 15  # Save first 15 QA overlay figures

    for key in total_keys:
        img_path = image_map.get(key)
        json_path = json_map.get(key)
        ais_path = ais_map.get(key)

        sample_id = f"UEIKAP_{key}"

        if not img_path:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "discovery",
                "reason": "missing_sar_image",
                "details": f"Annotation exists ({json_path}) but no matching SAR image found"
            })
            continue

        if not json_path:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "discovery",
                "reason": "missing_wake_annotation",
                "details": f"SAR image exists ({img_path}) but no matching JSON annotation found"
            })
            continue

        # Load image to verify dimensions
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
                img_np = np.array(im)
        except Exception as e:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "image_loading",
                "reason": "corrupt_image",
                "details": str(e)
            })
            continue

        # Load JSON annotation
        try:
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "annotation_loading",
                "reason": "corrupt_json",
                "details": str(e)
            })
            continue

        annotations = coco_data.get("annotations", [])
        if not annotations:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "qa_check",
                "reason": "empty_annotation_list",
                "details": "JSON contains no annotations"
            })
            continue

        all_segs = []
        for ann in annotations:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                all_segs.extend(seg)

        if not all_segs:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "qa_check",
                "reason": "no_valid_polygons",
                "details": "No polygon coordinates found in annotation"
            })
            continue

        # Rasterize mask
        mask = rasterize_polygon_mask(all_segs, width=img_w, height=img_h)
        mask_area = int(np.sum(mask > 0))

        # QA Check 1: Mask not empty
        if mask_area == 0:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "qa_check",
                "reason": "empty_rasterized_mask",
                "details": "Polygon rasterization yielded 0 non-zero pixels"
            })
            continue

        # QA Check 2: Mask area plausible (<80% of entire chip)
        total_pixels = img_w * img_h
        area_ratio = mask_area / total_pixels
        if area_ratio > 0.80:
            rejected_records.append({
                "sample_id": sample_id,
                "source": "UEIKAP",
                "stage": "qa_check",
                "reason": "implausibly_large_mask",
                "details": f"Mask occupies {area_ratio*100:.1f}% of image chip"
            })
            continue

        # Save binary mask (0 = background, 255 = wake)
        mask_filename = f"{sample_id}_mask.png"
        mask_out_path = os.path.join(INTERIM_MASKS_DIR, mask_filename)
        cv2.imwrite(mask_out_path, mask)

        # Generate QA visual overlay
        if save_overlays and overlay_count < max_overlays:
            if len(img_np.shape) == 2:
                overlay_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4:
                overlay_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                overlay_rgb = img_np.copy()

            # Green contour overlay for wake
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_rgb, contours, -1, (0, 255, 0), 2)
            cv2.putText(overlay_rgb, f"{sample_id} | Area: {mask_area}px", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            overlay_path = os.path.join(FIGURES_DIR, f"{sample_id}_overlay.png")
            cv2.imwrite(overlay_path, overlay_rgb)
            overlay_count += 1

        # Partition from folder name if available
        split = "train"
        parent = os.path.basename(os.path.dirname(img_path)).lower()
        if parent in ["train", "val", "test"]:
            split = parent

        valid_records.append({
            "sample_id": sample_id,
            "image_path": img_path,
            "annotation_path": json_path,
            "ais_path": ais_path if ais_path else "none",
            "mask_path": mask_out_path,
            "annotation_type": "polygon",
            "image_width": img_w,
            "image_height": img_h,
            "mask_area": mask_area,
            "mask_area_ratio": area_ratio,
            "polygon_count": len(all_segs),
            "quality_flag": "PASS",
            "split": split
        })

    # Save outputs as Parquet
    valid_df = pd.DataFrame(valid_records)
    rejected_df = pd.DataFrame(rejected_records)

    mask_index_path = os.path.join(PROCESSED_DIR, "ueikap_mask_index.parquet")
    valid_df.to_parquet(mask_index_path, index=False)

    rejected_path = os.path.join(REJECTED_DIR, "ueikap_rejected_files.parquet")
    rejected_df.to_parquet(rejected_path, index=False)

    print("\n=== UEIKAP Wake Mask Rasterization Summary ===")
    print(f"Total Valid Masks Generated: {len(valid_df)}")
    print(f"Total Rejected Samples: {len(rejected_df)}")
    print(f"Saved Mask Index Parquet: {mask_index_path}")
    print(f"Saved Rejection Log Parquet: {rejected_path}")
    print(f"Saved Visual QA Overlays in: {FIGURES_DIR}")

    return valid_df, rejected_df


if __name__ == "__main__":
    audit_ueikap_schema()
    generate_ueikap_masks()
