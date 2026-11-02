"""
Hydrodynamic Wake Centerline Extraction & Signed Curvature Geometry (Phases 18-20)
Performs morphological skeletonization, smooth spline fitting, analytical signed curvature calculation,
and canonical ship-frame orientation for physical maneuver inference.
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
import cv2


def clean_wake_mask(binary_mask, min_size=50, hole_size=100):
    """
    Cleans raw binary mask (0 or 255) using connected components filtering and small-hole filling.
    """
    bool_mask = binary_mask > 127
    bool_mask = remove_small_objects(bool_mask, min_size=min_size)
    bool_mask = remove_small_holes(bool_mask, area_threshold=hole_size)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = (bool_mask.astype(np.uint8) * 255)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def extract_ordered_centerline(cleaned_mask, ship_point=None):
    """
    Skeletonizes cleaned binary mask and extracts an ordered list of points from ship towards wake tail.
    """
    bool_mask = cleaned_mask > 127
    skeleton = skeletonize(bool_mask)
    y_coords, x_coords = np.where(skeleton)

    if len(x_coords) < 10:
        return None, 0, 0.0

    pts = np.column_stack([x_coords, y_coords])

    # If ship_point (x, y) is not provided, estimate ship end as the densest extremity
    if ship_point is None:
        # Distance transform to find thickest wake region (near vessel hull)
        dist_map = distance_transform_edt(bool_mask)
        dists = [dist_map[int(y), int(x)] for x, y in pts]
        ship_idx = int(np.argmax(dists))
        start_pt = pts[ship_idx]
    else:
        # Find closest skeleton point to ship coordinates
        dists = np.hypot(pts[:, 0] - ship_point[0], pts[:, 1] - ship_point[1])
        ship_idx = int(np.argmin(dists))
        start_pt = pts[ship_idx]

    # Greedy chain tracing from ship point to wake tail
    remaining = set(range(len(pts)))
    ordered = [start_pt]
    remaining.remove(ship_idx)
    curr_pt = start_pt

    branch_count = 0

    while remaining:
        rem_indices = list(remaining)
        rem_pts = pts[rem_indices]
        dist_to_curr = np.hypot(rem_pts[:, 0] - curr_pt[0], rem_pts[:, 1] - curr_pt[1])
        min_idx_in_rem = int(np.argmin(dist_to_curr))
        min_dist = dist_to_curr[min_idx_in_rem]

        # Connected neighborhood step threshold (max 5 pixels jump)
        if min_dist > 5.0:
            break

        nearest_orig_idx = rem_indices[min_idx_in_rem]
        curr_pt = pts[nearest_orig_idx]
        ordered.append(curr_pt)
        remaining.remove(nearest_orig_idx)

    ordered_pts = np.array(ordered)
    if len(ordered_pts) < 10:
        return None, branch_count, 0.0

    # Confidence score based on chain length vs total skeleton points
    confidence = min(1.0, len(ordered_pts) / float(len(pts)))
    return ordered_pts, branch_count, confidence


def compute_signed_curvature_profile(ordered_pts, num_eval_points=100, smoothing=5.0):
    """
    Fits smoothing B-spline to ordered centerline (x(s), y(s)),
    computes analytical derivatives and signed curvature kappa(s) = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2).
    """
    if ordered_pts is None or len(ordered_pts) < 10:
        return None

    # Remove consecutive duplicate points
    diffs = np.diff(ordered_pts, axis=0)
    non_zero = np.hypot(diffs[:, 0], diffs[:, 1]) > 1e-4
    pts = np.vstack([ordered_pts[0:1], ordered_pts[1:][non_zero]])

    if len(pts) < 10:
        return None

    try:
        # Fit cubic parametric spline
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=smoothing * len(pts), k=3)
        u_fine = np.linspace(0, 1, num_eval_points)
        x_smooth, y_smooth = interpolate.splev(u_fine, tck)

        # 1st and 2nd derivatives
        dx, dy = interpolate.splev(u_fine, tck, der=1)
        ddx, ddy = interpolate.splev(u_fine, tck, der=2)

        # Signed planar curvature: kappa = (dx * ddy - dy * ddx) / (dx^2 + dy^2)^(3/2)
        denom = (dx**2 + dy**2)**(1.5) + 1e-8
        kappa = (dx * ddy - dy * ddx) / denom

        # Arc length along smoothed centerline
        ds = np.hypot(np.diff(x_smooth), np.diff(y_smooth))
        wake_length = float(np.sum(ds))

        # Overall wake orientation angle (radians & degrees)
        chord_dx = x_smooth[-1] - x_smooth[0]
        chord_dy = y_smooth[-1] - y_smooth[0]
        wake_orientation_deg = float(np.degrees(np.arctan2(chord_dy, chord_dx)))

        # Summary curvature features
        mean_curv = float(np.mean(np.abs(kappa)))
        median_curv = float(np.median(np.abs(kappa)))
        max_abs_curv = float(np.max(np.abs(kappa)))
        signed_mean_curv = float(np.mean(kappa))
        curv_std = float(np.std(kappa))

        # Lateral asymmetry: area difference of curve relative to chord line
        chord_vec = np.array([chord_dx, chord_dy])
        chord_norm = np.hypot(chord_dx, chord_dy) + 1e-6
        normal_vec = np.array([-chord_dy, chord_dx]) / chord_norm

        deviations = np.array([
            np.dot(np.array([x_smooth[i] - x_smooth[0], y_smooth[i] - y_smooth[0]]), normal_vec)
            for i in range(len(x_smooth))
        ])
        wake_asymmetry = float(np.sum(deviations) / (wake_length + 1e-6))

        return {
            "smoothed_centerline": np.column_stack([x_smooth, y_smooth]),
            "curvature_profile": kappa,
            "wake_length": wake_length,
            "wake_orientation": wake_orientation_deg,
            "mean_curvature": mean_curv,
            "median_curvature": median_curv,
            "max_abs_curvature": max_abs_curv,
            "signed_mean_curvature": signed_mean_curv,
            "curvature_std": curv_std,
            "wake_asymmetry": wake_asymmetry
        }
    except Exception as e:
        return None


def extract_wake_features_from_mask(mask_path, ship_point=None):
    """
    High-level end-to-end extractor from a binary wake mask file to explicit geometry dictionary.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask_area = int(np.sum(mask > 127))
    if mask_area < 50:
        return None

    cleaned = clean_wake_mask(mask)
    centerline_pts, branch_count, conf = extract_ordered_centerline(cleaned, ship_point=ship_point)

    if centerline_pts is None:
        return None

    geom = compute_signed_curvature_profile(centerline_pts)
    if geom is None:
        return None

    geom["mask_area"] = mask_area
    geom["geometry_confidence"] = conf
    geom["branch_count"] = branch_count
    return geom
