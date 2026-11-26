"""
Hydrodynamic Wake Centerline Extraction & Signed Curvature Geometry (Phases 18-20)
Performs morphological skeletonization, smooth spline fitting, analytical signed curvature calculation,
and canonical ship-frame orientation for physical maneuver inference.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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


def extract_ordered_centerline(cleaned_mask, ship_point=None, max_pts=600):
    """
    Skeletonizes cleaned binary mask, counts true skeleton junction branches,
    and extracts an ordered chain from the ship end toward the wake tail in fast O(N) time.
    """
    bool_mask = cleaned_mask > 127
    if np.sum(bool_mask) > 0.35 * bool_mask.size:
        return None, 0, 0.0

    skeleton = skeletonize(bool_mask)
    y_coords, x_coords = np.where(skeleton)

    if len(x_coords) < 10:
        return None, 0, 0.0

    # Count 8-connected branch/junction pixels (>2 neighbors on skeleton)
    sk_u8 = skeleton.astype(np.uint8)
    nbrs = cv2.filter2D(sk_u8, -1, np.ones((3, 3), dtype=np.uint8)) - sk_u8
    branch_count = int(np.sum((sk_u8 == 1) & (nbrs >= 3)))

    pts = np.column_stack([x_coords, y_coords])
    if len(pts) > max_pts:
        step = max(1, len(pts) // max_pts)
        pts = pts[::step]

    if ship_point is None:
        h, w = cleaned_mask.shape
        ship_point = (w * 0.5, h * 0.5)

    # Start from the skeleton point closest to the ship center
    dists = np.hypot(pts[:, 0] - ship_point[0], pts[:, 1] - ship_point[1])
    ship_idx = int(np.argmin(dists))
    curr_pt = pts[ship_idx]
    ordered = [curr_pt]

    # Vectorized boolean-masked greedy chain tracing (avoids O(N^2) Python set/list rebuilding)
    active = np.ones(len(pts), dtype=bool)
    active[ship_idx] = False
    while np.any(active):
        act_idx = np.flatnonzero(active)
        cand = pts[act_idx]
        d2 = (cand[:, 0] - curr_pt[0]) ** 2 + (cand[:, 1] - curr_pt[1]) ** 2
        j = int(np.argmin(d2))
        if d2[j] > 225.0:  # 15.0 px radius squared
            break
        orig_i = int(act_idx[j])
        curr_pt = pts[orig_i]
        ordered.append(curr_pt)
        active[orig_i] = False

    ordered_pts = np.array(ordered)
    if len(ordered_pts) < 10:
        return None, branch_count, 0.0

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


def compute_pca_bend_features(bool_mask, ship_point=None, num_slices=16):
    """
    Gap-tolerant hydrodynamic wake geometry via oriented PCA frame and slice-median centerline fitting.
    Projects foreground wake pixels onto principal wake axis u (pointing ship -> tail) and normal v,
    then fits linear and quadratic profiles across u-slices to quantify curvature even across broken masks.
    """
    ys, xs = np.where(bool_mask)
    if len(xs) < 25:
        return None

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean_pt = pts.mean(axis=0)
    centered = pts - mean_pt
    cov = np.cov(centered.T) if len(pts) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]

    h, w = bool_mask.shape
    sp = np.array([w * 0.5, h * 0.5], dtype=np.float64) if ship_point is None else np.array(ship_point, dtype=np.float64)
    if np.dot(mean_pt - sp, axis) < 0.0:
        axis = -axis

    ortho = np.array([-axis[1], axis[0]], dtype=np.float64)
    rel = pts - sp
    u_c = rel @ axis
    v_c = rel @ ortho

    u_low, u_high = np.percentile(u_c, [2, 98])
    u_span = max(10.0, float(u_high - u_low))
    wake_width = float(np.percentile(v_c, 90) - np.percentile(v_c, 10))
    axis_angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))

    # Track slice-median centerline along u so Kelvin V-arms don't bias centroid curvature
    edges = np.linspace(u_low, u_high, num_slices + 1)
    uc_m, vc_m = [], []
    for i in range(num_slices):
        in_slice = (u_c >= edges[i]) & (u_c <= edges[i + 1])
        if np.sum(in_slice) >= 3:
            uc_m.append(0.5 * (edges[i] + edges[i + 1]))
            vc_m.append(float(np.median(v_c[in_slice])))

    if len(uc_m) < 5:
        uc_m = u_c
        vc_m = v_c
    else:
        uc_m = np.asarray(uc_m, dtype=np.float64)
        vc_m = np.asarray(vc_m, dtype=np.float64)

    u_norm = (uc_m - uc_m.mean()) / (0.5 * u_span)
    c_lin = np.polyfit(u_norm, vc_m, 1)
    rmse_lin = float(np.sqrt(np.mean((vc_m - np.polyval(c_lin, u_norm)) ** 2)))
    c_quad = np.polyfit(u_norm, vc_m, 2)
    rmse_quad = float(np.sqrt(np.mean((vc_m - np.polyval(c_quad, u_norm)) ** 2)))

    a2 = float(c_quad[0])  # sagitta deflection in px over normalized [-1, 1] wake span
    lin_quad_gain = max(0.0, (rmse_lin - rmse_quad) / (rmse_lin + 1e-4))

    u_med = float(np.median(uc_m))
    head = uc_m <= u_med
    tail = uc_m > u_med
    slope_h = float(np.polyfit(uc_m[head], vc_m[head], 1)[0]) if np.sum(head) >= 3 else 0.0
    slope_t = float(np.polyfit(uc_m[tail], vc_m[tail], 1)[0]) if np.sum(tail) >= 3 else 0.0
    pca_defl_deg = float(np.degrees(np.arctan(slope_t) - np.arctan(slope_h)))

    return {
        "pca_span": u_span,
        "pca_width": wake_width,
        "pca_abs_bend": abs(a2) / (u_span + 1.0),
        "pca_signed_bend": a2 / (u_span + 1.0),
        "pca_bend_px": abs(a2),
        "pca_lin_quad_gain": lin_quad_gain,
        "pca_abs_defl_deg": abs(pca_defl_deg),
        "pca_signed_defl_deg": pca_defl_deg,
        "pca_axis_angle_deg": axis_angle_deg,
    }


def estimate_sar_streak_angle_deg(sar_gray, ship_point=None):
    """
    Fallback canonical wake axis estimator directly from SAR intensity anomalies around ship center
    when the Wake-Net segmentation mask is too sparse (< 50 px) for PCA mask fitting.
    Returns an oriented angle in (-180, 180] pointing along the highest-contrast streak half.
    """
    if sar_gray is None or sar_gray.size == 0:
        return -90.0
    h, w = sar_gray.shape[:2]
    sp_x, sp_y = (w * 0.5, h * 0.5) if ship_point is None else ship_point

    f_img = sar_gray.astype(np.float32)
    bg = cv2.GaussianBlur(f_img, (31, 31), 0)
    anom = np.abs(f_img - bg)

    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx.astype(np.float32) - float(sp_x)
    dy = yy.astype(np.float32) - float(sp_y)
    r = np.hypot(dx, dy)
    # Exclude bright ship hull center (r < 12px) and border margin
    ring = (r >= 12.0) & (r <= 0.45 * min(h, w))
    weights = anom * ring.astype(np.float32)
    thr = float(np.percentile(weights[ring], 88)) if np.any(ring) else 0.0
    sel = weights >= max(thr, 1e-3)
    if np.sum(sel) < 25:
        return -90.0

    pts = np.column_stack([xx[sel].astype(np.float64), yy[sel].astype(np.float64)])
    w_pts = weights[sel].astype(np.float64)
    w_sum = float(np.sum(w_pts)) + 1e-8
    mean_pt = (pts * w_pts[:, None]).sum(axis=0) / w_sum
    centered = pts - mean_pt
    cov = (centered * w_pts[:, None]).T @ centered / w_sum
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    if np.dot(mean_pt - np.array([sp_x, sp_y]), axis) < 0.0:
        axis = -axis
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def extract_wake_features_from_mask(mask_path, ship_point=None):
    """
    High-level end-to-end extractor from a binary wake mask file to explicit geometry dictionary.
    Always computes gap-tolerant PCA quadratic bend descriptors whenever mask_area >= 50 px,
    even if skeletonization fragments.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask_area = int(np.sum(mask > 127))
    if mask_area < 50:
        return None

    cleaned = clean_wake_mask(mask)
    pca_feats = compute_pca_bend_features(cleaned > 127, ship_point=ship_point)
    centerline_pts, branch_count, conf = extract_ordered_centerline(cleaned, ship_point=ship_point)

    geom = compute_signed_curvature_profile(centerline_pts) if centerline_pts is not None else None
    if geom is None:
        if pca_feats is None:
            return None
        geom = {
            "wake_length": float(pca_feats["pca_span"]),
            "wake_orientation": float(pca_feats["pca_axis_angle_deg"]),
            "mean_curvature": float(pca_feats["pca_abs_bend"]),
            "median_curvature": float(pca_feats["pca_abs_bend"]),
            "max_abs_curvature": float(pca_feats["pca_abs_bend"]),
            "signed_mean_curvature": float(pca_feats["pca_signed_bend"]),
            "curvature_std": 0.0,
            "wake_asymmetry": float(pca_feats["pca_signed_bend"] * 0.5),
        }

    geom["mask_area"] = mask_area
    geom["geometry_confidence"] = float(conf) if conf > 0 else 0.25
    geom["branch_count"] = int(branch_count)
    if pca_feats is not None:
        geom.update(pca_feats)
    else:
        for k in (
            "pca_span", "pca_width", "pca_abs_bend", "pca_signed_bend",
            "pca_bend_px", "pca_lin_quad_gain", "pca_abs_defl_deg",
            "pca_signed_defl_deg", "pca_axis_angle_deg",
        ):
            geom[k] = 0.0
    return geom
