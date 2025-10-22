#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_orientation_mapping.py

Purpose
-------
Validate the (row, col) -> (x, y) physical mapping with in-plane direction-cosines (2x2).
The script:
  1) Builds a synthetic binary ellipse mask (2D).
  2) Saves it as NIfTI with chosen spacing/origin/direction (rotation).
  3) Loads it back with `load_mask2d_with_orientation(...)`.
  4) Extracts a contour via skimage and maps it with:
        (A) Our rc_to_physical_xy (D * diag(s) + origin)
        (B) SimpleITK's TransformContinuousIndexToPhysicalPoint
     and checks max |A - B|.
  5) Calls extract_primary_contour(...) with the same metadata and reports
     the Hausdorff distance to the SITK-mapped contour points.
  6) Optionally, saves a PNG overlay for visual QA.

Usage
-----
python scripts/check_orientation_mapping.py --out-dir reports/figures --plot
"""

import os
import math
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")  # safe in headless environments
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# Import your project modules (adjust path if needed)
# Assumes this script runs from repo root or PYTHONPATH includes the module folder.
from medvis.geometry.contour_slice_io import load_mask2d_with_orientation
from medvis.geometry.contour2d import (
    rc_to_physical_xy,
    extract_primary_contour,
    ContourExtractionConfig,
)


# ---------- Utilities ----------


def make_ellipse_mask(shape, center_rc, axes_rc, angle_deg=0.0):
    """
    Create a binary ellipse mask inside an array of given shape = (rows, cols).

    The ellipse is defined in pixel-index space, with a possible in-plane rotation
    (only for the synthetic mask geometry; orientation of the image is encoded
    separately via direction cosines in the image header).

    Args:
        shape: (rows, cols)
        center_rc: (row0, col0) center in pixel coordinates (float allowed)
        axes_rc: (a_rows, b_cols) semi-axes in pixels along row/col axes BEFORE rotation
        angle_deg: rotation of the ellipse in index space (degrees, CCW)

    Returns:
        mask: np.uint8 array with shape (rows, cols), values in {0,1}
    """
    rows, cols = shape
    rr = np.arange(rows, dtype=float)
    cc = np.arange(cols, dtype=float)
    R, C = np.meshgrid(rr, cc, indexing="ij")

    r0, c0 = center_rc
    a, b = axes_rc
    theta = math.radians(angle_deg)
    ct, st = math.cos(theta), math.sin(theta)

    # Shift to center:
    y = R - r0  # row-axis
    x = C - c0  # col-axis

    # Rotate (x, y) -> (x', y') with CCW angle
    xp = x * ct + y * st
    yp = -x * st + y * ct

    # Ellipse equation: (yp/a)^2 + (xp/b)^2 <= 1
    inside = (yp / a) ** 2 + (xp / b) ** 2 <= 1.0
    return inside.astype(np.uint8)


def directed_hausdorff(A, B):
    """
    Directed Hausdorff distance from set A to set B (A and B are Nx2, Mx2).
    For each point a∈A compute min_b ||a - b||, then take the maximum over a∈A.
    """
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    # (N,1,2) - (1,M,2) -> (N,M,2)
    dif = A[:, None, :] - B[None, :, :]
    d2 = np.sum(dif * dif, axis=2)  # (N, M)
    min_each = np.min(d2, axis=1)  # (N,)
    return float(np.sqrt(np.max(min_each)))


# ---------- Main test ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=128, help="mask rows")
    ap.add_argument("--cols", type=int, default=96, help="mask cols")
    ap.add_argument("--sx", type=float, default=0.7, help="spacing x (cols)")
    ap.add_argument("--sy", type=float, default=1.3, help="spacing y (rows)")
    ap.add_argument("--ox", type=float, default=12.0, help="origin x")
    ap.add_argument("--oy", type=float, default=-4.0, help="origin y")
    ap.add_argument(
        "--angle_deg",
        type=float,
        default=30.0,
        help="in-plane rotation (deg) for direction cosines",
    )
    ap.add_argument(
        "--ellipse_angle",
        type=float,
        default=17.0,
        help="index-space rotation (deg) for the synthetic ellipse",
    )
    ap.add_argument(
        "--slice_path",
        type=str,
        default="reports/interim/orientation_check_mask.nii.gz",
    )
    ap.add_argument("--out-dir", type=str, default="reports/figures")
    ap.add_argument("--plot", action="store_true", help="save an overlay PNG")
    ap.add_argument(
        "--resample_n", type=int, default=256, help="samples for the resampled polygon"
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.slice_path), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    rows, cols = args.rows, args.cols
    # Build a synthetic ellipse not touching the frame
    mask_np = make_ellipse_mask(
        shape=(rows, cols),
        center_rc=(rows * 0.45, cols * 0.55),
        axes_rc=(rows * 0.28, cols * 0.22),
        angle_deg=args.ellipse_angle,
    )

    # Create SimpleITK image and set geometry (spacing, origin, direction)
    img = sitk.GetImageFromArray(mask_np.astype(np.uint8))  # shape (rows, cols)
    img.SetSpacing((float(args.sx), float(args.sy)))
    img.SetOrigin((float(args.ox), float(args.oy)))

    # Direction cosines: 2D rotation by angle_deg
    ang = math.radians(args.angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    D = np.array([[c, -s], [s, c]], dtype=float)
    img.SetDirection(tuple(D.flatten()))  # (D00,D01,D10,D11)

    # Save NIfTI (2D)
    sitk.WriteImage(img, args.slice_path)

    # Load via our I/O
    io = load_mask2d_with_orientation(args.slice_path)
    mask2d = io.mask2d
    sx, sy = io.spacing_xy
    ox, oy = io.origin_xy
    D_loaded = io.direction_2x2

    # Sanity: geometry match
    assert np.isclose(sx, args.sx) and np.isclose(sy, args.sy), "Spacing mismatch"
    assert np.isclose(ox, args.ox) and np.isclose(oy, args.oy), "Origin mismatch"
    assert np.allclose(D_loaded, D, atol=1e-12), "Direction matrix mismatch"

    # Extract a raw contour in (row, col) from the loaded mask
    contours_rc = find_contours(mask2d.astype(float), level=0.5)
    # Pick the longest contour (or largest area). Here choose the longest polyline:
    lengths = [c.shape[0] for c in contours_rc]
    rc = np.asarray(
        contours_rc[int(np.argmax(lengths))], dtype=float
    )  # (N,2) rows, cols

    # Map with our rc_to_physical_xy
    xy_ours = rc_to_physical_xy(rc, (sx, sy), (ox, oy), D_loaded)

    # Map with SITK (ground truth)
    # SimpleITK expects continuous index (x=col, y=row) order:
    xy_sitk = np.empty_like(xy_ours)
    for i, (r, c_) in enumerate(rc):
        x_phys, y_phys = img.TransformContinuousIndexToPhysicalPoint(
            (float(c_), float(r))
        )
        xy_sitk[i, 0] = x_phys
        xy_sitk[i, 1] = y_phys

    # Error metrics between our mapping and SITK mapping
    diff = np.linalg.norm(xy_ours - xy_sitk, axis=1)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    tol = 1e-6 * max(1.0, sx, sy)

    print(
        f"[Mapping] max |ours - SITK| = {max_err:.3e}, mean = {mean_err:.3e}, tol = {tol:.3e}"
    )
    if not (max_err <= tol):
        raise AssertionError("Mapping check FAILED: error above tolerance.")

    # Now run the full extraction (orientation-aware) and compare shapes (Hausdorff)
    cfg = ContourExtractionConfig(
        spacing_xy=(sx, sy),
        origin_xy=(ox, oy),
        ensure_ccw=True,
        min_points=args.resample_n,
        # keep other defaults (fill_holes, frame_tol, etc.)
    )
    poly_xy = extract_primary_contour(mask2d, config=cfg, direction_2x2=D_loaded)

    # Hausdorff between our resampled polygon and the SITK-mapped raw contour
    # (Symmetric Hausdorff: max of both directed distances)
    H_ab = directed_hausdorff(poly_xy, xy_sitk)
    H_ba = directed_hausdorff(xy_sitk, poly_xy)
    H = max(H_ab, H_ba)

    # A reasonable tolerance: a small fraction of pixel spacing, since resampling may smooth corners slightly
    hd_tol = 0.5 * min(sx, sy)
    print(
        f"[Extract] Hausdorff distance(poly_xy, SITK_contour) = {H:.3e} (tol={hd_tol:.3e})"
    )
    if not (H <= hd_tol):
        raise AssertionError(
            "Extracted polygon deviates too much from SITK-mapped contour."
        )

    # Optional plot
    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        ax.plot(xy_sitk[:, 0], xy_sitk[:, 1], ".", ms=2, label="SITK-mapped contour")
        ax.plot(
            poly_xy[:, 0], poly_xy[:, 1], "-", lw=1.5, label="Resampled polygon (ours)"
        )
        ax.set_aspect("equal")
        ax.set_title(f"Orientation check: angle={args.angle_deg}°, spacing=({sx},{sy})")
        ax.legend(loc="best")
        out_png = os.path.join(args.out_dir, "orientation_check_overlay.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[Plot] Saved overlay: {out_png}")

    print(
        "OK — orientation mapping is consistent with SimpleITK and the extracted polygon aligns within tolerance."
    )


if __name__ == "__main__":
    main()
