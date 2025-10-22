from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_holes, binary_erosion

try:
    from skimage.morphology import footprint_rectangle as _fp_rect_backend

    def _fp_rect(h: int, w: int):
        return _fp_rect_backend((h, w))

except Exception:
    try:
        from skimage.morphology import rectangle as _fp_rect_backend_legacy

        def _fp_rect(h: int, w: int):
            return _fp_rect_backend_legacy(h, w)

    except Exception:
        import numpy as _np

        def _fp_rect(h: int, w: int):
            return _np.ones((h, w), dtype=bool)


ArrayF = NDArray[np.float64]
ArrayB = NDArray[np.bool_]

__all__ = [
    "ContourExtractionConfig",
    "extract_primary_contour",
    "resample_closed_polyline",
    "signed_area_2d",
]

# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def _touches_frame(rc_contour: ArrayF, H: int, W: int, tol: float = 1e-3) -> bool:
    """
    True if any vertex of (row,col) contour lies on/too-close to the image frame.
    'tol' is in pixels. Use a *small* tol so internal borders at ~0.5 px survive.
    """
    r = rc_contour[:, 0]
    c = rc_contour[:, 1]
    return (
        (r.min() <= 0.0 + tol)
        or (c.min() <= 0.0 + tol)
        or (r.max() >= (H - 1) - tol)
        or (c.max() >= (W - 1) - tol)
    )


def _filter_frame_contours(
    contours_rc: list[ArrayF], shape_hw: tuple[int, int], tol: float = 1e-3
) -> tuple[list[ArrayF], list[ArrayF]]:
    """Split into (non_frame, frame_touching) using a tight tolerance."""
    H, W = shape_hw
    non_frame: list[ArrayF] = []
    frame: list[ArrayF] = []
    for c in contours_rc:
        (frame if _touches_frame(c, H, W, tol) else non_frame).append(c)
    return non_frame, frame


def signed_area_2d(points_xy: ArrayF) -> float:
    """
    Signed area of a closed polygon via the shoelace formula.

    Parameters
    ----------
    points_xy : (N, 2) float64
        Polygon vertices. If not explicitly closed, the segment
        (N-1)→0 is considered for the area.

    Returns
    -------
    area : float
        Positive for CCW orientation; negative for CW.
    """
    p = np.asarray(points_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")

    x = p[:, 0]
    y = p[:, 1]
    # wrap-around term
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polyline_cumlen(points_xy: ArrayF, closed: bool = True) -> ArrayF:
    """
    Cumulative arclength of a polygon/polyline.

    Parameters
    ----------
    points_xy : (N, 2) float64
    closed : bool
        If True, includes the closing segment N-1→0.

    Returns
    -------
    s : (M,) float64
        Cumulative lengths. If closed, M=N+1 with s[-1] = perimeter.
        If open,   M=N   with s[-1] = total arclength.
    """
    P = np.asarray(points_xy, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")

    if closed:
        Q = np.vstack([P, P[0]])
    else:
        Q = P

    dif = np.diff(Q, axis=0)
    seg = np.linalg.norm(dif, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s


def resample_closed_polyline(points_xy: ArrayF, n: int) -> ArrayF:
    """
    Evenly resample a closed polygon to exactly `n` vertices using linear interpolation.

    Notes
    -----
    - The input polygon is treated as closed; i.e., last→first segment is included.
    - Output is strictly closed in the sense that the first and last vertices are *not*
      repeated (shape is (n, 2)). The closing edge is implied.
    """
    if n < 3:
        raise ValueError("n must be >= 3 for a closed polygon.")

    P = np.asarray(points_xy, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")

    s = _polyline_cumlen(P, closed=True)
    L = s[-1]
    if L <= 0.0:
        raise ValueError("Degenerate polygon (zero perimeter).")

    targets = np.linspace(0.0, L, n + 1)[:-1]  # exclude the duplicate at L
    # Segment indices for each target arclength
    idx = np.searchsorted(s, targets, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)

    # Linear interpolation on each segment
    t = (targets - s[idx]) / (s[idx + 1] - s[idx])
    # Build an expanded polyline with closure (for indexing)
    Q = np.vstack([P, P[0]])
    A = Q[idx]
    B = Q[idx + 1]
    R = A + (B - A) * t[:, None]
    return R


# ---------------------------------------------------------------------------
# Binary mask preprocessing
# ---------------------------------------------------------------------------


def _to_bool_mask(mask: np.ndarray, threshold: float = 0.5) -> ArrayB:
    """Convert input mask (numeric/bool) to a boolean array."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    if m.dtype == np.bool_:
        return m.copy()
    return m.astype(np.float64) > threshold


def _largest_connected_component(mask_bool: ArrayB, connectivity: int = 2) -> ArrayB:
    """Keep only the largest connected component in a boolean mask."""
    if mask_bool.dtype != np.bool_:
        raise ValueError("mask_bool must be boolean.")
    lab = label(mask_bool, connectivity=connectivity, background=0)
    if lab.max() == 0:
        # No foreground labels
        return np.zeros_like(mask_bool, dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # ignore background
    k = int(np.argmax(counts))
    return lab == k


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContourExtractionConfig:
    """
    Configuration for 2D contour extraction from a binary mask.

    Attributes
    ----------
    level : float
        Level passed to skimage.measure.find_contours (0.5 for binary masks).
    fill_holes : bool
        If True, fill all holes in the mask (with a large area threshold).
    simplify_tol : Optional[float]
        If not None, Douglas-Peucker simplification tolerance in *pixel* units
        applied to the *found* contour (not the mask). To keep the dependency
        surface minimal, we only resample evenly; polygon simplification can
        be done downstream if required.
    min_points : int
        Minimum number of points for the returned (resampled) contour.
    ensure_ccw : bool
        If True, enforce CCW orientation in the returned (x,y) polygon.
    spacing_xy : Tuple[float, float]
        Physical spacing (sx, sy) in mm per pixel along x (columns) and y (rows).
    origin_xy : Tuple[float, float]
        Physical origin (ox, oy) in mm at pixel (row=0, col=0).
    pad_on_boundary : bool
        If True, pad the mask by `pad_width` pixels on all sides if
        the foreground touches the image border. This helps to ensure
        closed contours when the object is in contact with the image edge.
    pad_width : int
        Number of pixels to pad the mask on all sides.
    frame_tol : float
        Tolerance in pixels for detecting frame-touching contours.
    erosion_last_resort : bool
        If True, perform one-pixel binary erosion as a last resort
        to detach the foreground from the image border if no
        usable contour is found.
    erosion_iters : int
        Number of binary erosion iterations for the last-resort step.
    """

    level: float = 0.5
    fill_holes: bool = True
    simplify_tol: Optional[float] = None
    min_points: int = 256
    ensure_ccw: bool = True
    spacing_xy: Tuple[float, float] = (1.0, 1.0)
    origin_xy: Tuple[float, float] = (0.0, 0.0)

    # frame handling
    pad_on_boundary: bool = True
    pad_width: int = 1
    frame_tol: float = 1e-3  # tight tolerance for frame-touching detection

    # last-resort erosion when the mask is all-ones
    erosion_last_resort: bool = True
    erosion_iters: int = 5


# ---------------------------------------------------------------------------
# Main contour extraction
# ---------------------------------------------------------------------------


def rc_to_physical_xy(
    points_rc: np.ndarray,
    spacing_xy: Tuple[float, float],
    origin_xy: Tuple[float, float],
    direction_2x2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map pixel coordinates (row, col) to physical coordinates (x, y) using spacing,
    origin, and the 2x2 in-plane direction-cosines matrix.

    Skimage's find_contours returns coordinates as (row, col); we internally
    construct the vector [col, row]^T to match the convention for the linear map:

        [x, y]^T = [ox, oy]^T + D * diag(sx, sy) * [col, row]^T

    Args:
        points_rc:  array (N, 2) with columns [row, col]
        spacing_xy: (sx, sy)
        origin_xy:  (ox, oy)
        direction_2x2: 2x2 direction-cosines; if None, identity is used.

    Returns:
        array (N, 2) with physical coordinates (x, y)
    """
    if points_rc.size == 0:
        return points_rc.copy()

    sx, sy = float(spacing_xy[0]), float(spacing_xy[1])
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    D = (
        np.eye(2, dtype=float)
        if direction_2x2 is None
        else np.array(direction_2x2, dtype=float).reshape(2, 2)
    )

    cols = points_rc[:, 1]
    rows = points_rc[:, 0]
    ij = np.vstack([cols, rows])  # shape (2, N), note [col; row]

    A = D @ np.diag([sx, sy])  # 2×2
    xy = (A @ ij).T + np.array([ox, oy], dtype=float)
    return xy


def extract_primary_contour(
    mask: np.ndarray,
    *,
    config: Optional[ContourExtractionConfig] = None,
    direction_2x2: Optional[np.ndarray] = None,
) -> ArrayF:
    """
    Extract the primary (outer) closed contour from a 2D binary mask and return
    an evenly resampled polygon in physical (x, y) coordinates.

    Orientation-aware: maps (row, col) pixel coordinates to physical (x, y) using
        [x, y]^T = [ox, oy]^T + D * diag(sx, sy) * [col, row]^T,
    where (sx, sy) are pixel spacings, (ox, oy) is the physical origin, and
    D ∈ R^{2×2} is the in-plane direction-cosines matrix. If neither the function
    argument `direction_2x2` nor `config.direction_2x2` is provided, the identity
    is assumed (backward-compatible behavior).

    Robust to: touching-frame artifacts, all-ones masks (via padding or erosion),
    and padding fallback.

    Raises:
        ValueError: if no usable contour can be found or mask is degenerate.
    """
    cfg = config or ContourExtractionConfig()

    # 1) Binarize
    m = _to_bool_mask(mask)

    # 2) Optional hole filling
    if cfg.fill_holes:
        m = remove_small_holes(m, area_threshold=int(m.size), connectivity=2)

    # 3) Largest CC
    m = _largest_connected_component(m, connectivity=2)
    if not m.any():
        raise ValueError("Empty mask after preprocessing (no foreground).")

    H, W = m.shape
    sx, sy = cfg.spacing_xy
    ox, oy = cfg.origin_xy

    # Resolve direction matrix: prefer explicit arg, otherwise from config, otherwise identity.
    D = direction_2x2
    if D is None:
        D = getattr(cfg, "direction_2x2", None)
    if D is None:
        D = np.eye(2, dtype=float)
    else:
        D = np.array(D, dtype=float).reshape(2, 2)

    def rc_to_xy(cont: ArrayF) -> ArrayF:
        """
        Map (row, col) -> (x, y) using: [x, y]^T = [ox, oy]^T + D * diag(sx, sy) * [col, row]^T.
        Skimage returns (row, col) in that order; note the [col; row] stacking below.
        """
        if cont.size == 0:
            return cont.copy()
        col = cont[:, 1]
        row = cont[:, 0]
        ij = np.vstack([col, row])  # shape (2, N) = [col; row]
        A = D @ np.diag([sx, sy])  # 2×2
        xy = (A @ ij).T + np.array([ox, oy], float)  # (N, 2)
        return xy

    # --- SPECIAL CASE: uniform mask ------------------------------------
    uniq = np.unique(m)
    if uniq.size == 1:
        if not uniq[0]:
            raise ValueError("All-zero mask: no contour exists.")
        # All-ones: create an inner boundary deterministically.
        # Option A: padding -> inner contour (preferred; avoids touching the frame after unpadding)
        if cfg.pad_on_boundary:
            pad = max(1, int(cfg.pad_width))
            mp = np.pad(m.astype(np.uint8), pad, mode="constant", constant_values=0)
            padded = find_contours(mp.astype(float), level=cfg.level)
            # Undo padding and clip back to valid window
            shifted: list[ArrayF] = []
            for c in padded:
                c2 = c - pad
                c2[:, 0] = np.clip(c2[:, 0], 0, H - 1)
                c2[:, 1] = np.clip(c2[:, 1], 0, W - 1)
                shifted.append(c2)
            # Filter frame-hugging contours with strict tolerance
            non_frame, _ = _filter_frame_contours(shifted, (H, W), tol=cfg.frame_tol)
            if len(non_frame) > 0:
                candidates_xy = [
                    rc_to_xy(np.asarray(c, dtype=np.float64)) for c in non_frame
                ]
                areas = np.array(
                    [abs(signed_area_2d(c)) for c in candidates_xy], dtype=np.float64
                )
                outer_xy = candidates_xy[int(np.argmax(areas))]
                if cfg.ensure_ccw and signed_area_2d(outer_xy) < 0.0:
                    outer_xy = outer_xy[::-1]
                return resample_closed_polyline(outer_xy, n=max(cfg.min_points, 3))
        # Option B: iterative erosion (if padding is disabled or fails)
        if cfg.erosion_last_resort:
            m_er = m.copy()
            for _ in range(max(1, int(cfg.erosion_iters))):
                m_er = binary_erosion(m_er, footprint=_fp_rect(3, 3))
                if not m_er.any():
                    break
            if m_er.any():
                m = m_er
            else:
                raise ValueError(
                    "All-ones mask and erosion removed everything; cannot extract a contour."
                )
        else:
            raise ValueError(
                "All-ones mask and padding disabled; enable pad_on_boundary or erosion_last_resort."
            )

    # --- STEP 1: direct extraction -------------------------------------
    contours_rc = find_contours(m.astype(float), level=cfg.level)
    non_frame, _ = _filter_frame_contours(contours_rc, (H, W), tol=cfg.frame_tol)

    # --- STEP 2: padding fallback if no candidates ----------------------
    if len(non_frame) == 0 and cfg.pad_on_boundary:
        pad = max(1, int(cfg.pad_width))
        mp = np.pad(m.astype(np.uint8), pad, mode="constant", constant_values=0)
        padded = find_contours(mp.astype(float), level=cfg.level)
        shifted: list[ArrayF] = []
        for c in padded:
            c2 = c - pad
            c2[:, 0] = np.clip(c2[:, 0], 0, H - 1)
            c2[:, 1] = np.clip(c2[:, 1], 0, W - 1)
            shifted.append(c2)
        non_frame, _ = _filter_frame_contours(shifted, (H, W), tol=cfg.frame_tol)

    # --- STEP 3: last-resort erosion -----------------------------------
    if len(non_frame) == 0 and cfg.erosion_last_resort:
        me = binary_erosion(m, footprint=_fp_rect(3, 3))
        if me.any():
            contours_rc = find_contours(me.astype(float), level=cfg.level)
            non_frame, _ = _filter_frame_contours(
                contours_rc, (H, W), tol=cfg.frame_tol
            )

    if len(non_frame) == 0:
        uniq = np.unique(m).tolist()
        raise ValueError(
            f"No usable (non-frame) contour found. "
            f"shape={m.shape}, sum={int(m.sum())}, unique={uniq}. "
            f"Try increasing internal erosion (erosion_iters), or ensure the object "
            f"does not fully cover the image."
        )

    # Final selection by area
    candidates_xy = [rc_to_xy(np.asarray(c, dtype=np.float64)) for c in non_frame]
    areas = np.array([abs(signed_area_2d(c)) for c in candidates_xy], dtype=np.float64)
    outer_xy = candidates_xy[int(np.argmax(areas))]

    if cfg.ensure_ccw and signed_area_2d(outer_xy) < 0.0:
        outer_xy = outer_xy[::-1]

    return resample_closed_polyline(outer_xy, n=max(cfg.min_points, 3))
