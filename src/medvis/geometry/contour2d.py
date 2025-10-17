from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_holes

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
    """

    level: float = 0.5
    fill_holes: bool = True
    simplify_tol: Optional[float] = None  # placeholder (see docstring)
    min_points: int = 256
    ensure_ccw: bool = True
    spacing_xy: Tuple[float, float] = (1.0, 1.0)
    origin_xy: Tuple[float, float] = (0.0, 0.0)

    # NEW: fallback when the object touches the image border
    pad_on_boundary: bool = True
    pad_width: int = 1


# ---------------------------------------------------------------------------
# Main contour extraction
# ---------------------------------------------------------------------------


def extract_primary_contour(
    mask: np.ndarray,
    *,
    config: Optional[ContourExtractionConfig] = None,
) -> ArrayF:
    """
    Extract the primary (outer) closed contour from a 2D binary mask and return
    an evenly resampled polygon in **physical (x,y)** coordinates.

    Steps
    -----
    1) Binarize the input (`> 0.5` if numeric). Optionally fill holes.
    2) Keep the largest connected component.
    3) `find_contours` at level=0.5 to extract candidate polylines in (row, col).
    4) Choose the outer boundary (largest absolute area).
    5) Convert to physical coordinates using (sx, sy) and (ox, oy).
    6) Enforce orientation (CCW) if requested.
    7) Evenly resample to `min_points` vertices.

    Parameters
    ----------
    mask : (H, W) array-like
        2D binary or numeric mask.
    config : ContourExtractionConfig, optional
        Extraction parameters (see dataclass).

    Returns
    -------
    contour_xy : (N, 2) float64
        Evenly resampled CCW polygon in physical coordinates (x, y),
        with N = config.min_points (closed implicitly).

    Raises
    ------
    ValueError
        If the mask has no foreground or no contour can be found.
    """
    cfg = config or ContourExtractionConfig()

    # 1) Binarize
    m = _to_bool_mask(mask)

    # 2) Optional hole filling (use very large area threshold)
    if cfg.fill_holes:
        # Fill any internal holes by setting a very large threshold
        # (safe for medical masks when you expect a solid bone region).
        m = remove_small_holes(m, area_threshold=int(m.size), connectivity=2)

    # 3) Largest CC
    m = _largest_connected_component(m, connectivity=2)
    if not m.any():
        raise ValueError("Empty mask after preprocessing (no foreground).")

    # 4) Find candidate contours at the binary boundary level
    contours_rc = find_contours(m.astype(float), level=cfg.level)

    if len(contours_rc) == 0 and cfg.pad_on_boundary:
        mp = np.pad(
            m.astype(np.uint8), cfg.pad_width, mode="constant", constant_values=0
        )
        padded = find_contours(mp.astype(float), level=cfg.level)
        contours_rc = [c - cfg.pad_width for c in padded]

    if len(contours_rc) == 0:
        uniq = np.unique(m).tolist()
        raise ValueError(
            f"No contour found at level={cfg.level}. "
            f"Check mask: shape={m.shape}, sum={int(m.sum())}, unique={uniq}. "
            f"Tip: if your object is all-ones or all-zeros after preprocessing, "
            f"there is no 0↔1 transition to contour."
        )

    # 5) Choose outer boundary by absolute area in (x,y) after mapping
    sx, sy = cfg.spacing_xy
    ox, oy = cfg.origin_xy

    def rc_to_xy(cont: ArrayF) -> ArrayF:
        # skimage returns (row, col). We map to (x, y):
        #   x = ox + col * sx
        #   y = oy + row * sy
        col = cont[:, 1]
        row = cont[:, 0]
        x = ox + col * sx
        y = oy + row * sy
        return np.column_stack([x, y])

    candidates_xy = [rc_to_xy(np.asarray(c, dtype=np.float64)) for c in contours_rc]
    areas = np.array([abs(signed_area_2d(c)) for c in candidates_xy], dtype=np.float64)
    if areas.size == 0 or np.all(areas <= 0.0):
        raise ValueError("Contours found but degenerate (zero area).")
    idx = int(np.argmax(areas))
    outer_xy = candidates_xy[idx]

    # 6) Enforce orientation if requested
    if cfg.ensure_ccw and signed_area_2d(outer_xy) < 0.0:
        outer_xy = outer_xy[::-1]

    # 7) Evenly resample to min_points
    N = int(max(cfg.min_points, 3))
    contour_xy = resample_closed_polyline(outer_xy, n=N)
    return contour_xy
