from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .bezier import BezierCurve, fit_cubic_bezier

ArrayF = NDArray[np.float64]

__all__ = [
    "PiecewiseBezier",
    "fit_piecewise_cubic_bezier",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _polyline_distance(p: ArrayF, q: ArrayF) -> float:
    """
    Minimum Euclidean distance from point p to a polyline q.
    Uses point-to-segment distance (not just vertices) for accuracy.
    """
    if q.shape[0] < 2:
        return float(np.linalg.norm(p - q[0]))

    min_dist = np.inf

    # Check distance to each segment
    for i in range(q.shape[0] - 1):
        a = q[i]
        b = q[i + 1]

        # Vector from a to b
        ab = b - a
        ab_len_sq = np.dot(ab, ab)

        if ab_len_sq < 1e-20:  # Degenerate segment
            dist = np.linalg.norm(p - a)
        else:
            # Project p onto line ab
            ap = p - a
            t = np.dot(ap, ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)  # Clamp to segment

            # Closest point on segment
            closest = a + t * ab
            dist = np.linalg.norm(p - closest)

        min_dist = min(min_dist, dist)

    return float(min_dist)


def _max_data_to_curve_error(
    data: ArrayF,
    curve: BezierCurve,
    samples: int = 1000,
) -> Tuple[float, int]:
    """
    Maximum distance from data points to curve (sampled as polyline).
    Returns (max_err, index_in_data).
    Uses denser sampling (1000) for accuracy.
    """
    try:
        poly = curve.to_polyline(samples=samples)
    except Exception:
        # Fallback for degenerate curves
        return float("inf"), 0

    if poly.shape[0] < 2:
        # Degenerate curve
        return float("inf"), 0

    max_err = -1.0
    argmax = 0

    for i in range(data.shape[0]):
        e = _polyline_distance(data[i], poly)
        if e > max_err:
            max_err = e
            argmax = i

    return max_err, argmax


def _enforce_c1_symmetric(segments: List[BezierCurve]) -> List[BezierCurve]:
    """
    Enforce practical C¹-like condition: align tangents at joints.
    Uses geometric mean for magnitude to prevent overshoots.
    Includes overshoot prevention and stability checks.
    """
    if len(segments) <= 1:
        return segments

    new_segments: List[BezierCurve] = [segments[0]]

    for i in range(len(segments) - 1):
        sL = new_segments[-1]
        sR = segments[i + 1]

        PL0, PL1, PL2, PL3 = sL.control_points
        PR0, PR1, PR2, PR3 = sR.control_points

        # Shared joint J (harden C0)
        J = 0.5 * (PL3 + PR0)
        PL3 = J
        PR0 = J

        # Tangents (before enforcement)
        tL = PL3 - PL2
        tR = PR1 - PR0
        nL = np.linalg.norm(tL)
        nR = np.linalg.norm(tR)

        # Degenerate cases
        if nL < 1e-12 and nR < 1e-12:
            new_segments[-1] = BezierCurve(np.vstack([PL0, PL1, PL2, PL3]))
            new_segments.append(BezierCurve(np.vstack([PR0, PR1, PR2, PR3])))
            continue

        if nL < 1e-12:
            v = tR / nR
            L = nR
        elif nR < 1e-12:
            v = tL / nL
            L = nL
        else:
            # Average direction (weighted by magnitude for stability)
            v = (nL * tL / nL + nR * tR / nR) / (nL + nR)
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-12 else tL / nL

            # Use geometric mean to prevent overshoots (more stable than arithmetic mean)
            L = np.sqrt(nL * nR)

            # Clamp magnitude to prevent extreme overshoots
            max_magnitude = 1.5 * max(nL, nR)
            min_magnitude = 0.5 * min(nL, nR)
            L = np.clip(L, min_magnitude, max_magnitude)

        # Compute new control points
        PL2_new = J - L * v
        PR1_new = J + L * v

        # Safety check: ensure control points don't create loops
        # Check that new control points stay on the correct side of the joint
        seg_L_length = np.linalg.norm(PL3 - PL0)
        seg_R_length = np.linalg.norm(PR3 - PR0)

        if seg_L_length > 1e-12:
            max_offset_L = 0.4 * seg_L_length
            if np.linalg.norm(PL2_new - PL3) > max_offset_L:
                PL2_new = PL3 - (PL2_new - PL3) * max_offset_L / np.linalg.norm(
                    PL2_new - PL3
                )

        if seg_R_length > 1e-12:
            max_offset_R = 0.4 * seg_R_length
            if np.linalg.norm(PR1_new - PR0) > max_offset_R:
                PR1_new = PR0 + (PR1_new - PR0) * max_offset_R / np.linalg.norm(
                    PR1_new - PR0
                )

        new_segments[-1] = BezierCurve(np.vstack([PL0, PL1, PL2_new, PL3]))
        new_segments.append(BezierCurve(np.vstack([PR0, PR1_new, PR2, PR3])))

    return new_segments


# ---------------------------------------------------------------------------
# Fallback cubics for short data (N < 4)
# ---------------------------------------------------------------------------


def _cubic_from_two_points(P0: ArrayF, P3: ArrayF) -> BezierCurve:
    """Straight segment as a cubic: P1=P0+1/3*chord, P2=P0+2/3*chord."""
    chord = P3 - P0
    P1 = P0 + chord / 3.0
    P2 = P0 + 2.0 * chord / 3.0
    return BezierCurve(np.vstack([P0, P1, P2, P3]))


def _cubic_from_three_points(P0: ArrayF, P1: ArrayF, P3: ArrayF) -> BezierCurve:
    """
    Heuristic Hermite-like cubic for three samples:
    tangents from finite differences at ends.
    """
    t0 = P1 - P0
    t1 = P3 - P1
    C1 = P0 + t0 / 3.0
    C2 = P3 - t1 / 3.0
    return BezierCurve(np.vstack([P0, C1, C2, P3]))


# ---------------------------------------------------------------------------
# Piecewise representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PiecewiseBezier:
    """
    Piecewise cubic Bézier curve B : [0, 1] → R^d.

    Attributes
    ----------
    segments : list[BezierCurve]
        Cubic segments in order; C0 continuity is guaranteed at joints.
        Optional C1-like continuity may be enforced by construction.
    knots : (K+1,) float64
        Normalized cumulative parameter for each segment boundary,
        with knots[0]=0 and knots[-1]=1.
    """

    segments: List[BezierCurve]
    knots: ArrayF

    def __post_init__(self) -> None:
        if len(self.segments) == 0:
            raise ValueError("PiecewiseBezier requires at least one segment.")
        if self.knots.shape[0] != len(self.segments) + 1:
            raise ValueError("knots must have length len(segments)+1.")
        if self.knots[0] != 0.0 or self.knots[-1] != 1.0:
            raise ValueError("knots must start at 0.0 and end at 1.0.")
        if not np.all(np.diff(self.knots) > 0):
            raise ValueError("knots must be strictly increasing.")

    @property
    def dim(self) -> int:
        return self.segments[0].dim

    def evaluate(self, u: float) -> ArrayF:
        """Evaluate at global parameter u ∈ [0, 1] using piecewise mapping."""
        if u <= 0.0:
            return self.segments[0].evaluate(0.0)
        if u >= 1.0:
            return self.segments[-1].evaluate(1.0)
        k = int(np.searchsorted(self.knots, u, side="right") - 1)
        u0, u1 = self.knots[k], self.knots[k + 1]
        t = (u - u0) / (u1 - u0)
        return self.segments[k].evaluate(float(t))

    def evaluate_batch(self, u: ArrayF) -> ArrayF:
        uu = np.asarray(u, dtype=np.float64).ravel()
        return np.vstack([self.evaluate(float(val)) for val in uu])

    def to_polyline(self, samples_per_segment: int = 100) -> ArrayF:
        """Uniform sampling per segment (for preview)."""
        pts = [seg.to_polyline(samples=samples_per_segment) for seg in self.segments]
        return np.vstack(pts)

    def length(self, samples_per_segment: int = 200) -> float:
        return float(
            sum(seg.length(samples=samples_per_segment) for seg in self.segments)
        )


# ---------------------------------------------------------------------------
# Main fitter (recursive splitting with robust base cases)
# ---------------------------------------------------------------------------


def fit_piecewise_cubic_bezier(
    points: ArrayF,
    *,
    max_error: float = 1.0,
    parameterization_alpha: float = 0.5,
    c1_enforce: bool = True,
    max_depth: int = 12,
    curve_samples_error: int = 1000,
    min_segment_points: int = 4,
) -> PiecewiseBezier:
    """
    Fit a piecewise cubic Bézier curve to a polyline by recursive splitting.

    Parameters
    ----------
    points : (N, d) array
        Input points to fit
    max_error : float, default=1.0
        Maximum allowed fitting error (pixels or units)
    parameterization_alpha : float, default=0.5
        Centripetal parameterization (0.5 recommended for stability)
    c1_enforce : bool, default=True
        Enforce C1 continuity at joints
    max_depth : int, default=12
        Maximum recursion depth
    curve_samples_error : int, default=1000
        Samples for error computation (increased for accuracy)
    min_segment_points : int, default=4
        Minimum points per segment before forcing leaf

    Returns
    -------
    PiecewiseBezier
        Fitted piecewise curve

    Notes
    -----
    Robustness improvements:
    - Point-to-segment distance (not vertex-only)
    - Geometric mean for C1 magnitude (prevents overshoots)
    - Overshoot clamping (max 40% of segment length)
    - Denser sampling (1000) for error computation
    - Better degenerate case handling
    - Improved splitting strategy with balance checking
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("points must be an array of shape (N, d), with N >= 2")

    def _fit(data: ArrayF, depth: int) -> List[BezierCurve]:
        n = data.shape[0]

        # Robust base cases
        if n <= 2:
            return [_cubic_from_two_points(data[0], data[-1])]
        if n == 3:
            return [_cubic_from_three_points(data[0], data[1], data[2])]
        if n == 4:
            try:
                seg4 = fit_cubic_bezier(
                    data,
                    parameterization_alpha=parameterization_alpha,
                    include_endpoints=True,
                )
                err4, _ = _max_data_to_curve_error(
                    data, seg4, samples=curve_samples_error
                )
                return [seg4]
            except Exception:
                return [_cubic_from_three_points(data[0], data[1], data[3])]

        # Try single cubic on this block
        try:
            seg = fit_cubic_bezier(
                data,
                parameterization_alpha=parameterization_alpha,
                include_endpoints=False,
            )

            # Check for degenerate curve (all control points too close)
            cp = seg.control_points
            if np.max(np.linalg.norm(np.diff(cp, axis=0), axis=1)) < 1e-10:
                # Degenerate curve, use fallback
                return [_cubic_from_two_points(data[0], data[-1])]

            err, idx = _max_data_to_curve_error(data, seg, samples=curve_samples_error)

            # Accept if error is acceptable or max depth reached
            if err <= max_error or depth >= max_depth:
                return [seg]
        except Exception:
            # Fitting failed, use fallback
            return [_cubic_from_two_points(data[0], data[-1])]

        # If we cannot split into two balanced parts with >= min_segment_points each, accept as leaf
        if n < 2 * min_segment_points:
            return [seg]

        # Choose split index ensuring both halves have >= min_segment_points
        # Prefer balanced split but respect error maximum location
        mid = n // 2
        i_split = int(np.clip(idx, min_segment_points, n - min_segment_points))

        # If error-based split is too unbalanced, use midpoint
        if abs(i_split - mid) > n // 3:
            i_split = mid

        # Recursively fit both halves
        left = _fit(data[: i_split + 1], depth + 1)
        right = _fit(data[i_split:], depth + 1)
        return left + right

    segments = _fit(x, depth=0)

    # Harden C0 at joints
    for k in range(len(segments) - 1):
        P3 = segments[k].control_points[-1]
        P0 = segments[k + 1].control_points[0]
        J = 0.5 * (P3 + P0)
        cpL = segments[k].control_points.copy()
        cpR = segments[k + 1].control_points.copy()
        cpL[-1] = J
        cpR[0] = J
        segments[k] = BezierCurve(cpL)
        segments[k + 1] = BezierCurve(cpR)

    # Optional C1-like enforcement
    if c1_enforce and len(segments) > 1:
        segments = _enforce_c1_symmetric(segments)

    # Knots by approximate arc-length (increased sampling for accuracy)
    try:
        seg_lengths = np.array(
            [s.length(samples=800) for s in segments], dtype=np.float64
        )
        total = float(seg_lengths.sum())

        if total <= 1e-12:
            # Degenerate case: uniform knots
            knots = np.linspace(0.0, 1.0, len(segments) + 1)
        else:
            cum = np.concatenate(([0.0], np.cumsum(seg_lengths / total)))
            cum[-1] = 1.0  # Ensure exact 1.0 at end
            knots = cum
    except Exception:
        # Fallback to uniform knots if length computation fails
        knots = np.linspace(0.0, 1.0, len(segments) + 1)

    return PiecewiseBezier(segments=segments, knots=knots)
