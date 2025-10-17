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
    Minimum Euclidean distance from point p to a polyline q (vertex-only).
    Robust and fast enough for adaptive splitting.
    """
    dif = q - p[None, :]
    d2 = np.einsum("ij,ij->i", dif, dif)
    return float(np.sqrt(d2.min()))


def _max_data_to_curve_error(
    data: ArrayF,
    curve: BezierCurve,
    samples: int = 400,
) -> Tuple[float, int]:
    """
    Maximum vertex-only distance from data to a sampled polyline of 'curve'.
    Returns (max_err, index_in_data).
    """
    poly = curve.to_polyline(samples=samples)
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
    Enforce a practical C¹-like condition: align outgoing/incoming tangents and
    use a common magnitude at joints. Keeps C⁰ by construction.
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

        if nL < 1e-12 and nR < 1e-12:
            new_segments[-1] = BezierCurve(np.vstack([PL0, PL1, PL2, PL3]))
            new_segments.append(BezierCurve(np.vstack([PR0, PR1, PR2, PR3])))
            continue

        if nL < 1e-12:
            v = tR / nR
        elif nR < 1e-12:
            v = tL / nL
        else:
            v = tL / nL + tR / nR
            nv = np.linalg.norm(v)
            v = v / nv if nv > 0 else tL / nL

        L = 0.5 * (nL + nR)  # common magnitude
        PL2_new = J - L * v
        PR1_new = J + L * v

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
    curve_samples_error: int = 400,
) -> PiecewiseBezier:
    """
    Fit a piecewise cubic Bézier curve to a polyline by recursive splitting.

    Robustness improvements:
    - Avoid splitting if both halves cannot have N>=4.
    - Provide fallback cubic constructors for short data (N=2 or N=3).
    - For N=4, fit with endpoints included; fallback to straight/heuristic if ill-conditioned.
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
        seg = fit_cubic_bezier(
            data,
            parameterization_alpha=parameterization_alpha,
            include_endpoints=False,
        )
        err, idx = _max_data_to_curve_error(data, seg, samples=curve_samples_error)
        if err <= max_error or depth >= max_depth:
            return [seg]

        # If we cannot split into two parts with >= 4 points each, accept as leaf
        if n < 8:
            return [seg]

        # Choose split index ensuring both halves have >=4 points
        i_split = int(
            np.clip(idx, 3, n - 4)
        )  # left has [0..i_split], right has [i_split..]
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

    # Knots by approximate arc-length
    seg_lengths = np.array([s.length(samples=400) for s in segments], dtype=np.float64)
    total = float(seg_lengths.sum())
    if total <= 0.0:
        knots = np.linspace(0.0, 1.0, len(segments) + 1)
    else:
        cum = np.concatenate(([0.0], np.cumsum(seg_lengths / total)))
        cum[-1] = 1.0
        knots = cum

    return PiecewiseBezier(segments=segments, knots=knots)
