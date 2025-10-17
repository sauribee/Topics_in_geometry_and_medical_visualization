from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

from .bezier_piecewise import PiecewiseBezier, fit_piecewise_cubic_bezier
from .contour2d import resample_closed_polyline

ArrayF = NDArray[np.float64]


__all__ = [
    "BezierPWConfig",
    "BSplineConfig",
    "FitMetrics",
    "BezierPWResult",
    "BSplineResult",
    "fit_contour_bezier_piecewise",
    "fit_contour_bspline_closed",
    "vertex_to_polyline_errors",
]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BezierPWConfig:
    """
    Configuration for piecewise cubic Bézier fitting.
    """

    max_error: float = 0.02
    parameterization_alpha: float = 0.5  # 0.5 = centripetal; 1.0 = chord-length
    c1_enforce: bool = True
    max_depth: int = 12
    curve_samples_error: int = 400  # discretization for error estimation
    sample_n: int = 200  # number of evenly-spaced output samples (closed)


@dataclass(frozen=True)
class BSplineConfig:
    """
    Configuration for closed B-spline fitting.
    """

    s: float = 0.0  # smoothing parameter (0.0 = interpolate)
    k: int = 3  # degree
    dense: int = 4000  # dense sampling for arc-length reparam
    sample_n: int = 200  # number of evenly-spaced output samples (closed)


# -----------------------------------------------------------------------------
# Results and metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FitMetrics:
    """
    Vertex-to-curve polyline distances (data → fitted curve polyline).
    """

    max_error: float
    mean_error: float


@dataclass(frozen=True)
class BezierPWResult:
    """
    Result for piecewise cubic Bézier fitting.
    """

    model: PiecewiseBezier
    samples_xy: ArrayF  # (N,2) evenly-spaced along the fitted curve (closed)
    metrics: FitMetrics  # distances data→curve polyline
    knots: ArrayF  # global knots in [0,1] for the piecewise model


@dataclass(frozen=True)
class BSplineResult:
    """
    Result for closed B-spline fitting.
    """

    tck: Tuple[ArrayF, ArrayF, int]  # SciPy (t, c, k)
    periodic: bool  # True (closed)
    samples_xy: ArrayF  # (N,2) evenly-spaced along the fitted spline (closed)
    metrics: FitMetrics  # distances data→curve polyline


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _ensure_closed(points_xy: ArrayF) -> ArrayF:
    """
    Ensure the input polygon is treated as closed (without duplicating last=first).
    """
    P = np.asarray(points_xy, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points_xy must have shape (N,2).")
    # We do not append P[0]; we treat closure implicitly.
    return P


def _polyline_length(P: ArrayF) -> float:
    """Euclidean polyline length including the closing edge (last→first)."""
    Q = np.vstack([P, P[0]])
    return float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))


def vertex_to_polyline_errors(data_xy: ArrayF, polyline_xy: ArrayF) -> FitMetrics:
    """
    Compute vertex-only distances from data points to a polyline (closed).

    Notes
    -----
    We use vertex-to-vertex distance to the polyline sampling; this is robust
    and fast for QA and adaptive procedures. If you need tighter bounds, a
    point-to-segment distance can be implemented later.
    """
    data = np.asarray(data_xy, dtype=np.float64)
    poly = np.asarray(polyline_xy, dtype=np.float64)
    diffs = data[:, None, :] - poly[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
    dmin = np.sqrt(d2.min(axis=1))
    return FitMetrics(max_error=float(dmin.max()), mean_error=float(dmin.mean()))


# -----------------------------------------------------------------------------
# Bézier piecewise fitting
# -----------------------------------------------------------------------------


def _piecewise_to_dense_polyline(
    model: PiecewiseBezier, samples_per_seg: int = 200
) -> ArrayF:
    """
    Dense polyline sampling of a piecewise Bézier (closed expected downstream).
    """
    return model.to_polyline(samples_per_segment=samples_per_seg)


def _piecewise_evenly_spaced(model: PiecewiseBezier, n: int) -> ArrayF:
    """
    Evenly spaced sampling along a piecewise Bézier by resampling a dense polyline.
    """
    dense = _piecewise_to_dense_polyline(model, samples_per_seg=max(200, n // 2))
    return resample_closed_polyline(dense, n=n)


def fit_contour_bezier_piecewise(
    contour_xy: ArrayF,
    cfg: Optional[BezierPWConfig] = None,
) -> BezierPWResult:
    """
    Fit a closed contour with a piecewise cubic Bézier and return evenly-spaced samples.

    Parameters
    ----------
    contour_xy : (N,2) float64
        Closed polygon in physical coordinates.
    cfg : BezierPWConfig, optional
        Fitting and sampling configuration.

    Returns
    -------
    BezierPWResult
        Piecewise model, evenly-spaced samples, and vertex→curve metrics.
    """
    C = _ensure_closed(contour_xy)
    conf = cfg or BezierPWConfig()

    # Fit model
    pw = fit_piecewise_cubic_bezier(
        C,
        max_error=conf.max_error,
        parameterization_alpha=conf.parameterization_alpha,
        c1_enforce=conf.c1_enforce,
        max_depth=conf.max_depth,
        curve_samples_error=conf.curve_samples_error,
    )

    # Evenly spaced polyline along the fitted curve
    samples = _piecewise_evenly_spaced(pw, n=conf.sample_n)

    # Dense polyline for metrics (reuse the dense used internally)
    dense = _piecewise_to_dense_polyline(pw, samples_per_seg=max(400, conf.sample_n))
    metrics = vertex_to_polyline_errors(C, dense)

    return BezierPWResult(model=pw, samples_xy=samples, metrics=metrics, knots=pw.knots)


# -----------------------------------------------------------------------------
# Closed B-spline fitting (periodic)
# -----------------------------------------------------------------------------


def _fit_closed_bspline(
    contour_xy: ArrayF,
    *,
    s: float = 0.0,
    k: int = 3,
) -> Tuple[ArrayF, ArrayF, int]:
    """
    Fit a **periodic** (closed) B-spline to a closed polygon using SciPy's splprep.

    Returns
    -------
    tck : (t, c, k)
        Standard SciPy representation.
    """
    P = _ensure_closed(contour_xy)
    # Prepare inputs for splprep: x, y as 1D arrays
    x, y = P[:, 0], P[:, 1]
    # per=True enforces periodic closure
    tck, _ = splprep([x, y], s=float(s), k=int(k), per=True)
    return tck


def _bspline_dense_polyline(tck: Tuple[ArrayF, ArrayF, int], M: int) -> ArrayF:
    """
    Dense uniform parameter sampling in [0,1] and evaluation.
    """
    t = np.linspace(0.0, 1.0, int(M), endpoint=False)
    x, y = splev(t, tck)
    return np.column_stack([x, y])


def _bspline_evenly_spaced(
    tck: Tuple[ArrayF, ArrayF, int],
    *,
    dense: int,
    n: int,
) -> ArrayF:
    """
    Evenly spaced sampling along a closed B-spline via arc-length reparameterization.

    Strategy
    --------
    1) Sample the spline densely at uniform parameter t.
    2) Compute cumulative arc-length along that dense polyline.
    3) Resample evenly in arc-length using linear interpolation + closure.
    """
    dense_poly = _bspline_dense_polyline(tck, M=max(dense, 10 * n))
    return resample_closed_polyline(dense_poly, n=n)


def fit_contour_bspline_closed(
    contour_xy: ArrayF,
    cfg: Optional[BSplineConfig] = None,
) -> BSplineResult:
    """
    Fit a closed contour with a closed (periodic) B-spline and return evenly-spaced samples.

    Parameters
    ----------
    contour_xy : (N,2) float64
        Closed polygon in physical coordinates.
    cfg : BSplineConfig, optional
        Spline and sampling configuration.

    Returns
    -------
    BSplineResult
        tck, evenly-spaced samples, and vertex→curve metrics.
    """
    C = _ensure_closed(contour_xy)
    conf = cfg or BSplineConfig()

    tck = _fit_closed_bspline(C, s=conf.s, k=conf.k)
    samples = _bspline_evenly_spaced(tck, dense=conf.dense, n=conf.sample_n)

    dense = _bspline_dense_polyline(tck, M=max(conf.dense, 10 * conf.sample_n))
    metrics = vertex_to_polyline_errors(C, dense)

    return BSplineResult(tck=tck, periodic=True, samples_xy=samples, metrics=metrics)
