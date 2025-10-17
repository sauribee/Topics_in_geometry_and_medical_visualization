from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]

__all__ = [
    "BezierCurve",
    "de_casteljau",
    "chord_parameterization",
    "fit_cubic_bezier",
]


# ---------------------------------------------------------------------------
# Parameterization utilities
# ---------------------------------------------------------------------------


def chord_parameterization(
    points: ArrayF,
    *,
    alpha: float = 1.0,
    normalize: bool = True,
) -> ArrayF:
    """
    Compute cumulative chord-length-like parameterization for a polyline.

    Parameters
    ----------
    points : (N, d) float64
        Sample points along a curve.
    alpha : float, optional
        Exponent for generalized chord-length parameterization:
        - alpha=1.0 → chord-length
        - alpha=0.5 → centripetal
    normalize : bool, optional
        If True, remap to [0, 1].

    Returns
    -------
    u : (N,) float64
        Monotone parameters, typically in [0, 1].
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("points must be an array of shape (N, d), N >= 2")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if alpha != 1.0:
        seg = seg**alpha

    u = np.concatenate(([0.0], np.cumsum(seg)))
    if normalize and u[-1] > 0:
        u /= u[-1]
    return u


# ---------------------------------------------------------------------------
# De Casteljau evaluation
# ---------------------------------------------------------------------------


def de_casteljau(control_points: ArrayF, t: float) -> ArrayF:
    """
    Evaluate a Bézier curve at parameter t using De Casteljau.

    Parameters
    ----------
    control_points : (n+1, d) float64
        Control polygon P_0, ..., P_n.
    t : float
        Parameter in [0, 1].

    Returns
    -------
    p : (d,) float64
        Curve point B(t).
    """
    cps = np.asarray(control_points, dtype=np.float64)
    if cps.ndim != 2 or cps.shape[0] < 2:
        raise ValueError("control_points must have shape (n+1, d), n >= 1")
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be in [0, 1]")

    temp = cps.copy()
    n = temp.shape[0] - 1
    for r in range(1, n + 1):
        temp[:-r] = (1.0 - t) * temp[:-r] + t * temp[1 : (n - r + 2)]
    return temp[0]


# ---------------------------------------------------------------------------
# Bezier curve dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BezierCurve:
    """
    Generic Bézier curve in R^d.

    Attributes
    ----------
    control_points : (n+1, d) float64
        Control polygon.
    """

    control_points: ArrayF

    def __post_init__(self) -> None:
        cps = np.asarray(self.control_points, dtype=np.float64)
        if cps.ndim != 2 or cps.shape[0] < 2:
            raise ValueError("control_points must have shape (n+1, d), n >= 1")
        object.__setattr__(self, "control_points", cps)

    @property
    def degree(self) -> int:
        return self.control_points.shape[0] - 1

    @property
    def dim(self) -> int:
        return self.control_points.shape[1]

    def evaluate(self, t: float) -> ArrayF:
        """Evaluate B(t) via De Casteljau."""
        return de_casteljau(self.control_points, t)

    def evaluate_batch(self, t: ArrayF) -> ArrayF:
        """
        Evaluate B(t) for many parameters.

        Parameters
        ----------
        t : (M,) float64
            Parameter values in [0, 1].

        Returns
        -------
        pts : (M, d) float64
        """
        ts = np.asarray(t, dtype=np.float64).ravel()
        return np.vstack([self.evaluate(float(τ)) for τ in ts])

    # ---- derivatives

    def derivative(self) -> "BezierCurve":
        """
        First derivative curve B'(t) as a Bézier curve of degree n-1.

        Returns
        -------
        dcurve : BezierCurve
            Control points are n * (P_{i+1} - P_i).
        """
        cps = self.control_points
        n = self.degree
        d_cps = n * (cps[1:] - cps[:-1])
        return BezierCurve(d_cps)

    def second_derivative(self) -> "BezierCurve":
        """Second derivative curve B''(t)."""
        return self.derivative().derivative()

    def tangent(self, t: float) -> ArrayF:
        """Unit tangent at parameter t (zero vector if degenerate)."""
        d = self.derivative().evaluate(t)
        nrm = np.linalg.norm(d)
        return d / nrm if nrm > 0 else d

    def curvature_2d(self, t: float) -> float:
        """
        Curvature κ(t) for a planar curve.

        κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        """
        if self.dim != 2:
            raise ValueError("curvature_2d is defined for 2D curves only.")
        d1 = self.derivative().evaluate(t)
        d2 = self.second_derivative().evaluate(t)
        num = abs(d1[0] * d2[1] - d1[1] * d2[0])
        den = (d1[0] ** 2 + d1[1] ** 2) ** 1.5
        return (num / den) if den > 1e-15 else 0.0

    # ---- subdivision & length

    def subdivide(self, t: float) -> Tuple["BezierCurve", "BezierCurve"]:
        """
        Split the curve at t into left/right Bézier curves.

        Returns
        -------
        (left, right) : Tuple[BezierCurve, BezierCurve]
        """
        cps = self.control_points
        n = cps.shape[0] - 1
        tmp = cps.copy()
        left = np.empty_like(cps)
        right = np.empty_like(cps)
        left[0] = tmp[0]
        right[n] = tmp[n]
        for r in range(1, n + 1):
            tmp[: (n - r + 1)] = (1.0 - t) * tmp[: (n - r + 1)] + t * tmp[
                1 : (n - r + 2)
            ]
            left[r] = tmp[0]
            right[n - r] = tmp[n - r]
        return BezierCurve(left), BezierCurve(right)

    def to_polyline(self, samples: int = 200) -> ArrayF:
        """Uniform sampling of the curve (for preview or length)."""
        ts = np.linspace(0.0, 1.0, int(samples))
        return self.evaluate_batch(ts)

    def length(self, samples: int = 512) -> float:
        """Approximate arc length by trapezoidal rule on a fine polyline."""
        poly = self.to_polyline(samples=samples)
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        return float(np.sum(seg))


# ---------------------------------------------------------------------------
# Least-squares cubic Bézier fit to a polyline
# ---------------------------------------------------------------------------


def fit_cubic_bezier(
    points: ArrayF,
    *,
    params: Optional[ArrayF] = None,
    parameterization_alpha: float = 1.0,
    include_endpoints: bool = False,
) -> BezierCurve:
    """
    Fit a single **cubic Bézier** B(t) to a polyline by linear least squares.

    We solve for control points P1 and P2 directly, keeping P0=x[0], P3=x[-1].
    For samples x_i with parameters u_i in (0,1), we write:

        x_i ≈ b0(u_i) P0 + b1(u_i) P1 + b2(u_i) P2 + b3(u_i) P3,

    where b0=(1-u)^3, b1=3(1-u)^2 u, b2=3(1-u) u^2, b3=u^3.
    This is linear in P1 and P2, so we can solve independently per dimension.

    Parameters
    ----------
    points : (N, d) float64
        Data samples to approximate. N >= 4 recommended.
    params : (N,) float64, optional
        Parameter values in [0,1]. If None, uses chord-length-like with `parameterization_alpha`.
    parameterization_alpha : float, optional
        Exponent for chord parameterization (1.0 = chord-length, 0.5 = centripetal).
    include_endpoints : bool, optional
        If True, include the first and last samples in LS (usually unnecessary).
        By default we drop them because b1=b2=0 at u=0,1, which weakens the system.

    Returns
    -------
    curve : BezierCurve
        Fitted cubic Bézier.

    Notes
    -----
    - This is a single-span fit; continuity constraints for a piecewise fit
      will be implemented in a separate module (next steps in the roadmap).
    - If the normal equations are ill-conditioned (rare with interior samples),
      we regularize via lstsq; fallback is a heuristic choice for P1,P2.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 4:
        raise ValueError(
            "points must have shape (N, d) with N >= 4 for a robust cubic fit"
        )

    n, d = x.shape
    if params is None:
        u = chord_parameterization(x, alpha=parameterization_alpha, normalize=True)
    else:
        u = np.asarray(params, dtype=np.float64).ravel()
        if u.shape[0] != n:
            raise ValueError("params must have length N equal to points.shape[0]")
        if (u < 0).any() or (u > 1).any():
            raise ValueError("params must lie in [0, 1]")

    idx_start, idx_end = 0, n - 1
    P0, P3 = x[idx_start], x[idx_end]

    # Build design matrix only with interior samples to avoid zero rows (b1=b2=0 at ends).
    if include_endpoints:
        mask = slice(0, n)  # include all
    else:
        mask = slice(1, n - 1)  # interior only

    u_int = u[mask]
    X_int = x[mask]

    # Bernstein basis for cubic at u
    b0 = (1 - u_int) ** 3
    b1 = 3 * (1 - u_int) ** 2 * u_int
    b2 = 3 * (1 - u_int) * (u_int**2)
    b3 = u_int**3

    # A @ [P1; P2] ≈ Y  with A=(b1 b2), Y = X - (b0 P0 + b3 P3)
    A = np.column_stack([b1, b2])  # shape (M, 2)
    Y = X_int - (b0[:, None] * P0[None, :] + b3[:, None] * P3[None, :])  # shape (M, d)

    # Solve LS for each dimension at once using normal equations with lstsq fallback
    # (A^T A) (P1; P2) = A^T Y  →  2xd unknowns
    AT = A.T
    ATA = AT @ A  # 2x2
    ATY = AT @ Y  # 2xd
    try:
        sol = np.linalg.solve(ATA, ATY)  # 2xd
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
        # lstsq returns 2xd directly when Y is (M,d)

    P1 = sol[0, :]
    P2 = sol[1, :]

    cps = np.vstack([P0, P1, P2, P3])
    return BezierCurve(cps)
