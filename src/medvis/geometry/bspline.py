from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

from .bezier import chord_parameterization

ArrayF = NDArray[np.float64]

__all__ = [
    "averaged_open_uniform_knots",
    "bspline_basis_vector",
    "bspline_basis_matrix",
    "fit_bspline_interpolate",
    "evaluate_bspline",
]


def averaged_open_uniform_knots(u: ArrayF, degree: int) -> ArrayF:
    uu = np.asarray(u, dtype=np.float64).ravel()
    N = uu.shape[0]
    p = int(degree)
    n_ctrl = N
    m = n_ctrl + p
    knots = np.empty(m + 1, dtype=np.float64)
    knots[: p + 1] = 0.0
    knots[m - p : m + 1] = 1.0
    count = n_ctrl - p - 1
    for j in range(1, count + 1):
        knots[p + j] = float(np.mean(uu[j : j + p]))
    return knots


def bspline_basis_vector(u: float, degree: int, knots: ArrayF, n_ctrl: int) -> ArrayF:
    p = int(degree)
    t = np.asarray(knots, dtype=np.float64)
    n = int(n_ctrl)
    N = np.zeros(n, dtype=np.float64)
    last = t[-1]
    for i in range(n):
        if (t[i] <= u < t[i + 1]) or (u == last and i == n - 1):
            N[i] = 1.0
    for k in range(1, p + 1):
        Nk = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d1 = t[i + k] - t[i]
            d2 = t[i + k + 1] - t[i + 1] if i + 1 < n else 0.0
            a = 0.0 if d1 <= 0.0 else (u - t[i]) / d1 * N[i]
            b = 0.0 if d2 <= 0.0 or i + 1 >= n else (t[i + k + 1] - u) / d2 * N[i + 1]
            Nk[i] = a + b
        N = Nk
    return N


def bspline_basis_matrix(u: ArrayF, degree: int, knots: ArrayF, n_ctrl: int) -> ArrayF:
    uu = np.asarray(u, dtype=np.float64).ravel()
    A = np.zeros((uu.shape[0], int(n_ctrl)), dtype=np.float64)
    for j, uj in enumerate(uu):
        A[j, :] = bspline_basis_vector(float(uj), degree, knots, n_ctrl)
    return A


def fit_bspline_interpolate(
    points: ArrayF,
    *,
    degree: int = 3,
    params: Optional[ArrayF] = None,
) -> Tuple[ArrayF, ArrayF, int]:
    X = np.asarray(points, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("points must have shape (N,d), N>=2")
    n_ctrl = X.shape[0]
    p = int(degree)
    if p < 1 or n_ctrl <= p:
        raise ValueError("degree must be >=1 and < number of points")
    if params is None:
        u = chord_parameterization(X, alpha=1.0, normalize=True)
    else:
        u = np.asarray(params, dtype=np.float64).ravel()
    knots = averaged_open_uniform_knots(u, p)
    A = bspline_basis_matrix(u, p, knots, n_ctrl)
    C, *_ = np.linalg.lstsq(A, X, rcond=None)
    return knots, C, p


def evaluate_bspline(knots: ArrayF, C: ArrayF, degree: int, t: ArrayF) -> ArrayF:
    tt = np.asarray(t, dtype=np.float64).ravel()
    A = bspline_basis_matrix(tt, degree, knots, C.shape[0])
    return A @ C
