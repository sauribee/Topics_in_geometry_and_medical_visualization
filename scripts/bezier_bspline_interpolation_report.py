from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from medvis.geometry.bezier import (
    BezierCurve,
    chord_parameterization,
    fit_bezier_interpolate,
    fit_bezier_lsq,
)
from medvis.geometry.bspline import (
    fit_bspline_interpolate,
    evaluate_bspline,
)


def ellipse_segment_points(
    n: int,
    *,
    center: Tuple[float, float] = (0.0, 0.0),
    a: float = 40.0,
    b: float = 25.0,
    angle_deg: float = 25.0,
    theta_start_deg: float = -60.0,
    theta_end_deg: float = 60.0,
    noise_std: float = 0.2,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    th0 = np.deg2rad(theta_start_deg)
    th1 = np.deg2rad(theta_end_deg)
    t = np.linspace(th0, th1, int(n))
    E = np.column_stack([a * np.cos(t), b * np.sin(t)])
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=float)
    P = E @ R.T + np.array(center, float)[None, :]
    if noise_std > 0:
        P = P + rng.normal(scale=float(noise_std), size=P.shape)
    return P.astype(float)


def plot_interpolation(
    pts: np.ndarray,
    bz_curve: BezierCurve,
    bs_knots: np.ndarray,
    bs_ctrl: np.ndarray,
    bs_degree: int,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    tt = np.linspace(0.0, 1.0, 800)
    B = bz_curve.evaluate_batch(tt)
    S = evaluate_bspline(bs_knots, bs_ctrl, bs_degree, tt)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.2))
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=5, label="data points")

    ax.plot(
        B[:, 0], B[:, 1], "-", lw=2.0, label=f"Bézier interp. (deg {bz_curve.degree})"
    )
    ax.plot(S[:, 0], S[:, 1], "-", lw=2.0, label=f"B-spline interp. (p={bs_degree})")
    ax.plot(
        bz_curve.control_points[:, 0],
        bz_curve.control_points[:, 1],
        "--o",
        ms=4,
        alpha=0.5,
        label="Bezier control poly",
    )
    ax.plot(
        bs_ctrl[:, 0],
        bs_ctrl[:, 1],
        "--o",
        ms=4,
        alpha=0.5,
        label="B-spline control poly",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Interpolation: Bézier vs Open B-spline")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_lsq(
    pts: np.ndarray,
    bz_lsq: BezierCurve,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tt = np.linspace(0.0, 1.0, 800)
    B = bz_lsq.evaluate_batch(tt)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.2))
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=5, label="data points")
    ax.plot(B[:, 0], B[:, 1], "-", lw=2.0, label=f"Bézier LSQ (deg {bz_lsq.degree})")
    ax.plot(
        bz_lsq.control_points[:, 0],
        bz_lsq.control_points[:, 1],
        "--o",
        ms=4,
        alpha=0.6,
        label="Bezier control poly",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Least-squares Bézier approximation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Interpolación con Bézier (grado N-1) y B-spline abierta (p=3) a partir de ~10 puntos de un segmento de elipse, "
            "y aproximación por mínimos cuadrados con Bézier de menor grado."
        )
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/bezier_bspline_interpolation"),
    )
    ap.add_argument("--n", type=int, default=10, help="Número de puntos de datos")
    ap.add_argument("--noise", type=float, default=0.2, help="Ruido gaussiano (std)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--lsq-degree",
        type=int,
        default=5,
        help="Grado para la Bézier de mínimos cuadrados",
    )
    args = ap.parse_args()

    pts = ellipse_segment_points(
        args.n, noise_std=float(args.noise), seed=int(args.seed)
    )

    # Parameterization for interpolation (chord-length in [0,1])
    u = chord_parameterization(pts, alpha=1.0, normalize=True)

    # 1) Bézier interpolation (degree N-1)
    bz_interp = fit_bezier_interpolate(pts, params=u)

    # 2) Open B-spline interpolation (p=3)
    knots, C, p = fit_bspline_interpolate(pts, degree=3, params=u)

    # 3) Optional: Bézier least-squares of lower degree
    deg_lsq = int(args.lsq_degree)
    bz_lsq = fit_bezier_lsq(pts, degree=deg_lsq, params=u)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_interpolation(pts, bz_interp, knots, C, p, out / "interpolation_overlay.png")
    plot_lsq(pts, bz_lsq, out / "bezier_lsq_overlay.png")

    np.savetxt(
        out / "points.csv", pts, delimiter=",", header="x,y", comments="", fmt="%.9f"
    )
    np.savetxt(
        out / "bezier_interpolation_control.csv",
        bz_interp.control_points,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.9f",
    )
    np.savetxt(
        out / "bspline_interpolation_control.csv",
        C,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.9f",
    )
    np.savetxt(
        out / "params_u.csv", u, delimiter=",", header="u", comments="", fmt="%.9f"
    )

    print(f"Report written to: {out.resolve()}")


if __name__ == "__main__":
    main()
