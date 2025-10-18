from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

# Non-interactive backend for headless/CI environments
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# medvis geometry utilities (already in your repo)
from medvis.geometry.contour2d import resample_closed_polyline
from medvis.geometry.contour_fit import (
    BezierPWConfig,
    BSplineConfig,
    fit_contour_bezier_piecewise,
    fit_contour_bspline_closed,
)

ArrayF = NDArray[np.float64]


# --------------------------------------------------------------------------------------
# Synthetic shape generators (closed point-clouds in physical coordinates, mm)
# --------------------------------------------------------------------------------------


def _rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass(frozen=True)
class ShapeParams:
    """Parameters controlling the base ellipse and deformations."""

    center: Tuple[float, float] = (0.0, 0.0)  # mm
    a: float = 40.0  # semi-major (mm)
    b: float = 25.0  # semi-minor (mm)
    angle_deg: float = 25.0  # rotation of the ellipse (deg)
    noise_std: float = 0.3  # additive Gaussian noise (mm)

    # Harmonic radial perturbations on the unit circle (then scaled/rotated):
    # r(t) = 1 + sum_j eps_j * cos(k_j * t + phi_j)
    harmonics: Tuple[Tuple[int, float, float], ...] = tuple()

    # Optional “notch” (local gaussian reduction on radius around t0)
    notch_enable: bool = False
    notch_center_deg: float = 20.0
    notch_sigma_deg: float = 8.0
    notch_depth: float = 0.06  # fraction of radius to remove


def _unit_radius(t: np.ndarray, params: ShapeParams) -> np.ndarray:
    """Unit-circle radius with harmonic perturbations and optional notch."""
    r = np.ones_like(t, dtype=np.float64)
    # Harmonics
    for k, eps, phi_deg in params.harmonics:
        r += eps * np.cos(k * t + np.deg2rad(phi_deg))
    # Notch
    if params.notch_enable:
        dt = t - np.deg2rad(params.notch_center_deg)
        g = np.exp(-0.5 * (dt / np.deg2rad(params.notch_sigma_deg)) ** 2)
        r *= 1.0 - params.notch_depth * g
    return r


def generate_deformed_ellipse(
    n_points: int,
    params: ShapeParams,
    rng: np.random.RandomState,
) -> ArrayF:
    """
    Generate a closed point-cloud resembling a deformed, rotated ellipse.

    Steps
    -----
    1) Start from the unit circle with deformations in radius.
    2) Apply anisotropic scaling (a, b) to make it elliptical.
    3) Apply in-plane rotation and translation.
    4) Add small Gaussian noise to emulate digitization / measurement noise.
    """
    t = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    r = _unit_radius(t, params)
    U = np.column_stack([r * np.cos(t), r * np.sin(t)])  # unit, deformed
    # Scale to ellipse
    S = np.diag([params.a, params.b]).astype(np.float64)
    E = U @ S.T
    # Rotate
    R = _rot2d(np.deg2rad(params.angle_deg))
    ER = E @ R.T
    # Translate
    C = np.array(params.center, dtype=np.float64)[None, :]
    X = ER + C
    # Noise
    if params.noise_std > 0.0:
        X += rng.normal(scale=params.noise_std, size=X.shape)
    return X.astype(np.float64)


# --------------------------------------------------------------------------------------
# Fitting + plotting
# --------------------------------------------------------------------------------------


def fit_and_plot(
    points_xy: ArrayF,
    out_png: Path,
    *,
    title: str,
    sample_n: int = 200,
    bezier_err: float = 1.0,
) -> None:
    """
    Fit a closed point-cloud with piecewise cubic Bézier and closed B-spline,
    and save an overlay PNG (data + both fits).
    Also save CSVs with the raw points and the evenly-spaced samples.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_base = out_png.with_suffix("")  # stem path

    # Uniformize data density a bit (helps metrics and fairness of fits)
    data = resample_closed_polyline(points_xy, n=max(points_xy.shape[0], sample_n))

    # Bézier (piecewise)
    bz_cfg = BezierPWConfig(max_error=bezier_err, sample_n=sample_n)
    bz_res = fit_contour_bezier_piecewise(data, bz_cfg)

    # Closed B-spline
    bs_cfg = BSplineConfig(s=0.001, k=3, sample_n=sample_n)
    bs_res = fit_contour_bspline_closed(data, bs_cfg)

    # --- plots ---
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    # raw points
    step = max(1, data.shape[0] // 600)
    ax.plot(
        data[::step, 0], data[::step, 1], ".", ms=2.0, alpha=0.6, label="point cloud"
    )
    # Bézier samples (evenly spaced)
    B = bz_res.samples_xy
    ax.plot(
        np.r_[B[:, 0], B[0, 0]],
        np.r_[B[:, 1], B[0, 1]],
        "-",
        lw=1.6,
        label="piecewise Bézier",
    )
    # B-spline samples (evenly spaced)
    S = bs_res.samples_xy
    ax.plot(
        np.r_[S[:, 0], S[0, 0]],
        np.r_[S[:, 1], S[0, 1]],
        "-",
        lw=1.6,
        label="closed B-spline",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # --- CSV outputs (raw + fitted samples) ---
    np.savetxt(
        out_base.with_suffix(".points.csv"),
        data,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.9f",
    )
    np.savetxt(
        out_base.with_name(out_base.name + "_bezier.samples.csv"),
        bz_res.samples_xy,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.9f",
    )
    np.savetxt(
        out_base.with_name(out_base.name + "_bspline.samples.csv"),
        bs_res.samples_xy,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.9f",
    )

    # --- quick metrics in a sidecar txt (max/mean vertex→curve) ---
    with out_base.with_suffix(".txt").open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(
            f"Bézier:  max_err = {bz_res.metrics.max_error:.4f}   mean_err = {bz_res.metrics.mean_error:.4f}\n"
        )
        f.write(
            f"B-spline: max_err = {bs_res.metrics.max_error:.4f}   mean_err = {bs_res.metrics.mean_error:.4f}\n"
        )


# --------------------------------------------------------------------------------------
# Preset cases (you can tweak or add more)
# --------------------------------------------------------------------------------------


def preset_cases(rng: np.random.RandomState) -> List[Tuple[str, Callable[[], ArrayF]]]:
    """
    Return a list of (case_name, generator) producing closed point-clouds.
    """
    base_center = (0.0, 0.0)

    cases: List[Tuple[str, Callable[[], ArrayF]]] = []

    # 1) Simple rotated ellipse, mild noise
    def case1() -> ArrayF:
        P = ShapeParams(
            center=base_center, a=42.0, b=27.0, angle_deg=25.0, noise_std=0.25
        )
        return generate_deformed_ellipse(n_points=420, params=P, rng=rng)

    cases.append(("01_rotated_ellipse", case1))

    # 2) Deformed ellipse (k=2 and k=3 harmonics)
    def case2() -> ArrayF:
        P = ShapeParams(
            center=base_center,
            a=43.0,
            b=28.0,
            angle_deg=35.0,
            noise_std=0.30,
            harmonics=((2, 0.05, 10.0), (3, 0.035, -20.0)),
        )
        return generate_deformed_ellipse(n_points=480, params=P, rng=rng)

    cases.append(("02_deformed_ellipse_harmonics", case2))

    # 3) Notched ellipse (small cortical notch)
    def case3() -> ArrayF:
        P = ShapeParams(
            center=base_center,
            a=44.0,
            b=26.0,
            angle_deg=18.0,
            noise_std=0.28,
            harmonics=((2, 0.03, 0.0),),
            notch_enable=True,
            notch_center_deg=20.0,
            notch_sigma_deg=7.0,
            notch_depth=0.08,
        )
        return generate_deformed_ellipse(n_points=480, params=P, rng=rng)

    cases.append(("03_notched_ellipse", case3))

    # 4) Stronger deformation (multi-harmonic), rotated
    def case4() -> ArrayF:
        P = ShapeParams(
            center=base_center,
            a=41.0,
            b=23.0,
            angle_deg=42.0,
            noise_std=0.35,
            harmonics=((2, 0.06, 30.0), (4, 0.04, -10.0), (5, 0.03, 60.0)),
        )
        return generate_deformed_ellipse(n_points=520, params=P, rng=rng)

    cases.append(("04_strong_deformation", case4))

    return cases


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic closed point-clouds (humerus-like cross-sections), "
        "fit piecewise Bézier and closed B-spline, and save overlays + CSVs."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/synthetic_report"),
        help="Output directory for PNG and CSV files.",
    )
    ap.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    ap.add_argument(
        "--bezier-max-err",
        type=float,
        default=1.0,
        help="Max vertex-to-polyline error target for piecewise Bézier (in data units).",
    )
    ap.add_argument(
        "--sample-n",
        type=int,
        default=200,
        help="Number of evenly-spaced samples per fitted curve.",
    )
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    suites = preset_cases(rng)

    # Run all cases
    for name, gen in suites:
        pts = gen()
        out_png = args.out_dir / f"{name}.png"
        fit_and_plot(
            pts,
            out_png,
            title=f"{name.replace('_', ' ')}",
            sample_n=int(args.sample_n),
            bezier_err=float(args.bezier_max_err),
        )
        print(f"[OK] {name} -> {out_png}")

    print(f"\nReport saved under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
