from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from medvis.geometry.bezier import (
    BezierCurve,
    fit_bezier_interpolate,
    chord_parameterization,
)


def load_skull_data(data_dir: Path) -> dict:
    left_x = np.loadtxt(data_dir / "borde_craneo_parte_izquierda_Eje_X.txt")
    left_y = np.loadtxt(data_dir / "borde_craneo_parte_izquierda_Eje_Y.txt")
    right_x = np.loadtxt(data_dir / "borde_craneo_parte_derecha_Eje_X.txt")
    right_y = np.loadtxt(data_dir / "borde_craneo_parte_derecha_Eje_Y.txt")

    left_pts = np.column_stack([left_x, left_y])
    right_pts = np.column_stack([right_x, right_y])

    return {
        "left": left_pts,
        "right": right_pts,
        "full": np.vstack([left_pts, right_pts[::-1]]),
    }


def sample_points_from_contour(
    contour: np.ndarray, n_samples: int, skip_ends: int = 0
) -> np.ndarray:
    N = contour.shape[0]
    start = skip_ends
    end = N - skip_ends
    if end <= start:
        start, end = 0, N
    indices = np.linspace(start, end - 1, n_samples, dtype=int)
    return contour[indices]


def split_skull_branches(sampled_pts: np.ndarray, center_y: float) -> tuple:
    mask_upper = sampled_pts[:, 1] <= center_y
    mask_lower = sampled_pts[:, 1] > center_y

    upper_pts = sampled_pts[mask_upper]
    lower_pts = sampled_pts[mask_lower]

    if upper_pts.shape[0] > 0:
        upper_pts = upper_pts[np.argsort(upper_pts[:, 0])]
    if lower_pts.shape[0] > 0:
        lower_pts = lower_pts[np.argsort(lower_pts[:, 0])]

    return upper_pts, lower_pts


def plot_skull_overview(skull_data: dict, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    left = skull_data["left"]
    right = skull_data["right"]

    ax.plot(left[:, 0], left[:, 1], ".", ms=1, alpha=0.5, label="Left contour")
    ax.plot(right[:, 0], right[:, 1], ".", ms=1, alpha=0.5, label="Right contour")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Complete Skull Contour")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_lemniscate_interpolation(
    skull_data: dict,
    sampled_pts: np.ndarray,
    branch1_curve: BezierCurve,
    branch2_curve: BezierCurve,
    center_y: float,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    full = skull_data["full"]
    ax.plot(
        full[:, 0],
        full[:, 1],
        ".",
        ms=0.5,
        alpha=0.3,
        color="gray",
        label="Full contour",
    )
    ax.plot(sampled_pts[:, 0], sampled_pts[:, 1], "o", ms=6, label="Sampled points")

    tt = np.linspace(0.0, 1.0, 500)
    if branch1_curve is not None:
        B1 = branch1_curve.evaluate_batch(tt)
        ax.plot(
            B1[:, 0], B1[:, 1], "-", lw=2.5, label="Lemniscate branch 1", color="red"
        )

    if branch2_curve is not None:
        B2 = branch2_curve.evaluate_batch(tt)
        ax.plot(
            B2[:, 0], B2[:, 1], "-", lw=2.5, label="Lemniscate branch 2", color="blue"
        )

    ax.axhline(
        y=center_y,
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Split at Y={center_y:.1f}",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Skull Contour with Lemniscate Bézier Interpolation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Skull contour approximation using lemniscate Bézier branches"
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/skull"),
        help="Directory containing skull contour data files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/skull_lemniscate"),
        help="Output directory for figures",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=15,
        help="Number of points to sample from contour",
    )
    ap.add_argument(
        "--center-y",
        type=float,
        default=None,
        help="Y coordinate to split branches (auto if None)",
    )
    args = ap.parse_args()

    skull_data = load_skull_data(Path(args.data_dir))

    plot_skull_overview(skull_data, args.out_dir / "skull_overview.png")
    print(f"Generated: {args.out_dir / 'skull_overview.png'}")

    full_contour = skull_data["full"]
    sampled_pts = sample_points_from_contour(full_contour, args.n_samples, skip_ends=3)

    if args.center_y is None:
        center_y = float(np.median(sampled_pts[:, 1]))
    else:
        center_y = float(args.center_y)

    upper_pts, lower_pts = split_skull_branches(sampled_pts, center_y)

    branch1_curve = None
    branch2_curve = None

    if upper_pts.shape[0] >= 3:
        u1 = chord_parameterization(upper_pts, alpha=1.0, normalize=True)
        branch1_curve = fit_bezier_interpolate(upper_pts, params=u1)
        print(
            f"Branch 1 (upper): {upper_pts.shape[0]} points, degree {branch1_curve.degree}"
        )

    if lower_pts.shape[0] >= 3:
        u2 = chord_parameterization(lower_pts, alpha=1.0, normalize=True)
        branch2_curve = fit_bezier_interpolate(lower_pts, params=u2)
        print(
            f"Branch 2 (lower): {lower_pts.shape[0]} points, degree {branch2_curve.degree}"
        )

    plot_lemniscate_interpolation(
        skull_data,
        sampled_pts,
        branch1_curve,
        branch2_curve,
        center_y,
        args.out_dir / "lemniscate_interpolation.png",
    )
    print(f"Generated: {args.out_dir / 'lemniscate_interpolation.png'}")

    out_csv = args.out_dir / "sampled_points.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_csv, sampled_pts, delimiter=",", header="x,y", comments="", fmt="%.3f"
    )
    print(f"Saved sampled points: {out_csv}")

    if branch1_curve is not None:
        np.savetxt(
            args.out_dir / "branch1_control.csv",
            branch1_curve.control_points,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%.3f",
        )

    if branch2_curve is not None:
        np.savetxt(
            args.out_dir / "branch2_control.csv",
            branch2_curve.control_points,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%.3f",
        )

    print(f"\nReport completed in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
