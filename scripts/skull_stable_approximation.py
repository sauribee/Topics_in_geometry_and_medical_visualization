"""
Stable skull contour approximation using low-degree Bézier approximation.

This script demonstrates numerically stable approaches for approximating
skull contours, avoiding the pitfalls of high-degree interpolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from medvis.geometry.bezier import (
    BezierCurve,
    fit_bezier_lsq,
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


def extract_protuberance(skull_data: dict, y_threshold: float = 50.0) -> np.ndarray:
    full = skull_data["full"]
    mask = full[:, 1] < y_threshold
    protuberance = full[mask]

    if protuberance.shape[0] > 0:
        sorted_idx = np.argsort(protuberance[:, 0])
        protuberance = protuberance[sorted_idx]

    return protuberance


def compute_approximation_error(
    data: np.ndarray, curve: BezierCurve, n_eval: int = 1000
) -> float:
    if curve is None:
        return float("inf")

    tt = np.linspace(0.0, 1.0, n_eval)
    curve_pts = curve.evaluate_batch(tt)

    errors = []
    for p in data:
        dists = np.linalg.norm(curve_pts - p[None, :], axis=1)
        errors.append(np.min(dists))

    return float(np.mean(errors))


def plot_approximation_comparison(
    protuberance: np.ndarray,
    curve_deg5: BezierCurve,
    curve_deg10: BezierCurve,
    curve_deg15: BezierCurve,
    curve_deg20: BezierCurve,
    error_deg5: float,
    error_deg10: float,
    error_deg15: float,
    error_deg20: float,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    tt = np.linspace(0.0, 1.0, 1000)

    configs = [
        (axes[0], curve_deg5, f"Degree 5 LSQ\nError: {error_deg5:.3f} px"),
        (axes[1], curve_deg10, f"Degree 10 LSQ\nError: {error_deg10:.3f} px"),
        (axes[2], curve_deg15, f"Degree 15 LSQ\nError: {error_deg15:.3f} px"),
        (axes[3], curve_deg20, f"Degree 20 LSQ\nError: {error_deg20:.3f} px"),
    ]

    for ax, curve, title in configs:
        # Plot original data
        ax.plot(
            protuberance[:, 0],
            protuberance[:, 1],
            ".",
            ms=1,
            alpha=0.3,
            color="gray",
            label="Original data",
        )

        # Plot approximation
        if curve is not None:
            if hasattr(curve, "evaluate_batch"):
                B = curve.evaluate_batch(tt)
            else:
                B = curve.to_polyline(samples=1000)

            ax.plot(
                B[:, 0],
                B[:, 1],
                "-",
                lw=2.5,
                color="red",
                label="Approximation",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stable skull protuberance approximation using low-degree Bézier"
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
        default=Path("reports/figures/skull_stable"),
        help="Output directory for figures",
    )
    ap.add_argument(
        "--y-threshold",
        type=float,
        default=50.0,
        help="Y coordinate threshold for protuberance detection",
    )
    args = ap.parse_args()

    skull_data = load_skull_data(Path(args.data_dir))

    protuberance = extract_protuberance(skull_data, y_threshold=args.y_threshold)
    print(
        f"Protuberance detected: {protuberance.shape[0]} points below Y={args.y_threshold}"
    )

    # Fit with different degrees using LEAST SQUARES (not interpolation)
    print("\nFitting with least-squares approximation...")

    curve_deg5 = fit_bezier_lsq(protuberance, degree=5, parameterization_alpha=1.0)
    curve_deg10 = fit_bezier_lsq(protuberance, degree=10, parameterization_alpha=1.0)
    curve_deg15 = fit_bezier_lsq(protuberance, degree=15, parameterization_alpha=1.0)
    curve_deg20 = fit_bezier_lsq(protuberance, degree=20, parameterization_alpha=1.0)

    print(f"Degree 5: {curve_deg5.degree}")
    print(f"Degree 10: {curve_deg10.degree}")
    print(f"Degree 15: {curve_deg15.degree}")
    print(f"Degree 20: {curve_deg20.degree}")

    # Compute errors
    error_deg5 = compute_approximation_error(protuberance, curve_deg5)
    error_deg10 = compute_approximation_error(protuberance, curve_deg10)
    error_deg15 = compute_approximation_error(protuberance, curve_deg15)
    error_deg20 = compute_approximation_error(protuberance, curve_deg20)

    print("\nMean approximation errors:")
    print(f" Degree 5: {error_deg5:.3f} pixels")
    print(f" Degree 10: {error_deg10:.3f} pixels")
    print(f" Degree 15: {error_deg15:.3f} pixels")
    print(f" Degree 20: {error_deg20:.3f} pixels")

    # Generate comparison plot
    plot_approximation_comparison(
        protuberance,
        curve_deg5,
        curve_deg10,
        curve_deg15,
        curve_deg20,
        error_deg5,
        error_deg10,
        error_deg15,
        error_deg20,
        args.out_dir / "stable_comparison.png",
    )
    print(f"\nGenerated: {args.out_dir / 'stable_comparison.png'}")

    # Save summary
    with open(args.out_dir / "stable_summary.txt", "w") as f:
        f.write("Stable Skull Protuberance Approximation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("APPROACH: Least-squares approximation (NOT interpolation)\n")
        f.write("This avoids numerical instability of high-degree interpolation.\n\n")
        f.write(f"Y threshold: {args.y_threshold} pixels\n")
        f.write(f"Total protuberance points: {protuberance.shape[0]}\n\n")
        f.write("Approximation Results:\n")
        f.write(f"  Degree 5: error = {error_deg5:.3f} px\n")
        f.write(f"  Degree 10: error = {error_deg10:.3f} px\n")
        f.write(f"  Degree 15: error = {error_deg15:.3f} px\n")
        f.write(f"  Degree 20: error = {error_deg20:.3f} px\n")

    print(f"\nReport completed in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
