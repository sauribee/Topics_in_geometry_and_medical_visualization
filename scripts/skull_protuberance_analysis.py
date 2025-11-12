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


def sample_uniform_points_arclength(contour: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample points uniformly along arc-length (not indices).
    This ensures even distribution along the curve geometry.
    """
    N = contour.shape[0]
    if n_samples >= N:
        return contour

    # Compute cumulative arc-length
    seg_lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = arc_lengths[-1]

    if total_length < 1e-10:
        # Degenerate case: uniform indices
        indices = np.linspace(0, N - 1, n_samples, dtype=int)
        return contour[indices]

    # Target arc-lengths for uniform sampling
    target_lengths = np.linspace(0, total_length, n_samples)

    # Interpolate to find corresponding indices
    sampled_points = []
    for target_s in target_lengths:
        # Find segment containing target_s
        idx = np.searchsorted(arc_lengths, target_s, side="right") - 1
        idx = np.clip(idx, 0, N - 2)

        # Linear interpolation within segment
        s0, s1 = arc_lengths[idx], arc_lengths[idx + 1]
        if s1 - s0 < 1e-10:
            t = 0.0
        else:
            t = (target_s - s0) / (s1 - s0)

        p = (1 - t) * contour[idx] + t * contour[idx + 1]
        sampled_points.append(p)

    return np.array(sampled_points)


def plot_protuberance_approximation(
    protuberance: np.ndarray,
    samples_5: np.ndarray,
    samples_7: np.ndarray,
    samples_9: np.ndarray,
    samples_6: np.ndarray,
    samples_8: np.ndarray,
    samples_10: np.ndarray,
    curve_5: BezierCurve,
    curve_7: BezierCurve,
    curve_9: BezierCurve,
    curve_6: BezierCurve,
    curve_8: BezierCurve,
    curve_10: BezierCurve,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    tt = np.linspace(0.0, 1.0, 1000)

    # Grid layout: [5, 7, 9] top row, [6, 8, 10] bottom row
    configs = [
        (axes[0, 0], samples_5, curve_5, "5 points"),
        (axes[0, 1], samples_7, curve_7, "7 points"),
        (axes[0, 2], samples_9, curve_9, "9 points"),
        (axes[1, 0], samples_6, curve_6, "6 points"),
        (axes[1, 1], samples_8, curve_8, "8 points"),
        (axes[1, 2], samples_10, curve_10, "10 points"),
    ]

    for ax, samples, curve, title in configs:
        ax.plot(
            protuberance[:, 0],
            protuberance[:, 1],
            ".",
            ms=0.5,
            alpha=0.3,
            color="gray",
            label="Original data",
        )

        ax.plot(
            samples[:, 0],
            samples[:, 1],
            "o",
            ms=8,
            color="blue",
            label="Sample points",
            zorder=5,
        )

        if curve is not None:
            B = curve.evaluate_batch(tt)
            ax.plot(
                B[:, 0],
                B[:, 1],
                "-",
                lw=2.5,
                color="red",
                label=f"Bézier (deg {curve.degree})",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(f"Protuberance approximation: {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def compute_approximation_error(
    protuberance: np.ndarray, curve: BezierCurve, n_eval: int = 1000
) -> float:
    if curve is None:
        return float("inf")

    tt = np.linspace(0.0, 1.0, n_eval)
    curve_pts = curve.evaluate_batch(tt)

    errors = []
    for p in protuberance:
        dists = np.linalg.norm(curve_pts - p[None, :], axis=1)
        errors.append(np.min(dists))

    return float(np.mean(errors))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Skull protuberance analysis with Bézier approximation"
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
        default=Path("reports/figures/skull_protuberance"),
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

    # Sample uniformly along arc-length
    samples_5 = sample_uniform_points_arclength(protuberance, 5)
    samples_7 = sample_uniform_points_arclength(protuberance, 7)
    samples_9 = sample_uniform_points_arclength(protuberance, 9)
    samples_6 = sample_uniform_points_arclength(protuberance, 6)
    samples_8 = sample_uniform_points_arclength(protuberance, 8)
    samples_10 = sample_uniform_points_arclength(protuberance, 10)

    # Use LSQ fitting with moderate degrees (NOT interpolation)
    # This is more stable than interpolation for similar number of points
    print("\nFitting with least-squares Bézier approximation...")
    curve_5 = fit_bezier_lsq(protuberance, degree=5, parameterization_alpha=1.0)
    curve_7 = fit_bezier_lsq(protuberance, degree=7, parameterization_alpha=1.0)
    curve_9 = fit_bezier_lsq(protuberance, degree=9, parameterization_alpha=1.0)
    curve_6 = fit_bezier_lsq(protuberance, degree=6, parameterization_alpha=1.0)
    curve_8 = fit_bezier_lsq(protuberance, degree=8, parameterization_alpha=1.0)
    curve_10 = fit_bezier_lsq(protuberance, degree=10, parameterization_alpha=1.0)

    print("\nBézier LSQ approximations:")
    print(f" Degree 5: {curve_5.degree}")
    print(f" Degree 7: {curve_7.degree}")
    print(f" Degree 9: {curve_9.degree}")
    print(f" Degree 6: {curve_6.degree}")
    print(f" Degree 8: {curve_8.degree}")
    print(f" Degree 10: {curve_10.degree}")

    plot_protuberance_approximation(
        protuberance,
        samples_5,
        samples_7,
        samples_9,
        samples_6,
        samples_8,
        samples_10,
        curve_5,
        curve_7,
        curve_9,
        curve_6,
        curve_8,
        curve_10,
        args.out_dir / "protuberance_comparison.png",
    )
    print(f"\nGenerated: {args.out_dir / 'protuberance_comparison.png'}")

    error_5 = compute_approximation_error(protuberance, curve_5)
    error_7 = compute_approximation_error(protuberance, curve_7)
    error_9 = compute_approximation_error(protuberance, curve_9)
    error_6 = compute_approximation_error(protuberance, curve_6)
    error_8 = compute_approximation_error(protuberance, curve_8)
    error_10 = compute_approximation_error(protuberance, curve_10)

    print("\nMean approximation errors:")
    print(f"  Degree 5: {error_5:.3f} pixels")
    print(f"  Degree 7: {error_7:.3f} pixels")
    print(f"  Degree 9: {error_9:.3f} pixels")
    print(f"  Degree 6: {error_6:.3f} pixels")
    print(f"  Degree 8: {error_8:.3f} pixels")
    print(f"  Degree 10: {error_10:.3f} pixels")

    out_csv = args.out_dir / "protuberance_data.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_csv, protuberance, delimiter=",", header="x,y", comments="", fmt="%.3f"
    )

    for n, samples in [
        (5, samples_5),
        (7, samples_7),
        (9, samples_9),
        (6, samples_6),
        (8, samples_8),
        (10, samples_10),
    ]:
        np.savetxt(
            args.out_dir / f"samples_{n}pts.csv",
            samples,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%.3f",
        )

    with open(args.out_dir / "approximation_summary.txt", "w") as f:
        f.write("Skull Protuberance Approximation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("METHOD: Least-squares approximation (NOT interpolation)\n")
        f.write("This provides stable, smooth curves without oscillations.\n\n")
        f.write(f"Y threshold: {args.y_threshold} pixels (below this value)\n")
        f.write(f"Total protuberance points: {protuberance.shape[0]}\n\n")
        f.write("Bézier LSQ Approximations (Grid 2x3):\n")
        f.write("Top row (odd degrees):\n")
        f.write(f"  Degree 5: error = {error_5:.3f} px\n")
        f.write(f"  Degree 7: error = {error_7:.3f} px\n")
        f.write(f"  Degree 9: error = {error_9:.3f} px\n")
        f.write("\nBottom row (even degrees):\n")
        f.write(f"  Degree 6: error = {error_6:.3f} px\n")
        f.write(f"  Degree 8: error = {error_8:.3f} px\n")
        f.write(f"  Degree 10: error = {error_10:.3f} px\n")

    print(f"\nReport completed in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
