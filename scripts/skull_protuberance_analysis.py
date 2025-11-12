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


def extract_protuberance(skull_data: dict, y_threshold: float = 50.0) -> np.ndarray:
    full = skull_data["full"]
    mask = full[:, 1] < y_threshold
    protuberance = full[mask]

    if protuberance.shape[0] > 0:
        sorted_idx = np.argsort(protuberance[:, 0])
        protuberance = protuberance[sorted_idx]

    return protuberance


def sample_uniform_points(contour: np.ndarray, n_samples: int) -> np.ndarray:
    N = contour.shape[0]
    if n_samples >= N:
        return contour

    indices = np.linspace(0, N - 1, n_samples, dtype=int)
    return contour[indices]


def plot_protuberance_approximation(
    protuberance: np.ndarray,
    samples_5: np.ndarray,
    samples_6: np.ndarray,
    samples_7: np.ndarray,
    curve_5: BezierCurve,
    curve_6: BezierCurve,
    curve_7: BezierCurve,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    tt = np.linspace(0.0, 1.0, 1000)

    configs = [
        (axes[0], samples_5, curve_5, "5 points"),
        (axes[1], samples_6, curve_6, "6 points"),
        (axes[2], samples_7, curve_7, "7 points"),
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
        f"Protuberance detected: {protuberance.shape[0]} points above Y={args.y_threshold}"
    )

    samples_5 = sample_uniform_points(protuberance, 5)
    samples_6 = sample_uniform_points(protuberance, 6)
    samples_7 = sample_uniform_points(protuberance, 7)

    u_5 = chord_parameterization(samples_5, alpha=1.0, normalize=True)
    u_6 = chord_parameterization(samples_6, alpha=1.0, normalize=True)
    u_7 = chord_parameterization(samples_7, alpha=1.0, normalize=True)

    curve_5 = fit_bezier_interpolate(samples_5, params=u_5)
    curve_6 = fit_bezier_interpolate(samples_6, params=u_6)
    curve_7 = fit_bezier_interpolate(samples_7, params=u_7)

    print("\nBézier approximations:")
    print(f" 5 points: degree {curve_5.degree}")
    print(f" 6 points: degree {curve_6.degree}")
    print(f" 7 points: degree {curve_7.degree}")

    plot_protuberance_approximation(
        protuberance,
        samples_5,
        samples_6,
        samples_7,
        curve_5,
        curve_6,
        curve_7,
        args.out_dir / "protuberance_comparison.png",
    )
    print(f"\nGenerated: {args.out_dir / 'protuberance_comparison.png'}")

    error_5 = compute_approximation_error(protuberance, curve_5)
    error_6 = compute_approximation_error(protuberance, curve_6)
    error_7 = compute_approximation_error(protuberance, curve_7)

    print("\nMean approximation errors:")
    print(f"  5 points: {error_5:.3f} pixels")
    print(f"  6 points: {error_6:.3f} pixels")
    print(f"  7 points: {error_7:.3f} pixels")

    out_csv = args.out_dir / "protuberance_data.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_csv, protuberance, delimiter=",", header="x,y", comments="", fmt="%.3f"
    )

    for n, samples in [(5, samples_5), (6, samples_6), (7, samples_7)]:
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
        f.write("=" * 40 + "\n\n")
        f.write(f"Y threshold: {args.y_threshold} pixels\n")
        f.write(f"Total protuberance points: {protuberance.shape[0]}\n\n")
        f.write("Bézier Approximations:\n")
        f.write(f" 5 points: degree {curve_5.degree}, error = {error_5:.3f} px\n")
        f.write(f" 6 points: degree {curve_6.degree}, error = {error_6:.3f} px\n")
        f.write(f" 7 points: degree {curve_7.degree}, error = {error_7:.3f} px\n")

    print(f"\nReport completed in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
