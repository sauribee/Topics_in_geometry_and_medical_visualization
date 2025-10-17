"""
Script to optimize the spline estimation for an ellipse without requiring graphical interface.

This script tests different point positions to find the best
approximation of the spline to an ellipse.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import argparse
from assignment_2.curves.spline2d.curve import Spline2D


def calculate_error(curve, a, b, num_samples=100):
    """
    Calculates the error between a spline and an ellipse.

    Args:
        curve: Spline2D object.
        a, b: Semi-axes of the ellipse.
        num_samples: Number of points to calculate the error.

    Returns:
        error: Average error between the spline and the ellipse.
    """
    # Sample the ellipse
    theta = np.linspace(0, 2 * np.pi, num_samples)
    ellipse_points = np.column_stack((a * np.cos(theta), b * np.sin(theta)))

    # Map the theta values to the range of t parameters of the spline
    # We need to avoid problems with extreme values, so we limit to the exact range
    t_min, t_max = curve.t[0], curve.t[-1]

    # Use a linear space within the safe range to avoid errors
    t_safe = np.linspace(t_min, t_max - 1e-10, num_samples)

    # Evaluate the spline at the safe parameters
    spline_points = curve(t_safe)

    # Calculate the mean squared error
    # Since the spline and ellipse points are now at different parameters,
    # we calculate an approximate error by finding the average distance
    # between each point on the ellipse and its closest point on the spline

    # Initialize the error
    squared_distances = np.zeros(num_samples)

    # For each point on the ellipse, find the closest point on the spline
    for i, ellipse_point in enumerate(ellipse_points):
        # Calculate distances to all points on the spline
        distances = np.sum((spline_points - ellipse_point) ** 2, axis=1)
        # Take the minimum distance
        squared_distances[i] = np.min(distances)

    # Calculate the square root of the mean of squared distances
    error = np.sqrt(np.mean(squared_distances))
    return error


def generate_curve_with_points(points_angles, a=4, b=5):
    """
    Generates a Spline2D object from point angles.

    Args:
        points_angles: Angles (in radians) for the points on the ellipse.
        a, b: Semi-axes of the ellipse.

    Returns:
        curve: Generated Spline2D object.
    """
    # Sort the angles to maintain parameterization
    points_angles = np.sort(points_angles)

    # Generate points on the ellipse
    points = np.array(
        [(a * np.cos(angle), b * np.sin(angle)) for angle in points_angles]
    )

    # Create and return the spline
    return Spline2D(points, t=points_angles)


def random_search(num_iterations=1000, num_points=5, a=4, b=5, seed=None):
    """
    Random search to find the best position of points.

    Args:
        num_iterations: Number of point configurations to try.
        num_points: Number of points to use.
        a, b: Semi-axes of the ellipse.
        seed: Seed for reproducibility.

    Returns:
        best_angles: Best angles found.
        best_error: Minimum error found.
        best_curve: Best spline found.
    """
    if seed is not None:
        np.random.seed(seed)

    best_error = float("inf")
    best_angles = None
    best_curve = None

    for i in range(num_iterations):
        # Generate random angles
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))

        # Create curve with these angles
        curve = generate_curve_with_points(angles, a, b)

        # Calculate error
        error = calculate_error(curve, a, b)

        # Update if we find a better result
        if error < best_error:
            best_error = error
            best_angles = angles.copy()
            best_curve = curve

            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"Iteration {i+1}/{num_iterations}: New best error = {best_error:.6f}"
                )

    print(f"\nBest error found: {best_error:.6f}")
    print(f"Best angles: {best_angles}")

    return best_angles, best_error, best_curve


def save_comparison_plot(best_curve, original_curve, a, b, output_dir, timestamp=None):
    """
    Saves a comparison between the original and optimized splines.

    Args:
        best_curve: Best spline found.
        original_curve: Original spline (from the ellipse example).
        a, b: Semi-axes of the ellipse.
        output_dir: Directory where to save the image.
        timestamp: Timestamp for the filename.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot for the original spline
    t_orig = np.linspace(original_curve.t[0], original_curve.t[-1] - 1e-10, 100)
    original_points = original_curve(t_orig)
    ax1.plot(
        original_points[:, 0], original_points[:, 1], "b-", label="Original Spline"
    )

    # Show original control points
    ax1.plot(
        original_curve.points[:, 0],
        original_curve.points[:, 1],
        "ro",
        label="Orig. Control Points",
    )

    # Draw the ellipse in both plots
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)
    ax1.plot(ellipse_x, ellipse_y, "r--", label="Real Ellipse")

    # Calculate and show the original error
    original_error = calculate_error(original_curve, a, b)
    ax1.set_title(f"Original Spline (Error: {original_error:.6f})")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    # Plot for the optimized spline
    t_best = np.linspace(best_curve.t[0], best_curve.t[-1] - 1e-10, 100)
    best_points = best_curve(t_best)
    ax2.plot(best_points[:, 0], best_points[:, 1], "g-", label="Optimized Spline")

    # Show optimized control points
    ax2.plot(
        best_curve.points[:, 0],
        best_curve.points[:, 1],
        "mo",
        label="Opt. Control Points",
    )

    # Draw the ellipse
    ax2.plot(ellipse_x, ellipse_y, "r--", label="Real Ellipse")

    # Calculate and show the optimized error
    best_error = calculate_error(best_curve, a, b)
    ax2.set_title(f"Optimized Spline (Error: {best_error:.6f})")
    ax2.grid(True)
    ax2.axis("equal")
    ax2.legend()

    fig.suptitle(f"Comparison of Splines for Ellipse (a={a}, b={b})")
    plt.tight_layout()

    # Save the image
    filename = f"optimized_ellipse_spline_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Image saved to: {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Spline optimizer for ellipse")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for random search",
    )
    parser.add_argument(
        "--points", type=int, default=5, help="Number of control points for the spline"
    )
    parser.add_argument(
        "--a", type=float, default=4, help="Semi-major axis of the ellipse"
    )
    parser.add_argument(
        "--b", type=float, default=5, help="Semi-minor axis of the ellipse"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )

    args = parser.parse_args()

    # Configure output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    # Print informative message
    print(
        f"Searching for the best configuration of {args.points} points to approximate an ellipse..."
    )
    print(f"Ellipse parameters: a={args.a}, b={args.b}")
    print(f"Performing {args.iterations} iterations...")

    # Perform random search
    best_angles, best_error, best_curve = random_search(
        num_iterations=args.iterations,
        num_points=args.points,
        a=args.a,
        b=args.b,
        seed=args.seed,
    )

    # Create the original spline for comparison
    np.random.seed(42)  # Same seed as in the original example
    original_angles = np.sort(np.random.uniform(0, 2 * np.pi, args.points))
    original_curve = generate_curve_with_points(original_angles, args.a, args.b)

    # Save comparison
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = save_comparison_plot(
        best_curve, original_curve, args.a, args.b, output_dir, timestamp
    )

    print("\nOptimization completed!")
    print(f"Comparison image saved to: {filepath}")
    print(f"Original error: {calculate_error(original_curve, args.a, args.b):.6f}")
    print(f"Optimized error: {best_error:.6f}")

    # Save the best angles for future reference
    print("\nOptimized points:")
    for i, angle in enumerate(best_angles):
        x = args.a * np.cos(angle)
        y = args.b * np.sin(angle)
        print(f"Point {i+1}: ({x:.4f}, {y:.4f}) [angle: {angle:.4f} rad]")


if __name__ == "__main__":
    main()
