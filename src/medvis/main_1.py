"""
Main script for Assignment 1: Spline Interpolation in 1D.

This script allows running examples of 1D spline interpolation methods,
including both linear and quadratic splines.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from assignment_1.linear.spline import (
    LinearSpline,
    example_rocket as linear_example_rocket,
)
from assignment_1.quadratic.spline import (
    QuadraticSpline,
    example_rocket as quadratic_example_rocket,
)
from assignment_1.utils.compare import compare_methods


def main():
    """
    Main function to run spline interpolation examples.
    """
    parser = argparse.ArgumentParser(description="Spline Interpolation Examples")
    parser.add_argument(
        "--method",
        choices=["linear", "quadratic", "compare"],
        default="all",
        help="Interpolation method to use",
    )
    parser.add_argument(
        "--eval_point", type=float, default=16.0, help="Evaluation point for velocity"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom data points instead of rocket example",
    )
    parser.add_argument("--x", nargs="+", type=float, help="Custom x values (times)")
    parser.add_argument("--y", nargs="+", type=float, help="Custom y values (heights)")
    parser.add_argument(
        "--save_only",
        action="store_true",
        help="Only save the plot as image without displaying it",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assignment_1",
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Custom data handling
    if args.custom:
        if args.x is None or args.y is None:
            print("Error: Custom data requires both --x and --y arguments")
            return

        if len(args.x) != len(args.y):
            print("Error: Number of x and y values must be the same")
            return

        print(f"Using custom data with {len(args.x)} points")
        t = np.array(args.x)
        h = np.array(args.y)

        # Run selected method with custom data
        if args.method == "linear" or args.method == "all":
            fig, _ = plt.subplots(figsize=(10, 6))
            linear_spline = LinearSpline(t, h)
            linear_spline.plot()
            plt.title("Linear Spline Interpolation (Custom Data)")

            # Save the plot
            filename = f"linear_spline_custom_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Linear spline plot saved to: {filepath}")

            if not args.save_only:
                plt.show()

        if args.method == "quadratic" or args.method == "all":
            fig, _ = plt.subplots(figsize=(10, 6))
            quadratic_spline = QuadraticSpline(t, h)
            quadratic_spline.plot()
            plt.title("Quadratic Spline Interpolation (Custom Data)")

            # Save the plot
            filename = f"quadratic_spline_custom_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Quadratic spline plot saved to: {filepath}")

            if not args.save_only:
                plt.show()

        if args.method == "compare":
            fig, _ = plt.subplots(figsize=(10, 6))
            compare_methods(t, h)
            plt.title("Comparison of Spline Methods (Custom Data)")

            # Save the plot
            filename = f"spline_comparison_custom_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Comparison plot saved to: {filepath}")

            if not args.save_only:
                plt.show()

    # Rocket example
    else:
        if args.method == "linear" or args.method == "all":
            print("\n=== Linear Spline Example (Rocket Motion) ===")
            v_linear = linear_example_rocket(
                args.eval_point,
                show_plot=not args.save_only,
                save_plot=True,
                output_dir=output_dir,
                timestamp=timestamp,
            )
            print(f"Velocity at t = {args.eval_point}s: {v_linear:.2f} m/s")

        if args.method == "quadratic" or args.method == "all":
            print("\n=== Quadratic Spline Example (Rocket Motion) ===")
            results = quadratic_example_rocket(
                args.eval_point,
                show_plot=not args.save_only,
                save_plot=True,
                output_dir=output_dir,
                timestamp=timestamp,
            )
            print(f"Velocity at t = {args.eval_point}s: {results['velocity']:.2f} m/s")
            print(
                f"Distance covered between t = 11s and t = 16s: {results['distance']:.2f} m"
            )
            print(
                f"Acceleration at t = {args.eval_point}s: {results['acceleration']:.2f} m/s²"
            )

        if args.method == "compare":
            print("\n=== Comparing Linear and Quadratic Splines (Rocket Motion) ===")
            # Rocket data
            t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
            h = np.array(
                [0, 2, 5, 9, 15, 22, 29, 37, 46, 58, 72, 88, 106, 127, 150, 176]
            )

            fig, _ = plt.subplots(figsize=(10, 6))
            compare_methods(t, h)
            plt.title("Comparison of Spline Methods (Rocket Motion)")

            # Save the plot
            filename = f"spline_comparison_rocket_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Comparison plot saved to: {filepath}")

            if not args.save_only:
                plt.show()

    # Uncomment to run the Runge Phenomenon example
    # demonstrate_runge_phenomenon()


def demonstrate_runge_phenomenon():
    """
    Demonstrate the Runge Phenomenon with high-degree polynomials vs. splines.
    """
    # Create Runge function data
    x = np.linspace(-5, 5, 11)  # 11 equally spaced points
    y = 1 / (1 + x**2)  # Runge function

    # Create denser x values for plotting the true function and interpolations
    x_dense = np.linspace(-5, 5, 1000)
    y_true = 1 / (1 + x_dense**2)

    # Create splines
    linear_spline = LinearSpline(x, y)
    quadratic_spline = QuadraticSpline(x, y)

    # Evaluate splines at dense points
    y_linear = np.array([linear_spline(xi) for xi in x_dense])
    y_quadratic = np.array([quadratic_spline(xi) for xi in x_dense])

    # Create high-degree polynomial interpolation
    poly_coeffs = np.polyfit(x, y, len(x) - 1)
    y_poly = np.polyval(poly_coeffs, x_dense)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(x_dense, y_true, "k-", linewidth=2, label="True Runge Function")
    plt.plot(
        x_dense, y_poly, "r-", linewidth=1.5, label=f"Polynomial (degree {len(x)-1})"
    )
    plt.plot(x_dense, y_linear, "g--", linewidth=1.5, label="Linear Spline")
    plt.plot(x_dense, y_quadratic, "b-.", linewidth=1.5, label="Quadratic Spline")
    plt.plot(x, y, "ko", markersize=6, label="Data Points")

    plt.title("Runge Phenomenon: High-Degree Polynomial vs. Splines")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    # Add zoomed inset to show oscillations near the edges
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    # Create a zoomed inset axes
    axins = zoomed_inset_axes(plt.gca(), zoom=6, loc="upper right")
    axins.plot(x_dense, y_true, "k-", linewidth=2)
    axins.plot(x_dense, y_poly, "r-", linewidth=1.5)
    axins.plot(x_dense, y_linear, "g--", linewidth=1.5)
    axins.plot(x_dense, y_quadratic, "b-.", linewidth=1.5)
    axins.plot(x, y, "ko", markersize=6)

    # Set the limits of the zoomed region
    x1, x2 = 4.0, 5.0
    y1, y2 = -0.1, 0.1
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True)

    # Draw connecting lines
    mark_inset(plt.gca(), axins, loc1=3, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assignment_1",
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the plot
    filename = f"runge_phenomenon_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Runge phenomenon plot saved to: {filepath}")

    plt.show()


if __name__ == "__main__":
    main()
