"""
Script for 3D visualization of contours with B-splines.

This script generates a 3D surface where the Z-axis represents curvature or
some other property of the curve, useful for visual presentations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import splprep, splev, griddata
from datetime import datetime
import argparse


def sort_points_by_angle(points):
    """
    Sort points in counterclockwise order around their center of mass.

    Args:
        points (ndarray): Array of 2D points.

    Returns:
        ndarray: Sorted array of 2D points.
    """
    # Calculate center of mass
    center = np.mean(points, axis=0)

    # Calculate angles
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    print(f"Points sorted by angle around center {center}")
    return sorted_points


def load_points_from_mat(file_path):
    """
    Load points from a MATLAB .mat file.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        ndarray: Array of 2D points as float64.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        # Extract the points array (assuming it's stored with key 'ptos')
        points = mat_data.get("ptos", None)

        if points is None:
            raise ValueError(f"No 'ptos' key found in {file_path}")

        # Convert to float64
        points = points.astype(np.float64)

        print(f"Loaded {len(points)} points from {file_path}")
        print(f"Points shape: {points.shape}, dtype: {points.dtype}")

        return points
    except Exception as e:
        print(f"Error loading points from {file_path}: {e}")
        raise


def calculate_curvature(x, y):
    """
    Calculate the curvature at each point of a 2D curve.

    Args:
        x (ndarray): x-coordinates of the curve.
        y (ndarray): y-coordinates of the curve.

    Returns:
        ndarray: Curvature values for each point.
    """
    # Calculate first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Calculate second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Calculate curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2) ** (1.5)

    # Replace NaN or inf values
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    return curvature


def create_3d_visualization(
    points_file="ptos.mat",
    output_dir=None,
    degree=3,
    smoothing=0.0,
    sort_angular=True,
    z_scale=50.0,
    surface_type="curvature",
):
    """
    Create a 3D visualization of the contour where height represents
    curvature or another property.

    Args:
        points_file (str): Path to the .mat file with contour points.
        output_dir (str): Directory to save output images.
        degree (int): B-spline degree.
        smoothing (float): Smoothing parameter.
        sort_angular (bool): Whether to sort points angularly.
        z_scale (float): Scale factor for the Z-axis.
        surface_type (str): Type of surface. Options: 'curvature', 'distance'.
    """
    # Load the points
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(points_file):
        points_file = os.path.join(script_dir, points_file)

    # Configure default output directory inside assignment_3
    if output_dir is None:
        output_dir = os.path.join(script_dir, "output")

    # Create subfolder for 3D visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(
        output_dir, f"3d_visualization_{surface_type}_{timestamp}"
    )
    os.makedirs(output_subdir, exist_ok=True)

    points = load_points_from_mat(points_file)

    # Sort points by angle if requested
    if sort_angular:
        points = sort_points_by_angle(points)

    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Close the curve if necessary
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Create the B-spline
    tck, u = splprep([x, y], s=smoothing, k=degree, per=1)

    # Evaluate the B-spline at evenly spaced points
    u_new = np.linspace(0, 1, 200)
    x_new, y_new = splev(u_new, tck)

    # Calculate values for the Z-axis based on the selected type
    if surface_type == "curvature":
        # Calculate curvature along the curve
        z_values = calculate_curvature(x_new, y_new)
        z_label = "Curvature"
    elif surface_type == "distance":
        # Calculate distance from center of mass
        center = np.mean(points, axis=0)
        z_values = np.sqrt((x_new - center[0]) ** 2 + (y_new - center[1]) ** 2)
        z_label = "Distance to Center"
    else:
        # Constant value (to visualize only the shape)
        z_values = np.ones_like(x_new)
        z_label = "Height"

    # Scale Z values
    z_values = z_values * z_scale

    # Create mesh for the surface
    # First expand the points into an area
    margin = 20
    x_min, x_max = min(x_new) - margin, max(x_new) + margin
    y_min, y_max = min(y_new) - margin, max(y_new) + margin

    # Create regular grid
    grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # Interpolate Z values onto the grid
    # Use inverse distance as interpolation method
    grid_z = griddata(
        (x_new, y_new), z_values, (grid_x, grid_y), method="cubic", fill_value=0
    )

    # For points far from the curve, make Z decay with distance
    # This creates an "elevation" effect around the curve
    for i in range(grid_z.shape[0]):
        for j in range(grid_z.shape[1]):
            _grid_point = np.array([grid_x[i, j], grid_y[i, j]])
            # Find the minimum distance to any point on the curve
            distances = np.sqrt(
                (grid_x[i, j] - x_new) ** 2 + (grid_y[i, j] - y_new) ** 2
            )
            min_distance = np.min(distances)

            # Apply a decay factor based on distance
            decay_factor = np.exp(-(min_distance**2) / 300.0)
            grid_z[i, j] *= decay_factor

    # Create the 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        grid_x, grid_y, grid_z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.8
    )

    # Plot the 3D curve
    ax.plot(x_new, y_new, z_values, color="r", linewidth=3)

    # Plot control points
    ax.scatter(x, y, np.zeros_like(x), color="#FF4500", s=50, label="Control Points")

    # Configure limits and labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel(z_label)

    # Invert Y-axis to maintain the orientation of the original image
    ax.set_ylim(y_max, y_min)

    # Add title
    ax.set_title(
        f"3D Visualization of Rotator Cuff Contour\n"
        f"(B-spline Degree {degree}, Smoothing {smoothing})",
        fontsize=14,
    )

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=z_label)

    # Save images from different viewing angles

    # Save main view
    filename = "principal_view.png"
    filepath = os.path.join(output_subdir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"3D view saved to: {filepath}")

    # Save views from different angles
    angles = [(30, 45), (0, 0), (0, 90), (60, 30), (60, 120), (80, 200)]
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev, azim)
        angle_filename = f"angle{i}_elev{elev}_azim{azim}.png"
        angle_filepath = os.path.join(output_subdir, angle_filename)
        plt.savefig(angle_filepath, dpi=300, bbox_inches="tight")
        print(f"View from angle {elev}°, {azim}° saved to: {angle_filepath}")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Visualization of Contour with B-spline"
    )
    parser.add_argument(
        "--points_file",
        default="ptos.mat",
        help="Path to the .mat file with contour points",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to save output images"
    )
    parser.add_argument("--degree", type=int, default=3, help="B-spline degree")
    parser.add_argument(
        "--smoothing", type=float, default=10.0, help="Smoothing parameter"
    )
    parser.add_argument(
        "--no_sort_angular", action="store_true", help="Do not sort points angularly"
    )
    parser.add_argument(
        "--z_scale", type=float, default=50.0, help="Scale factor for the Z-axis"
    )
    parser.add_argument(
        "--surface_type",
        choices=["curvature", "distance"],
        default="curvature",
        help="Type of 3D surface: curvature or distance to center",
    )

    args = parser.parse_args()

    create_3d_visualization(
        points_file=args.points_file,
        output_dir=args.output_dir,
        degree=args.degree,
        smoothing=args.smoothing,
        sort_angular=not args.no_sort_angular,
        z_scale=args.z_scale,
        surface_type=args.surface_type,
    )
