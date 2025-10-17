#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating 3D visualizations of the humerus from detected contours.
"""

import os
import argparse
import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt
from humerus_detection.io import dicom_utils


def extract_contours_from_images(results_path):
    """
    Extract contours from result images.

    Args:
        results_path: Path to the folder with result images

    Returns:
        list: List of contours for each slice
    """
    # Search for result files (using specific name pattern for splines)
    files = sorted(glob.glob(os.path.join(results_path, "*_advanced.png")))
    terminal_files = sorted(glob.glob(os.path.join(results_path, "*_terminal_*.png")))

    files = files + terminal_files

    # Sort by image number
    def extract_number(name):
        try:
            return int(os.path.basename(name).split("_")[0][1:])
        except Exception:
            return 0

    files = sorted(files, key=extract_number)

    if not files:
        print("No result files found.")
        return []

    print(f"Found {len(files)} result files for 3D reconstruction.")

    # Extract blue spline contours from images
    contours = []

    for i, file in enumerate(files):
        # Read the image
        img = imageio.imread(file)

        # Find blue pixels (spline) - high Blue channel, low Red and Green
        blue_mask = (img[:, :, 2] > 200) & (img[:, :, 0] < 100) & (img[:, :, 1] < 100)

        # Extract coordinates of blue pixels
        coords = np.column_stack(np.where(blue_mask))

        if len(coords) > 10:  # If there are enough points to form a contour
            # Sort points to form a closed contour
            center_y, center_x = np.mean(coords, axis=0)

            # Convert to polar coordinates
            y_rel = coords[:, 0] - center_y
            x_rel = coords[:, 1] - center_x
            angles = np.arctan2(y_rel, x_rel)

            # Sort by angle
            idx_sorted = np.argsort(angles)
            sorted_contour = coords[idx_sorted]

            contours.append(sorted_contour)
            print(f"Processed {os.path.basename(file)}: {len(sorted_contour)} points")
        else:
            # If not enough points (possibly end of humerus)
            print(f"No contour found in {os.path.basename(file)}")
            # Use the last available contour but smaller to simulate termination
            if contours:
                last_contour = contours[-1].copy()
                last_center = np.mean(last_contour, axis=0)
                reduction_factor = max(0.1, 1.0 - i / len(files))  # Gradual reduction

                reduced_contour = (
                    last_center + (last_contour - last_center) * reduction_factor
                )
                contours.append(reduced_contour)

                print(f"Generated terminal contour for {os.path.basename(file)}")
            else:
                # If no previous contour, create an empty one
                contours.append(np.array([]))

    return contours


def generate_3d_model(contours, z_spacing, pixel_spacing, output_path):
    """
    Generate a 3D model from stacked contours.

    Args:
        contours: List of contours for each slice
        z_spacing: Spacing in mm between slices
        pixel_spacing: Pixel resolution (x, y) in mm
        output_path: Path to save the 3D model
    """
    # Filter empty contours
    valid_contours = [c for c in contours if len(c) > 0]

    if not valid_contours:
        print("No valid contours to generate the 3D model.")
        return

    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # For uniform visualization, normalize each contour to an equal number of points
    n_points = 100

    # Colormap to color the contours by height
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_contours)))

    # Create 3D surface
    for i, contour in enumerate(valid_contours):
        # Z coordinate of the slice (based on DICOM spacing)
        z = i * z_spacing

        # Scale XY coordinates according to pixel resolution
        x = contour[:, 1] * pixel_spacing[0]
        y = contour[:, 0] * pixel_spacing[1]

        # Resample the contour with B-splines to have uniform points
        if len(contour) > 3:
            try:
                from scipy.interpolate import splprep, splev

                # Arc length parameterization
                t = np.zeros(len(x))
                t[1:] = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
                t = t / t[-1]

                # Closed B-spline
                tck, u = splprep([x, y], u=t, s=0, per=1)
                u_new = np.linspace(0, 1, n_points)
                x_new, y_new = splev(u_new, tck)

                # Draw contour
                ax.plot(
                    x_new,
                    y_new,
                    [z] * len(x_new),
                    "-",
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2,
                )

                # Connect with previous contour if it exists
                if i > 0 and len(valid_contours[i - 1]) > 3:
                    x_prev = valid_contours[i - 1][:, 1] * pixel_spacing[0]
                    y_prev = valid_contours[i - 1][:, 0] * pixel_spacing[1]
                    z_prev = (i - 1) * z_spacing

                    # Resample previous contour
                    t_prev = np.zeros(len(x_prev))
                    t_prev[1:] = np.cumsum(
                        np.sqrt(np.diff(x_prev) ** 2 + np.diff(y_prev) ** 2)
                    )
                    t_prev = t_prev / t_prev[-1]

                    tck_prev, u_prev = splprep([x_prev, y_prev], u=t_prev, s=0, per=1)
                    x_prev_new, y_prev_new = splev(u_new, tck_prev)

                    # Connect corresponding points between contours
                    for j in range(
                        0, n_points, 5
                    ):  # Connect every 5 points to avoid overloading
                        ax.plot(
                            [x_prev_new[j], x_new[j]],
                            [y_prev_new[j], y_new[j]],
                            [z_prev, z],
                            "-",
                            color="gray",
                            alpha=0.3,
                            linewidth=1,
                        )
            except Exception as e:
                print(f"Error processing contour {i}: {e}")
                continue

    # Visualization settings
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Reconstruction of the Humerus")

    # Adjust view
    ax.view_init(elev=30, azim=45)

    # Save image
    os.makedirs(output_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "3d_model.png"), dpi=300, bbox_inches="tight")

    # Also save as a GIF with rotating view
    print("Generating GIF with 3D rotation...")

    # Create temporary folder for frames
    temp_dir = os.path.join(output_path, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate frames with different angles
    frames = []
    for angle in range(0, 360, 10):
        ax.view_init(elev=20, azim=angle)
        plt.savefig(os.path.join(temp_dir, f"frame_{angle:03d}.png"), dpi=150)
        frames.append(imageio.imread(os.path.join(temp_dir, f"frame_{angle:03d}.png")))

    # Create GIF
    imageio.mimsave(
        os.path.join(output_path, "3d_model_rotation.gif"), frames, duration=0.2
    )

    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"3D model saved to: {os.path.join(output_path, '3d_model.png')}")
    print(
        f"Rotation GIF saved to: {os.path.join(output_path, '3d_model_rotation.gif')}"
    )

    plt.show()


def main():
    """
    Main function to parse arguments and run the 3D visualization.
    """
    parser = argparse.ArgumentParser(
        description="Generate 3D visualization of the humerus from detected contours."
    )

    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="assignment_4/axial_sections",
        help="Directory containing the original DICOM files (for spacing information)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="assignment_4/advanced_results",
        help="Directory containing the detection results (images with contours)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="assignment_4/3d_model",
        help="Directory where to save the 3D model",
    )

    args = parser.parse_args()

    # Print configuration
    print("Humerus 3D Visualization")
    print("-----------------------")
    print(f"DICOM directory: {args.dicom_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-----------------------")

    # Load DICOM metadata for correct spacing
    z_spacing, pixel_spacing = dicom_utils.load_slice_spacing(args.dicom_dir)
    print(f"Slice spacing: {z_spacing:.2f} mm")
    print(f"Pixel resolution: {pixel_spacing[0]:.2f} x {pixel_spacing[1]:.2f} mm")

    # Extract contours from result images
    contours = extract_contours_from_images(args.results_dir)

    if contours:
        # Generate 3D model
        generate_3d_model(contours, z_spacing, pixel_spacing, args.output_dir)
    else:
        print("Could not load contours for 3D visualization.")


if __name__ == "__main__":
    main()
