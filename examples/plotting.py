#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for visualizing the results of humerus detection and modeling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def visualize_results(
    pixel_array: np.ndarray,
    contour: np.ndarray,
    x_spline: Optional[np.ndarray] = None,
    y_spline: Optional[np.ndarray] = None,
    title: str = "Humerus Detection",
) -> plt.Figure:
    """
    Visualizes the results of humerus detection and modeling.

    Args:
        pixel_array: Original image
        contour: Detected contour
        x_spline: X-coordinates of the B-spline
        y_spline: Y-coordinates of the B-spline
        title: Title of the figure

    Returns:
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show original image
    ax.imshow(pixel_array, cmap="gray")

    # Show detected contour
    if len(contour) > 0:
        ax.plot(
            contour[:, 1], contour[:, 0], "r-", linewidth=1, label="Detected contour"
        )

    # Show B-spline if available
    if x_spline is not None and y_spline is not None:
        ax.plot(x_spline, y_spline, "b-", linewidth=2, label="B-spline")

    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()

    return fig


def save_results(
    figure: plt.Figure, output_path: str, base_name: str, suffix: str = "result"
) -> None:
    """
    Saves the figure with the results.

    Args:
        figure: Figure to save
        output_path: Directory to save the results
        base_name: Base name for the saved files
        suffix: Suffix for the filename
    """
    os.makedirs(output_path, exist_ok=True)

    full_path = os.path.join(output_path, f"{base_name}_{suffix}.png")
    figure.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"Result saved to: {full_path}")


def create_composite_figure(
    pixel_array: np.ndarray,
    contour: np.ndarray,
    x_spline: Optional[np.ndarray] = None,
    y_spline: Optional[np.ndarray] = None,
    show_segmentation: bool = True,
) -> plt.Figure:
    """
    Creates a composite figure with multiple views of the results.

    Args:
        pixel_array: Original image
        contour: Detected contour
        x_spline: X-coordinates of the B-spline
        y_spline: Y-coordinates of the B-spline
        show_segmentation: Whether to show the segmentation mask

    Returns:
        The generated figure
    """
    # Determine number of subplots
    if show_segmentation:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(pixel_array, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Contour and spline
    axes[1].imshow(pixel_array, cmap="gray")
    if len(contour) > 0:
        axes[1].plot(contour[:, 1], contour[:, 0], "r-", linewidth=1, label="Contour")
    if x_spline is not None and y_spline is not None:
        axes[1].plot(x_spline, y_spline, "b-", linewidth=2, label="B-spline")
    axes[1].set_title("Detected Contour")
    axes[1].legend(loc="upper right")
    axes[1].axis("off")

    # Segmentation mask if requested
    if show_segmentation:
        # Create a mask from the spline
        mask = np.zeros_like(pixel_array, dtype=bool)
        if x_spline is not None and y_spline is not None:
            from skimage.draw import polygon

            # Convert spline to polygon for filling
            rr, cc = polygon(
                y_spline.astype(int), x_spline.astype(int), pixel_array.shape
            )
            if len(rr) > 0 and len(cc) > 0:
                mask[rr, cc] = True

        # Show mask
        axes[2].imshow(mask, cmap="viridis")
        axes[2].set_title("Segmentation Mask")
        axes[2].axis("off")

    plt.tight_layout()
    return fig


def visualize_3d_contours(
    contours: List[np.ndarray],
    z_positions: List[float],
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> plt.Figure:
    """
    Creates a 3D visualization of stacked contours.

    Args:
        contours: List of contours at different z positions
        z_positions: Z position of each contour
        pixel_spacing: Pixel spacing in mm (x, y)

    Returns:
        The generated figure
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Apply colormap based on height
    colors = plt.cm.viridis(np.linspace(0, 1, len(contours)))

    # Plot each contour
    for i, (contour, z) in enumerate(zip(contours, z_positions)):
        if len(contour) > 0:
            # Scale according to pixel spacing
            x = contour[:, 1] * pixel_spacing[0]
            y = contour[:, 0] * pixel_spacing[1]

            # Plot 3D contour
            ax.plot(x, y, [z] * len(x), "-", color=colors[i], alpha=0.7, linewidth=2)

    # Visualization settings
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Visualization of Contours")

    # Adjust view
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig
