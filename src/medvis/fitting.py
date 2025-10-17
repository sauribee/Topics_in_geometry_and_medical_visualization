#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for fitting B-splines to contours.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import splprep, splev
from ..contour import outliers


def apply_bspline(
    contour_points: np.ndarray, degree: int = 3, smoothing: float = 0.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Applies a B-spline to the contour points with improvements to avoid peaks.

    Args:
        contour_points: Contour points as pairs (y, x)
        degree: Degree of the B-spline
        smoothing: Smoothing parameter

    Returns:
        Tuple of (x_new, y_new) points of the B-spline, or (None, None) if fitting fails
    """
    if len(contour_points) < degree + 1:
        print("Insufficient points to create a B-spline")
        return None, None

    # Pre-process the contour to eliminate peaks
    smoothed_points = outliers.detect_and_correct_outliers(
        contour_points, is_spline=True
    )

    # Apply signal smoothing to reduce local variations
    smoothed_points = outliers.apply_contour_smoothing(smoothed_points, sigma=2)

    # Extract x and y coordinates
    x = smoothed_points[:, 1]  # Column coordinate
    y = smoothed_points[:, 0]  # Row coordinate

    # Ensure it's a closed contour by duplicating the first point
    if not np.array_equal(smoothed_points[0], smoothed_points[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Apply centrifugal chord length parameterization
    # This parameterization gives more weight to more pronounced curves
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    accumulated_distance = np.zeros(len(x))
    accumulated_distance[1:] = np.cumsum(distances)

    # Create the B-spline with greater curvature control
    try:
        # Increase the smoothing parameter for a more natural contour
        adjusted_smoothing = smoothing * 2.0

        # Use periodic=True to ensure a closed contour without discontinuities
        tck, u = splprep([x, y], u=None, s=adjusted_smoothing, k=degree, per=1)

        # Evaluate the B-spline with more points for greater detail
        u_new = np.linspace(0, 1, 400)
        x_new, y_new = splev(u_new, tck)

        # Check for peaks in the resulting spline
        xy_spline = np.column_stack([y_new, x_new])
        xy_spline_smoothed = outliers.detect_and_correct_outliers(
            xy_spline, is_spline=True
        )

        # Apply a second correction pass for difficult cases
        xy_spline_smoothed = outliers.detect_and_correct_outliers(
            xy_spline_smoothed, is_spline=True
        )

        # Apply additional smoothing to maintain continuity
        xy_spline_smoothed = outliers.apply_contour_smoothing(
            xy_spline_smoothed, sigma=1.5
        )

        # Extract the final coordinates
        x_new = xy_spline_smoothed[:, 1]
        y_new = xy_spline_smoothed[:, 0]

        return x_new, y_new
    except Exception as e:
        print(f"Error creating B-spline: {e}")
        return None, None


def apply_adaptive_bspline(
    contour_points: np.ndarray,
    min_degree: int = 2,
    max_degree: int = 4,
    base_smoothing: float = 10.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Adaptively applies a B-spline based on contour complexity.
    Uses higher degree and less smoothing for complex contours,
    and lower degree and more smoothing for simple contours.

    Args:
        contour_points: Contour points as pairs (y, x)
        min_degree: Minimum degree to use
        max_degree: Maximum degree to use
        base_smoothing: Base smoothing parameter

    Returns:
        Tuple of (x_new, y_new) points of the B-spline, or (None, None) if fitting fails
    """
    if len(contour_points) < max_degree + 1:
        # If not enough points for max_degree, use the highest possible
        degree = min(min_degree, len(contour_points) - 1)
        if degree < 1:
            print("Insufficient points to create a B-spline")
            return None, None
    else:
        # Determine complexity
        complexity = outliers.get_contour_complexity(contour_points)

        # Scale to 0-1 range approximately
        normalized_complexity = min(
            1.0, complexity / 0.5
        )  # 0.5 is an empirical threshold

        # Interpolate degree based on complexity
        degree_range = max_degree - min_degree
        degree = min_degree + int(round(normalized_complexity * degree_range))

        # Adjust smoothing inversely to complexity
        smoothing = base_smoothing * (1.0 - normalized_complexity * 0.5)

    return apply_bspline(contour_points, degree, smoothing)


def extract_contour_from_spline(
    x_spline: np.ndarray, y_spline: np.ndarray, num_points: int = 100
) -> np.ndarray:
    """
    Extracts a contour with a specific number of points from a spline.

    Args:
        x_spline: X-coordinates of the spline
        y_spline: Y-coordinates of the spline
        num_points: Number of points to extract

    Returns:
        Contour with the specified number of points
    """
    # Create indices to sample
    indices = np.linspace(0, len(x_spline) - 1, num_points, dtype=int)

    # Extract the points
    x = x_spline[indices]
    y = y_spline[indices]

    # Stack into contour points (y, x)
    contour = np.column_stack((y, x))

    return contour
