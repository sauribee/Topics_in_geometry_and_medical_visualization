#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometric operations for contours and shapes.
"""

import numpy as np
from typing import Tuple


def calculate_contour_area(contour: np.ndarray) -> float:
    """
    Calculates the area enclosed by a contour using the polygon area formula.

    Args:
        contour: Contour points as pairs (y, x)

    Returns:
        Area of the contour
    """
    # Shoelace formula (or Gauss's area formula)
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_circularity(contour: np.ndarray) -> float:
    """
    Calculates the circularity of a contour.
    Circularity = 4π * Area / Perimeter²
    A perfect circle has circularity = 1.

    Args:
        contour: Contour points as pairs (y, x)

    Returns:
        Circularity value (0 to 1)
    """
    area = calculate_contour_area(contour)
    perimeter = len(contour)  # Approximation of perimeter

    if perimeter > 0:
        return 4 * np.pi * area / (perimeter**2)
    return 0.0


def draw_polygon(
    r: np.ndarray, c: np.ndarray, shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a mask of a polygon.

    Args:
        r: Row coordinates of the vertices
        c: Column coordinates of the vertices
        shape: Shape of the image

    Returns:
        Tuple of (row indices, column indices) of the pixels inside the polygon
    """
    # Convert to integers for indexing
    r = r.astype(int)
    c = c.astype(int)

    # Ensure coordinates are within bounds
    r = np.clip(r, 0, shape[0] - 1)
    c = np.clip(c, 0, shape[1] - 1)

    # Create a closed contour
    if not np.array_equal(np.array([r[0], c[0]]), np.array([r[-1], c[-1]])):
        r = np.append(r, r[0])
        c = np.append(c, c[0])

    # Draw the polygon (using skimage.draw.polygon)
    try:
        from skimage.draw import polygon

        rr, cc = polygon(r, c, shape)
        return rr, cc
    except Exception as e:
        print(f"Error drawing polygon: {e}")
        return np.array([]), np.array([])


def contour_centroid(contour: np.ndarray) -> np.ndarray:
    """
    Calculates the centroid (center of mass) of a contour.

    Args:
        contour: Contour points as pairs (y, x)

    Returns:
        Centroid coordinates as [y, x]
    """
    return np.mean(contour, axis=0)


def contour_bounding_box(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calculates the bounding box of a contour.

    Args:
        contour: Contour points as pairs (y, x)

    Returns:
        Tuple of (min_y, min_x, max_y, max_x)
    """
    min_y = np.min(contour[:, 0])
    min_x = np.min(contour[:, 1])
    max_y = np.max(contour[:, 0])
    max_x = np.max(contour[:, 1])

    return int(min_y), int(min_x), int(max_y), int(max_x)


def resample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resamples a contour to have a specified number of points.
    Uses linear interpolation along the contour.

    Args:
        contour: Contour points as pairs (y, x)
        num_points: Number of points in the resampled contour

    Returns:
        Resampled contour
    """
    # Calculate the accumulated length along the contour
    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    accumulated_length = np.zeros(len(contour))
    accumulated_length[1:] = np.cumsum(segment_lengths)

    # Normalize to [0, 1]
    if accumulated_length[-1] > 0:
        accumulated_length = accumulated_length / accumulated_length[-1]

    # Create new parameter values for interpolation
    new_t = np.linspace(0, 1, num_points)

    # Create resampled contour
    resampled_contour = np.zeros((num_points, 2))

    # Interpolate y and x separately
    from scipy.interpolate import interp1d

    for dim in range(2):
        interpolator = interp1d(accumulated_length, contour[:, dim])
        resampled_contour[:, dim] = interpolator(new_t)

    return resampled_contour
