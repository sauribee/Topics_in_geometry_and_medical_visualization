#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for detecting and correcting outliers in contours.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def detect_and_correct_outliers(
    points: np.ndarray, is_spline: bool = False
) -> np.ndarray:
    """
    Detects and corrects outlier points in the contour that could cause peaks.

    Args:
        points: Contour points as pairs (y, x)
        is_spline: Indicates if we're processing a spline (more strict)

    Returns:
        Contour with corrected outliers
    """
    if len(points) < 5:
        return points

    # Calculate distance between consecutive points (local smoothness)
    corrected_points = points.copy()
    n_points = len(points)

    # Use a sliding window to detect abrupt changes in direction
    window = 5
    # Stricter thresholds for splines
    angle_threshold = 30 if is_spline else 45  # degrees - lower = stricter

    # Calculate the center of the contour (useful for peaks that deviate)
    center = np.mean(points, axis=0)

    # Calculate distances to center
    center_distances = np.sqrt(np.sum((points - center) ** 2, axis=1))

    # Calculate statistics of distances
    mean_distance = np.mean(center_distances)
    distance_std = np.std(center_distances)

    # Threshold to consider a point as a potential peak by distance
    distance_threshold = mean_distance + 1.5 * distance_std

    # First pass: detect points based on angles
    detected_peaks = []
    for i in range(n_points):
        is_peak = False

        # Indices of previous and next points (with circular handling)
        idx_prev = (i - window // 2) % n_points
        idx_next = (i + window // 2) % n_points

        # Vectors from current point to previous and next points
        v_prev = points[idx_prev] - points[i]
        v_next = points[idx_next] - points[i]

        # Normalize vectors
        norm_prev = np.linalg.norm(v_prev)
        norm_next = np.linalg.norm(v_next)

        if norm_prev > 0 and norm_next > 0:
            v_prev = v_prev / norm_prev
            v_next = v_next / norm_next

            # Calculate angle between vectors (in degrees)
            dot_product = np.clip(np.dot(v_prev, v_next), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))

            # If angle is very small, there's a peak
            if angle < angle_threshold:
                is_peak = True

        # Also detect peaks based on distance to center
        # (abnormally distant points)
        if center_distances[i] > distance_threshold:
            # Check if it's an isolated peak by comparing with neighbors
            idx_neighbors = [
                (i - 2) % n_points,
                (i - 1) % n_points,
                (i + 1) % n_points,
                (i + 2) % n_points,
            ]
            neighbor_distances = [center_distances[idx] for idx in idx_neighbors]

            # If most neighbors are closer to center, it's an isolated peak
            if np.mean(neighbor_distances) < 0.9 * center_distances[i]:
                is_peak = True

        if is_peak:
            detected_peaks.append(i)

    # Second pass: correct points identified as peaks
    for i in detected_peaks:
        # For peaks in splines, use wider window
        if is_spline:
            # Use more neighbors for more aggressive smoothing
            idx_neighbors = [
                (i - 3) % n_points,
                (i - 2) % n_points,
                (i - 1) % n_points,
                (i + 1) % n_points,
                (i + 2) % n_points,
                (i + 3) % n_points,
            ]
        else:
            idx_neighbors = [
                (i - 2) % n_points,
                (i - 1) % n_points,
                (i + 1) % n_points,
                (i + 2) % n_points,
            ]

        neighbors = np.array([points[idx] for idx in idx_neighbors])
        corrected_points[i] = np.mean(neighbors, axis=0)

    return corrected_points


def apply_contour_smoothing(contour: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Applies Gaussian smoothing to a contour.

    Args:
        contour: Contour points as pairs (y, x)
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Smoothed contour
    """
    return gaussian_filter1d(contour, sigma=sigma, axis=0)


def correct_contour_self_intersections(contour: np.ndarray) -> np.ndarray:
    """
    Detects and corrects self-intersections in a contour.

    Args:
        contour: Contour points as pairs (y, x)

    Returns:
        Contour with corrected self-intersections
    """
    # This is a simple implementation - for complex cases, more sophisticated algorithms are needed
    if len(contour) < 5:
        return contour

    _corrected_contour = contour.copy()
    _n_points = len(contour)

    # Calculate the center of the contour
    center = np.mean(contour, axis=0)

    # Convert to polar coordinates relative to the center
    y_rel = contour[:, 0] - center[0]
    x_rel = contour[:, 1] - center[1]
    angles = np.arctan2(y_rel, x_rel)

    # Sort by angle to eliminate self-intersections
    sort_idx = np.argsort(angles)

    # Create a new contour with points sorted by angle
    sorted_contour = contour[sort_idx]

    # Apply smoothing to the sorted contour
    smoothed_contour = apply_contour_smoothing(sorted_contour, sigma=1.5)

    return smoothed_contour


def get_contour_complexity(contour: np.ndarray) -> float:
    """
    Calculates a measure of the contour's complexity based on its curvature.

    Args:
        contour: Contour points as pairs (y, x)

    Returns:
        Complexity score (higher values indicate more complex contours)
    """
    if len(contour) < 5:
        return 0.0

    # Calculate differences between consecutive points
    diffs = np.diff(contour, axis=0, append=contour[0:1])

    # Calculate angles between consecutive segments
    angles = []
    for i in range(len(contour)):
        prev_idx = (i - 1) % len(contour)
        curr_idx = i

        # Get vectors
        v1 = diffs[prev_idx]
        v2 = diffs[curr_idx]

        # Calculate angle
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 0 and norm2 > 0:
            v1_norm = v1 / norm1
            v2_norm = v2 / norm2
            dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot)
            angles.append(angle)

    # Complexity is related to the variance of angles
    if len(angles) > 0:
        return np.std(angles)

    return 0.0
