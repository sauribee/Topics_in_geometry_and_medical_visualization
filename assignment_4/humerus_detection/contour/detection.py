#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for detecting the humerus contour in DICOM images.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation
from skimage import measure, feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.segmentation import active_contour
from skimage.draw import circle_perimeter
from skimage.morphology import disk as morphology_disk

from ..preprocessing import image_processing
from . import outliers
from . import geometry

def detect_humerus_circle(pixel_array: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Detects the humerus using the Hough circular transform.
    The humerus typically appears as an approximately circular structure in axial sections.
    
    Args:
        pixel_array: Pixel matrix of the DICOM image
    
    Returns:
        Tuple containing (center_x, center_y, radius) of the detected circle for the humerus,
        or (None, None, None) if detection fails
    """
    # Normalize the image
    norm_image = image_processing.normalize_image(pixel_array)
    
    # Invert so the humerus (dark) becomes bright
    inverted_image = 1 - norm_image
    
    # Apply Gaussian filter to reduce noise
    smoothed_image = gaussian_filter(inverted_image, sigma=2.0)
    
    # Detect edges for the Hough transform
    edges = feature.canny(
        smoothed_image,
        sigma=2.0,
        low_threshold=0.1,
        high_threshold=0.2
    )
    
    # Estimate radius range based on image size
    # The humerus generally occupies between 1/15 and 1/6 of image width
    min_radius = min(pixel_array.shape) // 20
    max_radius = min(pixel_array.shape) // 6
    
    # Apply Hough transform to detect circles
    hough_radii = np.arange(min_radius, max_radius, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    # Extract the best circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, 
        total_num_peaks=5,  # Look for the 5 most prominent circles
        threshold=0.5 * np.max(hough_res)
    )
    
    # If no circles were detected, return None
    if len(accums) == 0:
        return None, None, None
    
    # Evaluate each circle based on its position and accumulator
    best_score = -float('inf')
    best_cx, best_cy, best_radius = None, None, None
    
    image_center = np.array([pixel_array.shape[1] // 2, pixel_array.shape[0] // 2])
    
    for i, (cx_i, cy_i, radius_i, acc_i) in enumerate(zip(cx, cy, radii, accums)):
        # Calculate distance to image center
        distance = np.sqrt((cx_i - image_center[0])**2 + (cy_i - image_center[1])**2)
        
        # Score based on accumulator and proximity to center
        # Strongly penalize circles far from center
        score = acc_i - (distance / 10.0)**2
        
        # Check if this circle contains dark regions (likely humerus)
        # Create a circular mask using morphology.disk
        circle_mask = np.zeros_like(pixel_array, dtype=bool)
        rr, cc = circle_perimeter(int(cy_i), int(cx_i), int(radius_i), shape=pixel_array.shape)
        
        if len(rr) > 0 and len(cc) > 0:
            circle_mask[rr, cc] = True
            # Dilate to fill the circle
            circle_mask = binary_dilation(circle_mask, structure=morphology_disk(radius_i//2))
            
            # Calculate mean intensity inside the circle (lower for humerus)
            mean_intensity = np.mean(norm_image[circle_mask])
            # Penalize circles with high intensity
            score -= mean_intensity * 10
        
        if score > best_score:
            best_score = score
            best_cx, best_cy, best_radius = cx_i, cy_i, radius_i
    
    return best_cx, best_cy, best_radius

def generate_circular_contour(center_x: float, center_y: float, radius: float, num_points: int = 100) -> np.ndarray:
    """
    Generates a perfect circular contour.
    
    Args:
        center_x: X-coordinate of center
        center_y: Y-coordinate of center
        radius: Radius of the circle
        num_points: Number of points in the contour
        
    Returns:
        Circular contour as a numpy array of shape (num_points, 2)
    """
    # Generate uniform angles
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Generate coordinates
    y = center_y + radius * np.sin(angles)
    x = center_x + radius * np.cos(angles)
    
    # Create contour (in [y, x] format)
    contour = np.column_stack((y, x))
    
    return contour

def generate_terminal_contour(previous_contour: np.ndarray, reduction_factor: float) -> np.ndarray:
    """
    Generates a smaller terminal contour based on the previous contour,
    simulating how the humerus ends in a spherical shape.
    
    Args:
        previous_contour: Contour from the previous slice
        reduction_factor: Reduction factor (0-1) where 0 means disappear
        
    Returns:
        Reduced contour
    """
    if previous_contour is None or len(previous_contour) < 3:
        return np.array([])
        
    # Calculate the center of the previous contour
    center = np.mean(previous_contour, axis=0)
    
    # For very small reduction factors, make a more spherical shape
    if reduction_factor < 0.3:
        # Create a perfect small circle
        radius = np.max(np.sqrt(np.sum((previous_contour - center)**2, axis=1))) * reduction_factor
        
        # Generate points in a circle
        num_points = len(previous_contour)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        
        # Circle coordinates
        y = center[0] + radius * np.sin(angles)
        x = center[1] + radius * np.cos(angles)
        
        # Create circular contour (in [y, x] format)
        reduced_contour = np.column_stack((y, x))
    else:
        # Reduce the contour towards its center maintaining the general shape
        reduced_contour = center + (previous_contour - center) * reduction_factor
    
    # Smooth the resulting contour for a more rounded shape
    from scipy.ndimage import gaussian_filter1d
    reduced_contour = gaussian_filter1d(reduced_contour, sigma=2, axis=0)
    
    return reduced_contour

def detect_advanced_humerus_contour(
    pixel_array: np.ndarray, 
    previous_contour: Optional[np.ndarray] = None, 
    file_name: Optional[str] = None,
    slice_height: Optional[float] = None
) -> np.ndarray:
    """
    Detects the humerus contour (dark region) in a DICOM image.
    1. Detects the circular zone (humerus) with Hough.
    2. Searches for the real contour only within that zone.
    3. Adjusts the contour with snake.
    4. If it fails, uses the circle as a fallback.
    
    Includes improvements to maintain continuity between images.
    
    Args:
        pixel_array: Pixel matrix of the DICOM image
        previous_contour: Contour from the previous slice for continuity
        file_name: Name of the file being processed (used for special handling of terminal slices)
        slice_height: Height of the slice in mm
        
    Returns:
        Detected contour as a numpy array of shape (n_points, 2)
    """
    # Check if we're in the final slices where the humerus disappears
    final_slices = ["I18", "I19", "I20"]
    
    # If we're in a final slice, force the termination of the humerus
    if file_name in final_slices:
        # Force the end of the humerus in I19 and I20 directly
        if file_name in ["I19", "I20"]:
            print(f"Forcing the end of the humerus in {file_name}")
            return np.array([])  # Return an empty contour
            
        # For I18, analyze intensity and contrast in the central region
        norm_image = image_processing.normalize_image(pixel_array)
        center_y, center_x = pixel_array.shape[0] // 2, pixel_array.shape[1] // 2
        analysis_radius = min(pixel_array.shape) // 6
        
        # Create a circular mask in the center
        central_mask = image_processing.create_circular_mask(center_y, center_x, analysis_radius, pixel_array.shape)
        
        # Calculate statistics in the central region
        central_intensity = np.mean(norm_image[central_mask])
        central_deviation = np.std(norm_image[central_mask])
        central_contrast = central_deviation / max(central_intensity, 0.01)
        
        # Calculate a "humerus presence" score
        presence_score = (1 - central_intensity) * 10 + central_contrast * 5
        
        # Stricter threshold for I18
        if presence_score < 2.0:
            print(f"Humerus not detected in {file_name} (score={presence_score:.2f})")
            return np.array([])  # Return an empty contour
    
    # Detect if it's a problematic image
    is_problematic_image = file_name in ["I06", "I07", "I12", "I13", "I18", "I19", "I20"]
    
    # 1. Detect circle (humerus area)
    center_x, center_y, radius = detect_humerus_circle(pixel_array)
    if center_x is None:
        if previous_contour is not None and len(previous_contour) > 10:
            print(f"Using previous contour for {file_name} because Hough failed")
            return previous_contour
        # Use old method as fallback
        humerus_mask = image_processing.segment_humerus(pixel_array)
        contours = measure.find_contours(humerus_mask, 0.5)
        return select_humerus_contour(contours, pixel_array)
    
    # 2. Create circular mask
    circular_mask = image_processing.create_circular_mask(
        int(center_y), int(center_x), int(0.95*radius), pixel_array.shape
    )
    
    # 3. Search for the real contour only within the circular zone
    norm_image = image_processing.normalize_image(pixel_array)
    
    # Parameter adjustment specific for problematic images
    if is_problematic_image:
        block_size = 21  # Reduced for greater local sensitivity
        offset = -0.08   # Adjusted to improve detection
    else:
        block_size = 31
        offset = -0.1
    
    # Apply thresholding within the ROI
    humerus_mask = image_processing.threshold_roi(
        norm_image, circular_mask, block_size, offset
    )
    
    # Extract contours
    contours = measure.find_contours(humerus_mask, 0.5)
    
    # Debug: Print contour information
    print(f"Number of detected contours: {len(contours)}")
    for idx, cont in enumerate(contours):
        print(f"Contour {idx}: Points={len(cont)}, Area={geometry.calculate_contour_area(cont):.2f}")
    
    # If no contours detected and we're in final slices, the humerus might have already disappeared
    if len(contours) == 0 and file_name in final_slices:
        print(f"No contours detected in {file_name}, the humerus may have already disappeared")
        return np.array([])
    
    # If there's a previous contour and we're in a problematic image, use it as reference for selection
    if previous_contour is not None and is_problematic_image:
        previous_center = np.mean(previous_contour, axis=0)
        # Select the contour closest to the previous one
        best_contour = None
        min_distance = float('inf')
        
        for contour in contours:
            if len(contour) < 10:
                continue
            current_center = np.mean(contour, axis=0)
            distance = np.linalg.norm(current_center - previous_center)
            if distance < min_distance:
                min_distance = distance
                best_contour = contour
                
        # If we found a close contour and it has a reasonable area, use it
        if best_contour is not None and geometry.calculate_contour_area(best_contour) > 100:
            print(f"Using contour close to previous for {file_name}")
        else:
            # If we didn't find a good contour, modify the previous one
            print(f"Using adjusted previous contour for {file_name}")
            if previous_contour is not None and len(previous_contour) > 10:
                # Adapt the previous contour to this slice
                current_center = np.array([center_y, center_x])
                prev_cont_center = np.mean(previous_contour, axis=0)
                # Translation to the new estimated center
                best_contour = previous_contour - prev_cont_center + current_center
                # Verify it's within the image
                best_contour = np.clip(best_contour, 0, np.array(pixel_array.shape) - 1)
    else:
        # Normal selection of the largest and most circular contour
        best_contour = None
        best_circularity = 0
        for contour in contours:
            area = geometry.calculate_contour_area(contour)
            perimeter = len(contour)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if area > 100 and circularity > best_circularity:
                center = np.mean(contour, axis=0)
                if np.linalg.norm([center[1] - center_x, center[0] - center_y]) < 0.7*radius:
                    best_circularity = circularity
                    best_contour = contour
    
    # In the final slices, check if the contour is likely noise
    if file_name in final_slices and best_contour is not None:
        contour_area = geometry.calculate_contour_area(best_contour)
        if contour_area < 200:  # Higher minimum area threshold for final slices
            print(f"Contour too small in {file_name} (area={contour_area:.2f}), possible noise")
            return np.array([])
    
    # 4. Adjust with snake if there's a contour
    if best_contour is not None and len(best_contour) > 10:
        try:
            # First smooth the detected contour to avoid initial peaks
            best_contour = outliers.detect_and_correct_outliers(best_contour)
            
            # Apply an additional Gaussian filter to the contour
            from scipy.ndimage import gaussian_filter1d
            best_contour = gaussian_filter1d(best_contour, sigma=2, axis=0)
            
            # Custom parameters for problematic images
            if is_problematic_image:
                smoothed_image = gaussian_filter(norm_image, sigma=6.0)  # More smoothing
                snake = active_contour(
                    smoothed_image,
                    best_contour,
                    alpha=0.08,  # Higher tension to avoid collapse
                    beta=0.5,    # Higher rigidity for smoother shapes
                    gamma=0.0003, # Lower convergence rate
                    w_line=0,
                    w_edge=1,
                    max_num_iter=200
                )
            else:
                smoothed_image = gaussian_filter(norm_image, sigma=3.0)
                snake = active_contour(
                    smoothed_image,
                    best_contour,
                    alpha=0.03,
                    beta=0.2,
                    gamma=0.0008,
                    w_line=0,
                    w_edge=1,
                    max_num_iter=200
                )
            
            # Verify that the snake didn't collapse
            snake_area = geometry.calculate_contour_area(snake)
            if snake_area < 50 and previous_contour is not None:
                print(f"Snake collapsed for {file_name} (area={snake_area:.2f}). Using previous contour.")
                return previous_contour
            
            # Smooth the snake contour to remove possible peaks
            snake = outliers.detect_and_correct_outliers(snake)
            snake = gaussian_filter1d(snake, sigma=2, axis=0)
                
            # Force the contour position
            if previous_contour is not None:
                previous_center = np.mean(previous_contour, axis=0)
                snake_center = np.mean(snake, axis=0)
                displacement = previous_center - snake_center
                # Limit maximum displacement to 1/4 of the radius
                max_disp = radius/4
                if np.linalg.norm(displacement) > max_disp:
                    displacement = displacement * (max_disp / np.linalg.norm(displacement))
                snake += displacement * 0.6  # Adjust towards the previous center
            
            # Check the circularity of the final contour
            area = geometry.calculate_contour_area(snake)
            perimeter = len(snake)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # If the contour is very irregular (low circularity), apply more smoothing
            if circularity < 0.7:
                print(f"Irregular contour detected in {file_name} (circularity={circularity:.2f}). Applying additional smoothing.")
                snake = gaussian_filter1d(snake, sigma=3, axis=0)
            
            return snake
        except Exception as e:
            print(f"Snake failed: {e}")
            if previous_contour is not None:
                return previous_contour
            return best_contour
    
    # If there's no valid contour or it's too small
    if best_contour is None or geometry.calculate_contour_area(best_contour) < 50:
        # In final slices, it's more likely that the humerus has really disappeared
        if file_name in final_slices:
            print(f"No valid contour detected in {file_name}, the humerus may have already disappeared")
            return np.array([])
        else:
            print(f"Using previous contour or circle for {file_name} due to insufficient detection.")
            return previous_contour if previous_contour is not None else generate_circular_contour(center_x, center_y, radius)
    
    return best_contour

def select_humerus_contour(contours: List[np.ndarray], pixel_array: np.ndarray) -> np.ndarray:
    """
    Selects the contour most likely to correspond to the humerus (central dark and round region).
    Only accepts contours with circularity > 0.8. If none, forces circle with Hough or central circle.
    
    Args:
        contours: List of contours extracted from the binary mask
        pixel_array: Original DICOM pixel array
        
    Returns:
        The contour most likely to correspond to the humerus
    """
    if not contours:
        return np.array([])
    
    image_center = np.array([pixel_array.shape[1] // 2, pixel_array.shape[0] // 2])
    best_contour = None
    best_score = -float('inf')
    norm_image = image_processing.normalize_image(pixel_array)
    
    for contour in contours:
        area = geometry.calculate_contour_area(contour)
        if area < 150 or area > (pixel_array.shape[0] * pixel_array.shape[1] // 4):
            continue
        contour_center = np.mean(contour, axis=0)
        contour_center = np.array([contour_center[1], contour_center[0]])
        distance_to_center = np.linalg.norm(contour_center - image_center)
        perimeter = len(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Only consider very circular contours
        if circularity < 0.8:
            continue
        
        # Create a mask for this contour
        mask = np.zeros_like(pixel_array, dtype=bool)
        rr, cc = geometry.draw_polygon(contour[:, 0], contour[:, 1], pixel_array.shape)
        if len(rr) > 0 and len(cc) > 0:
            mask[rr, cc] = True
            mean_intensity = np.mean(norm_image[mask])
            intensity_factor = 1.0 - mean_intensity
            penalized_distance = (distance_to_center / 10.0)**2
            score = (circularity * 15.0 - penalized_distance + intensity_factor * 8.0)
            if score > best_score:
                best_score = score
                best_contour = contour
    
    # If no sufficiently circular contours, force circle with Hough
    if best_contour is None:
        center_x, center_y, radius = detect_humerus_circle(pixel_array)
        if center_x is not None:
            best_contour = generate_circular_contour(center_x, center_y, radius)
    
    # If still no contour, generate circle in the center
    if best_contour is None or len(best_contour) == 0:
        estimated_radius = min(pixel_array.shape) // 10
        best_contour = generate_circular_contour(image_center[0], image_center[1], estimated_radius)
    
    return best_contour 