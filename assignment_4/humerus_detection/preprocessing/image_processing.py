#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image processing utilities for enhancing and segmenting DICOM images.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation, binary_erosion
from skimage import measure, morphology, filters, feature
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk, remove_small_objects, closing
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter, disk as draw_disk
from skimage.util import invert
from typing import Tuple, Optional, Union, List

def normalize_image(pixel_array: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values to a range of 0 to 1.
    
    Args:
        pixel_array: Pixel matrix of the DICOM image
        
    Returns:
        Normalized image
    """
    img_min = np.min(pixel_array)
    img_max = np.max(pixel_array)
    
    if img_max > img_min:
        return (pixel_array - img_min) / (img_max - img_min)
    return pixel_array

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhances image contrast to facilitate detection.
    
    Args:
        image: Normalized image
        
    Returns:
        Contrast-enhanced image
    """
    # Simple contrast stretching
    p_low, p_high = np.percentile(image, (2, 98))
    img_enhanced = np.clip(image, p_low, p_high)
    img_enhanced = (img_enhanced - p_low) / (p_high - p_low)
    return img_enhanced

def segment_humerus(pixel_array: np.ndarray) -> np.ndarray:
    """
    Segments the humerus in the DICOM image using advanced techniques.
    In MRI images, the humerus appears as a dark region.
    
    Args:
        pixel_array: Pixel matrix of the DICOM image
        
    Returns:
        Binary mask of the humerus
    """
    # Normalize the image
    norm_image = normalize_image(pixel_array)
    
    # Enhance contrast
    enhanced_image = enhance_contrast(norm_image)
    
    # Apply Gaussian filter to reduce noise
    smoothed_image = gaussian_filter(enhanced_image, sigma=1.0)
    
    # INVERSION: Invert the image so the humerus (dark) becomes bright
    inverted_image = 1 - smoothed_image
    
    # Apply adaptive threshold to improve segmentation
    local_threshold = threshold_local(inverted_image, block_size=31, method='gaussian')
    local_mask = inverted_image > local_threshold
    
    # Remove small objects and close small holes
    filtered_mask = remove_small_objects(local_mask, min_size=150)
    
    # Apply morphological operations for a smoother shape
    structuring_element = disk(3)
    closed_mask = closing(filtered_mask, structuring_element)
    
    # Fill holes in the mask
    final_mask = binary_fill_holes(closed_mask)
    
    return final_mask

def create_circular_mask(center_y: int, center_x: int, radius: int, 
                         shape: Tuple[int, int], fill: bool = True) -> np.ndarray:
    """
    Creates a circular mask centered at (center_y, center_x) with the given radius.
    
    Args:
        center_y: Y-coordinate of the center
        center_x: X-coordinate of the center
        radius: Radius of the circle
        shape: Shape of the output mask (height, width)
        fill: If True, fills the circle; if False, creates only the perimeter
        
    Returns:
        Binary mask with the circle
    """
    mask = np.zeros(shape, dtype=bool)
    
    if fill:
        # Create a filled disk
        rr, cc = draw_disk((center_y, center_x), radius, shape=shape)
    else:
        # Create only the perimeter
        rr, cc = circle_perimeter(center_y, center_x, radius, shape=shape)
    
    mask[rr, cc] = True
    return mask

def threshold_roi(image: np.ndarray, roi_mask: np.ndarray, 
                 block_size: int = 31, offset: float = -0.1) -> np.ndarray:
    """
    Applies local thresholding to a region of interest defined by a mask.
    
    Args:
        image: Input image (normalized, 0-1 range)
        roi_mask: Binary mask defining the region of interest
        block_size: Size of the blocks for local thresholding
        offset: Offset for the threshold
        
    Returns:
        Binary mask after thresholding
    """
    # Apply the ROI mask to the image
    roi_image = image * roi_mask
    
    # Apply local thresholding
    local_threshold = threshold_local(roi_image, block_size=block_size, method='gaussian', offset=offset)
    
    # Create the thresholded mask
    thresholded_mask = (roi_image < local_threshold) & roi_mask
    
    # Remove small objects
    filtered_mask = remove_small_objects(thresholded_mask, min_size=50)
    
    # Fill holes
    final_mask = binary_fill_holes(filtered_mask)
    
    return final_mask 