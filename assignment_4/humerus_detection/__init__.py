#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Humerus Detection and Modeling Package.
Provides a modular approach to detect, model and visualize the humerus from DICOM axial sections.
"""

import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

# Import submodules
from .io import dicom_utils
from .preprocessing import image_processing
from .contour import detection, outliers
from .spline import fitting
from .visualization import plotting

def run_pipeline(dicom_directory: str, output_directory: str, show_images: bool = False) -> None:
    """
    Run the complete humerus detection and modeling pipeline.
    
    Args:
        dicom_directory: Directory containing the DICOM files
        output_directory: Directory where to save the results
        show_images: Whether to display the images during processing
    """
    print(f"Running humerus detection pipeline:")
    print(f"  - DICOM directory: {dicom_directory}")
    print(f"  - Output directory: {output_directory}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process all DICOM files in the directory
    files = sorted([f for f in os.listdir(dicom_directory) if f.endswith('.dcm')])
    
    print(f"Found {len(files)} DICOM files to process")
    previous_contour = None
    
    for i, file in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {file}")
        full_path = os.path.join(dicom_directory, file)
        
        # Step 1: Read DICOM file
        dataset, pixel_array = dicom_utils.read_dicom(full_path)
        if dataset is None or pixel_array is None:
            print(f"  Error reading {file}")
            continue
        
        base_name = os.path.splitext(os.path.basename(full_path))[0]
            
        # Step 2: Detect the humerus contour
        contour = detection.detect_advanced_humerus_contour(
            pixel_array, 
            previous_contour=previous_contour, 
            file_name=base_name,
            slice_height=dicom_utils.get_slice_height(dataset)
        )
        
        # Step 3: If no contour (humerus disappeared), continue to next file
        if len(contour) == 0:
            print(f"  No humerus detected in {base_name}")
            previous_contour = None
            
            # Save an empty image for visualization
            fig = plt.figure(figsize=(10, 8))
            plt.imshow(pixel_array, cmap='gray')
            plt.title(f"Axial slice - No humerus detected")
            plt.axis('off')
            plt.tight_layout()
            plotting.save_results(fig, output_directory, base_name, "end_humerus")
            plt.close(fig)
            continue
            
        # Step 4: Apply B-spline
        smoothed_contour = outliers.detect_and_correct_outliers(contour, is_spline=False)
        
        # Use more smoothing for a more anatomical result
        smoothing = 10.0 * 2.5
        degree = 3
        x_spline, y_spline = fitting.apply_bspline(smoothed_contour, degree, smoothing)
        
        if x_spline is None or y_spline is None:
            print(f"  Error applying B-spline to {base_name}")
            continue
        
        # Step 5: Visualize and save results
        title = f"Axial slice - B-spline (degree {degree}, smoothing {smoothing})"
        fig = plotting.visualize_results(
            pixel_array, contour, x_spline, y_spline, title
        )
        
        plotting.save_results(fig, output_directory, base_name, "advanced")
        
        if not show_images:
            plt.close(fig)
        else:
            plt.show()
        
        # Update previous contour for next iteration
        previous_contour = contour
    
    print(f"Advanced processing completed. Results saved to {output_directory}")

def process_single_slice(dicom_path: str, output_directory: str, 
                        previous_contour: Optional[np.ndarray] = None, 
                        show_images: bool = False) -> Optional[np.ndarray]:
    """
    Process a single DICOM slice to detect the humerus.
    
    Args:
        dicom_path: Path to the DICOM file
        output_directory: Directory where to save the results
        previous_contour: Contour from the previous slice for continuity
        show_images: Whether to display the images during processing
        
    Returns:
        The detected contour or None if detection failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Step 1: Read DICOM file
    dataset, pixel_array = dicom_utils.read_dicom(dicom_path)
    if dataset is None or pixel_array is None:
        return None
    
    base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        
    # Step 2: Detect the humerus contour
    contour = detection.detect_advanced_humerus_contour(
        pixel_array, 
        previous_contour=previous_contour, 
        file_name=base_name,
        slice_height=dicom_utils.get_slice_height(dataset)
    )
    
    # If no contour (humerus disappeared), return None
    if len(contour) == 0:
        return None
        
    # Step 3: Apply B-spline
    smoothed_contour = outliers.detect_and_correct_outliers(contour, is_spline=False)
    
    # Use more smoothing for a more anatomical result
    smoothing = 10.0 * 2.5
    degree = 3
    x_spline, y_spline = fitting.apply_bspline(smoothed_contour, degree, smoothing)
    
    if x_spline is None or y_spline is None:
        return None
    
    # Step 4: Visualize and save results
    title = f"Axial slice - B-spline (degree {degree}, smoothing {smoothing})"
    fig = plotting.visualize_results(
        pixel_array, contour, x_spline, y_spline, title
    )
    
    plotting.save_results(fig, output_directory, base_name, "advanced")
    
    if not show_images:
        plt.close(fig)
    else:
        plt.show()
    
    return contour 