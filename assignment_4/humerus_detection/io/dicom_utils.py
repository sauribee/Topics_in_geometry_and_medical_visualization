#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for reading and processing DICOM files.
"""

import os
import pydicom
import numpy as np
from typing import Tuple, Optional

def read_dicom(file_path: str) -> Tuple[Optional[pydicom.dataset.FileDataset], Optional[np.ndarray]]:
    """
    Reads a DICOM file and returns its information and pixel data.
    
    Args:
        file_path: Path to the DICOM file
        
    Returns:
        Tuple containing the dataset with metadata and the pixel array with image data,
        or (None, None) if reading fails
    """
    try:
        dataset = pydicom.dcmread(file_path)
        return dataset, dataset.pixel_array
    except Exception as e:
        print(f"Error reading DICOM file {file_path}: {e}")
        return None, None

def get_slice_height(dataset: pydicom.dataset.FileDataset) -> Optional[float]:
    """
    Gets the height of the axial slice from DICOM metadata.
    
    Args:
        dataset: DICOM dataset containing metadata
        
    Returns:
        Height of the slice in mm, or None if not available
    """
    if hasattr(dataset, 'SliceLocation'):
        return float(dataset.SliceLocation)
    elif hasattr(dataset, 'ImagePositionPatient'):
        return float(dataset.ImagePositionPatient[2])
    return None

def get_pixel_spacing(dataset: pydicom.dataset.FileDataset) -> Tuple[float, float]:
    """
    Gets the pixel spacing from DICOM metadata.
    
    Args:
        dataset: DICOM dataset containing metadata
        
    Returns:
        Tuple with pixel spacing in mm (row_spacing, col_spacing)
    """
    if hasattr(dataset, 'PixelSpacing'):
        return float(dataset.PixelSpacing[0]), float(dataset.PixelSpacing[1])
    return 1.0, 1.0  # Default values if not available

def get_slice_thickness(dataset: pydicom.dataset.FileDataset) -> float:
    """
    Gets the slice thickness from DICOM metadata.
    
    Args:
        dataset: DICOM dataset containing metadata
        
    Returns:
        Slice thickness in mm
    """
    if hasattr(dataset, 'SliceThickness'):
        return float(dataset.SliceThickness)
    return 1.0  # Default value if not available

def load_slice_spacing(dicom_directory: str) -> Tuple[float, Tuple[float, float]]:
    """
    Loads spacing information from DICOM files in a directory.
    
    Args:
        dicom_directory: Directory containing DICOM files
        
    Returns:
        Tuple containing (z_spacing, (pixel_spacing_x, pixel_spacing_y))
    """
    import glob
    
    files = sorted(glob.glob(os.path.join(dicom_directory, "*.dcm")))
    
    if not files:
        return 1.0, (1.0, 1.0)  # Default values if no files
    
    # Load first file to get pixel resolution
    try:
        ds = pydicom.dcmread(files[0])
        pixel_spacing = get_pixel_spacing(ds)
    except:
        pixel_spacing = (1.0, 1.0)  # Default value
    
    # If there's more than one file, calculate spacing between slices
    if len(files) > 1:
        try:
            ds1 = pydicom.dcmread(files[0])
            ds2 = pydicom.dcmread(files[1])
            
            if hasattr(ds1, 'SliceLocation') and hasattr(ds2, 'SliceLocation'):
                z_spacing = abs(ds2.SliceLocation - ds1.SliceLocation)
            elif hasattr(ds1, 'ImagePositionPatient') and hasattr(ds2, 'ImagePositionPatient'):
                z_spacing = abs(ds2.ImagePositionPatient[2] - ds1.ImagePositionPatient[2])
            else:
                z_spacing = 1.0  # Default value
        except:
            z_spacing = 1.0  # Default value
    else:
        z_spacing = 1.0  # Default value for a single file
    
    return z_spacing, pixel_spacing 