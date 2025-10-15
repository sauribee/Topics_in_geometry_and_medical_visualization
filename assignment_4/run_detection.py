#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the humerus detection pipeline.
"""

import os
import argparse
import time
from humerus_detection import run_pipeline

def main():
    """
    Main function to parse arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Detect and model the humerus from DICOM axial sections.")
    
    parser.add_argument(
        "--dicom_dir", 
        type=str, 
        default="assignment_4/axial_sections",
        help="Directory containing the DICOM files"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="assignment_4/advanced_results",
        help="Directory where to save the results"
    )
    
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show images during processing"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("Humerus Detection Pipeline")
    print("--------------------------")
    print(f"DICOM directory: {args.dicom_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Show images: {args.show}")
    print("--------------------------")
    
    # Record start time
    start_time = time.time()
    
    # Run the pipeline
    run_pipeline(args.dicom_dir, args.output_dir, args.show)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 