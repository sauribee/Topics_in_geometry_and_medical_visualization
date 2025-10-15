"""
Simplified script to generate clean contour curves.

This script loads points from a MATLAB file and generates a clean visualization
using scipy.interpolate directly for B-splines.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import splprep, splev
from datetime import datetime


def sort_points_by_angle(points):
    """
    Sort points in counterclockwise order around their center of mass.
    
    Args:
        points (ndarray): Array of 2D points.
        
    Returns:
        ndarray: Sorted array of 2D points.
    """
    # Calculate center of mass
    center = np.mean(points, axis=0)
    
    # Calculate angles
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    print(f"Points sorted by angle around center {center}")
    return sorted_points


def load_points_from_mat(file_path):
    """
    Load points from a MATLAB .mat file.
    
    Args:
        file_path (str): Path to the .mat file.
        
    Returns:
        ndarray: Array of 2D points as float64.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        # Extract the points array (assuming it's stored with key 'ptos')
        points = mat_data.get('ptos', None)
        
        if points is None:
            raise ValueError(f"No 'ptos' key found in {file_path}")
        
        # Convert to float64
        points = points.astype(np.float64)
        
        print(f"Loaded {len(points)} points from {file_path}")
        print(f"Points shape: {points.shape}, dtype: {points.dtype}")
        
        return points
    except Exception as e:
        print(f"Error loading points from {file_path}: {e}")
        raise


def generate_clean_bspline(points_file='ptos.mat', degree=3, smoothing=0.0, 
                          save_only=False, output_dir=None, sort_angular=True):
    """
    Create and visualize a clean B-spline curve for the rotator cuff contour.
    
    Args:
        points_file (str): Path to the .mat file with contour points.
        degree (int): B-spline degree (3=cubic, 2=quadratic).
        smoothing (float): Smoothing parameter (0.0 for exact interpolation).
        save_only (bool): Whether to only save the plot without displaying.
        output_dir (str): Directory to save output images.
        sort_angular (bool): Whether to sort points angularly around center of mass.
    """
    # Load the points
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(points_file):
        points_file = os.path.join(script_dir, points_file)
    
    # Configure default output directory inside assignment_3
    if output_dir is None:
        output_dir = os.path.join(script_dir, "output")
    
    # Create subfolder for simple visualizations
    output_subdir = os.path.join(output_dir, "simple")
    os.makedirs(output_subdir, exist_ok=True)
    
    points = load_points_from_mat(points_file)
    
    # Sort points by angle if requested
    if sort_angular:
        points = sort_points_by_angle(points)
    
    # Apply splprep/splev directly (scipy approach)
    x = points[:, 0]
    y = points[:, 1]
    
    # Close the curve by duplicating the first point at the end
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    
    # Create the B-spline using splprep
    tck, u = splprep([x, y], s=smoothing, k=degree, per=1)
    
    # Evaluate the B-spline at evenly spaced points
    u_new = np.linspace(0, 1, 200)
    x_new, y_new = splev(u_new, tck)
    
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # White background
    ax.set_facecolor('white')
    
    # Configure axis limits for MRI image
    ax.set_xlim(100, 300)
    ax.set_ylim(250, 100)  # Inverted y-axis for image coordinates
    
    # Plot the original points
    ax.plot(x, y, 'o', color='#FF4500', markersize=6, label='Original Points')
    
    # Plot the B-spline curve
    ax.plot(x_new, y_new, '-', color='#0080FF', linewidth=2, label='B-spline')
    
    # Configure axis labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Add detailed title
    title_text = f'Rotator Cuff Contour with Degree {degree} B-spline'
    if sort_angular:
        title_text += ' (Angular Sorting)'
    ax.set_title(title_text, fontsize=14)
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate the filename
    method_tag = "angular" if sort_angular else "simple"
    smooth_tag = f"_smooth{int(smoothing)}" if smoothing > 0 else ""
    filename = f"rotator_cuff_simple_degree{degree}_{method_tag}{smooth_tag}_{timestamp}.png"
    filepath = os.path.join(output_subdir, filename)
    
    # Save the figure
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    # Show the figure if not save_only
    if not save_only:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple B-Spline for Medical Image Contours')
    parser.add_argument('--points_file', default='ptos.mat',
                       help='Path to the .mat file with contour points')
    parser.add_argument('--degree', type=int, default=3,
                       help='B-spline degree (3=cubic, 2=quadratic)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                       help='Smoothing parameter (0.0 for exact interpolation)')
    parser.add_argument('--save_only', action='store_true',
                       help='Only save the plot as image without displaying it')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save output images')
    parser.add_argument('--no_sort_angular', action='store_true',
                       help='Do not sort points angularly around center of mass')
    
    args = parser.parse_args()
    
    # Execute with simplified parameters
    generate_clean_bspline(
        points_file=args.points_file,
        degree=args.degree,
        smoothing=args.smoothing,
        save_only=args.save_only,
        output_dir=args.output_dir,
        sort_angular=not args.no_sort_angular
    ) 