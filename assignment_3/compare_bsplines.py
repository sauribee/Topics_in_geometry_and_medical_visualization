"""
Script to compare different B-spline parameters for medical contours.

This script generates multiple visualizations of the same set of points
using different B-spline degrees and smoothing parameters.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import splprep, splev
from datetime import datetime
import argparse


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


def create_bspline(points, degree, smoothing, per=1):
    """
    Create a B-spline from points.
    
    Args:
        points (ndarray): Control points.
        degree (int): B-spline degree.
        smoothing (float): Smoothing parameter.
        per (int): Whether the curve is periodic (1) or not (0).
        
    Returns:
        tuple: (tck, u, x_new, y_new) where tck are the spline coefficients,
        u are the parameters, and x_new, y_new are the evaluated spline points.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Close the curve by duplicating the first point at the end if needed
    if per and (x[0] != x[-1] or y[0] != y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    
    # Create the B-spline using splprep
    tck, u = splprep([x, y], s=smoothing, k=degree, per=per)
    
    # Evaluate the B-spline at evenly spaced points
    u_new = np.linspace(0, 1, 200)
    x_new, y_new = splev(u_new, tck)
    
    return tck, u, x_new, y_new


def generate_comparison_grid(points_file='ptos.mat', output_dir=None, sort_angular=True):
    """
    Generate a comparison grid of different B-splines.
    
    Args:
        points_file (str): Path to the .mat file with contour points.
        output_dir (str): Directory to save output images.
        sort_angular (bool): Whether to sort points angularly.
    """
    # Load the points
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(points_file):
        points_file = os.path.join(script_dir, points_file)
    
    # Configure default output directory inside assignment_3
    if output_dir is None:
        output_dir = os.path.join(script_dir, "output")
    
    # Create subfolder for comparisons
    output_subdir = os.path.join(output_dir, "comparison_grid")
    os.makedirs(output_subdir, exist_ok=True)
    
    points = load_points_from_mat(points_file)
    
    # Sort points by angle if requested
    if sort_angular:
        points = sort_points_by_angle(points)
    
    # Define degrees and smoothing parameters to compare
    degrees = [1, 2, 3, 5]
    smoothing_values = [0, 10, 50, 200]
    
    # Create a large figure for the grid
    fig, axes = plt.subplots(len(degrees), len(smoothing_values), figsize=(16, 12))
    fig.suptitle('Comparison of B-splines by Degree and Smoothing', fontsize=16)
    
    # Iterate over all combinations
    for i, degree in enumerate(degrees):
        for j, smoothing in enumerate(smoothing_values):
            ax = axes[i, j]
            
            # Calculate the B-spline for this combination
            _, _, x_new, y_new = create_bspline(points, degree, smoothing)
            
            # Configure this subplot
            ax.set_facecolor('white')
            ax.set_xlim(100, 300)
            ax.set_ylim(250, 100)
            
            # Plot the original points
            ax.plot(points[:, 0], points[:, 1], 'o', color='#FF4500', markersize=3)
            
            # Plot the B-spline curve
            ax.plot(x_new, y_new, '-', color='#0080FF', linewidth=1.5)
            
            # Add title to subplot
            ax.set_title(f'Degree {degree}, Smoothing {smoothing}', fontsize=10)
            
            # Hide axis labels on internal subplots
            if i < len(degrees) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    
    # Adjust spacing
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bspline_comparison_grid_{timestamp}.png"
    filepath = os.path.join(output_subdir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to: {filepath}")
    
    # Show the figure
    plt.show()


def generate_animation_frames(points_file='ptos.mat', output_dir=None, 
                             degree=3, smoothing=0.0, sort_angular=True):
    """
    Generate a series of images showing the curve from different angles.
    
    Args:
        points_file (str): Path to the .mat file with contour points.
        output_dir (str): Directory to save output images.
        degree (int): B-spline degree.
        smoothing (float): Smoothing parameter.
        sort_angular (bool): Whether to sort points angularly.
    """
    # Load the points
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(points_file):
        points_file = os.path.join(script_dir, points_file)
    
    # Configure default output directory inside assignment_3
    if output_dir is None:
        output_dir = os.path.join(script_dir, "output")
    
    # Create subfolder for animations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    animation_dir = os.path.join(output_dir, f"animation_{timestamp}")
    os.makedirs(animation_dir, exist_ok=True)
    
    points = load_points_from_mat(points_file)
    
    # Sort points by angle if requested
    if sort_angular:
        points = sort_points_by_angle(points)
    
    # Calculate the B-spline
    _, _, x_new, y_new = create_bspline(points, degree, smoothing)
    
    # Generate images from different angles
    n_frames = 36
    for i in range(n_frames):
        # Calculate rotation angle
        angle = i * (360 / n_frames)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = plt.subplot(111)
        
        # Configure the plot
        ax.set_facecolor('white')
        ax.set_xlim(100, 300)
        ax.set_ylim(250, 100)
        
        # Plot the original points
        ax.plot(points[:, 0], points[:, 1], 'o', color='#FF4500', markersize=6, label='Original Points')
        
        # Plot the B-spline curve
        ax.plot(x_new, y_new, '-', color='#0080FF', linewidth=2, label='B-spline')
        
        # Add title with angle information
        ax.set_title(f'Contour with Degree {degree} B-spline (View {i+1}/{n_frames})', fontsize=14)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Save the image
        frame_filename = f"frame_{i:03d}.png"
        frame_path = os.path.join(animation_dir, frame_filename)
        plt.savefig(frame_path, bbox_inches='tight')
        plt.close(fig)
        
        # Show progress
        if (i + 1) % 5 == 0 or i == 0 or i == n_frames - 1:
            print(f"Generated frame {i+1}/{n_frames}")
    
    print(f"Animation images saved to: {animation_dir}")
    print("To create a GIF animation, you can use:")
    print(f"convert -delay 10 -loop 0 {animation_dir}/frame_*.png {output_dir}/animation_degree{degree}_smooth{int(smoothing)}_{timestamp}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='B-spline Comparison for Medical Contours')
    parser.add_argument('--points_file', default='ptos.mat',
                       help='Path to the .mat file with contour points')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save output images')
    parser.add_argument('--no_sort_angular', action='store_true',
                       help='Do not sort points angularly')
    parser.add_argument('--mode', choices=['grid', 'animation'], default='grid',
                       help='Operation mode: grid (comparison grid) or animation (frames for animation)')
    parser.add_argument('--degree', type=int, default=3,
                       help='B-spline degree for animation mode')
    parser.add_argument('--smoothing', type=float, default=0.0,
                       help='Smoothing parameter for animation mode')
    
    args = parser.parse_args()
    
    # Execute the selected mode
    if args.mode == 'grid':
        generate_comparison_grid(
            points_file=args.points_file,
            output_dir=args.output_dir,
            sort_angular=not args.no_sort_angular
        )
    else:  # mode == 'animation'
        generate_animation_frames(
            points_file=args.points_file,
            output_dir=args.output_dir,
            degree=args.degree,
            smoothing=args.smoothing,
            sort_angular=not args.no_sort_angular
        ) 