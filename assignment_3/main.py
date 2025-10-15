"""
Main script for Assignment 3: B-Spline Curve for Rotator Cuff MRI Contours.

This script loads points from a MATLAB file and creates a quadratic B-spline
that interpolates the rotator cuff contour.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from datetime import datetime

from assignment_3.curves.spline2d.bspline import QuadraticBSpline


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
        
        # Convert to float64 to ensure compatibility with splprep
        points = points.astype(np.float64)
        
        print(f"Loaded {len(points)} points from {file_path}")
        print(f"Points shape: {points.shape}, dtype: {points.dtype}")
        
        return points
    except Exception as e:
        print(f"Error loading points from {file_path}: {e}")
        raise


def rotator_cuff_example(points_file='ptos.mat', show_derivatives=False, 
                          save_only=False, output_dir=None, timestamp=None,
                          smoothing=0.0, auto_correct=True, sort_angular=False):
    """
    Create and visualize a B-spline curve for the rotator cuff contour.
    
    Args:
        points_file (str): Path to the .mat file with the contour points.
        show_derivatives (bool): Whether to show derivative vectors along the curve.
        save_only (bool): Whether to only save the plot without displaying it.
        output_dir (str): Directory to save the output images.
        timestamp (str): Timestamp to use for file naming.
        smoothing (float): Smoothing parameter for the spline (0.0 for exact interpolation).
        auto_correct (bool): Whether to automatically correct points to avoid self-intersections.
        sort_angular (bool): Whether to sort points angularly around the center of mass.
        
    Returns:
        QuadraticBSpline: The created B-spline curve.
    """
    # Load the points
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(points_file):
        points_file = os.path.join(script_dir, points_file)
    
    points = load_points_from_mat(points_file)
    
    # Sort points by angle around center if requested (ensures no self-intersections)
    if sort_angular:
        points = sort_points_by_angle(points)
    
    # Create the B-spline with improved settings
    print(f"Creating B-spline with smoothing={smoothing}, auto_correct={auto_correct}")
    spline = QuadraticBSpline(
        points, 
        closed=True, 
        smoothing=smoothing, 
        auto_correct=auto_correct,
        monotone_parameterization=(not sort_angular)  # If points are already sorted, skip this
    )
    
    # Generate a grid of parameters for experiments
    if smoothing == 0.0 and output_dir:
        # Experiment with different smoothing parameters
        smoothing_values = [0, 5, 10, 20, 50]
        for s in smoothing_values:
            try:
                # Create spline with current smoothing
                test_spline = QuadraticBSpline(
                    points, 
                    closed=True, 
                    smoothing=s, 
                    auto_correct=auto_correct,
                    monotone_parameterization=True
                )
                
                # Create figure for this test
                test_fig, test_ax = plt.subplots(figsize=(10, 8))
                test_ax.set_facecolor('white')
                test_ax.set_title(f'B-spline con suavizado {s}')
                
                # Set axis limits
                test_ax.set_xlim(100, 300)
                test_ax.set_ylim(250, 100)
                
                # Plot the test spline
                test_spline.plot(
                    ax=test_ax, 
                    show_points=True, 
                    show_derivatives=show_derivatives,
                    curve_color='#0080FF', 
                    point_color='#FF4500', 
                    der_color='#00CC00',
                    n_curve_points=200, 
                    n_der_points=15, 
                    der_scale=10,
                    highlight_self_intersections=True
                )
                
                # Save this test figure
                test_filename = f"rotator_cuff_bspline_smooth{s}_{timestamp}.png"
                test_filepath = os.path.join(output_dir, test_filename)
                test_fig.savefig(test_filepath, dpi=300, bbox_inches='tight')
                plt.close(test_fig)
                print(f"Test plot with smoothing={s} saved to: {test_filepath}")
            except Exception as e:
                print(f"Error generating test plot with smoothing={s}: {e}")
    
    # Create a figure for main visualization
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Make background white
    ax.set_facecolor('white')
    
    # Set axis limits for MRI image
    ax.set_xlim(100, 300)
    ax.set_ylim(250, 100)  # Inverted y-axis for image coordinates
    
    # Plot the B-spline with enhanced options
    spline.plot(
        ax=ax, 
        show_points=True, 
        show_derivatives=show_derivatives,
        show_curvature=True,  # Show curvature indicators
        curve_color='#0080FF', 
        point_color='#FF4500', 
        der_color='#00CC00',
        curvature_color='#8A2BE2',  # Purple for curvature
        n_curve_points=200, 
        n_der_points=20, 
        der_scale=10,
        curvature_scale=200,  # Scale for curvature visualization
        highlight_self_intersections=True
    )
    
    # Set axis labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Add detailed title
    title_text = 'Interpolación de contorno de manguito rotador con B-spline cúbico mejorado'
    if sort_angular:
        title_text += ' (ordenación angular)'
    ax.set_title(title_text, fontsize=14, pad=20)
    
    # Add information about parameterization
    textstr = '\n'.join((
        'Características de la interpolación:',
        '- B-spline cúbico (grado 3)',
        '- Parameterización por longitud de cuerda',
        '- Corrección para evitar auto-intersecciones',
        f'- Suavizado: {smoothing}',
        f'- {len(points)} puntos de control',
        '- Curva cerrada'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the filename
        method_tag = "angular" if sort_angular else "mejorado"
        smooth_tag = f"_smooth{int(smoothing)}" if smoothing > 0 else ""
        filename = f"rotator_cuff_bspline_{method_tag}{smooth_tag}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")
    
    # Show the figure if not save_only
    if not save_only:
        plt.show()
    
    return spline


def main():
    """
    Main function to run the B-spline interpolation example.
    """
    parser = argparse.ArgumentParser(description='B-Spline Curve for Medical Image Contours')
    parser.add_argument('--points_file', default='ptos.mat',
                       help='Path to the .mat file with the contour points')
    parser.add_argument('--show_derivatives', action='store_true',
                       help='Show tangent vectors along the curve')
    parser.add_argument('--save_only', action='store_true',
                       help='Only save the plot as image without displaying it')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save the output images')
    parser.add_argument('--smoothing', type=float, default=10.0,
                       help='Smoothing parameter for the spline (0.0 for exact interpolation)')
    parser.add_argument('--no_auto_correct', action='store_true',
                       help='Disable automatic correction of points')
    parser.add_argument('--sort_angular', action='store_true',
                       help='Sort points angularly around center of mass to prevent self-intersections')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the example with improved parameters
    rotator_cuff_example(
        points_file=args.points_file,
        show_derivatives=args.show_derivatives,
        save_only=args.save_only,
        output_dir=args.output_dir,
        timestamp=timestamp,
        smoothing=args.smoothing,
        auto_correct=not args.no_auto_correct,
        sort_angular=args.sort_angular
    )


if __name__ == "__main__":
    main() 