"""
Script to apply the optimized points to the ellipse example.

This script creates a spline image using the optimized points
found during the search process.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import argparse
from assignment_2.curves.spline2d.curve import Spline2D

def create_optimized_curve(a=4, b=5):
    """
    Creates a spline curve with optimized points.
    
    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        
    Returns:
        Spline2D: Optimized spline curve.
    """
    # These are the best angles found with seed=100 and iterations=1000
    best_angles = np.array([0.19091285, 0.95369017, 2.07253858, 3.61990273, 5.98016705])
    
    # Generate points on the ellipse using these angles
    points = np.array([(a*np.cos(angle), b*np.sin(angle)) for angle in best_angles])
    
    # Create and return the spline
    return Spline2D(points, t=best_angles)

def create_original_curve(a=4, b=5):
    """
    Creates the original spline curve (for comparison).
    
    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        
    Returns:
        Spline2D: Original spline curve.
    """
    # These are the angles from the original example
    np.random.seed(42)
    original_angles = np.sort(np.random.uniform(0, 2*np.pi, 5))
    
    # Generate points on the ellipse
    points = np.array([(a*np.cos(angle), b*np.sin(angle)) for angle in original_angles])
    
    # Create and return the spline
    return Spline2D(points, t=original_angles)

def calculate_error(curve, a, b, num_samples=100):
    """
    Calculates the error between a spline and an ellipse.
    
    Args:
        curve: Spline2D object.
        a, b: Semi-axes of the ellipse.
        num_samples: Number of points to calculate the error.
        
    Returns:
        error: Average error between the spline and the ellipse.
    """
    # Sample the ellipse
    theta = np.linspace(0, 2*np.pi, num_samples)
    ellipse_points = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
    
    # Map the theta values to the range of t parameters of the spline
    t_min, t_max = curve.t[0], curve.t[-1]
    t_safe = np.linspace(t_min, t_max - 1e-10, num_samples)
    
    # Evaluate the spline at the safe parameters
    spline_points = curve(t_safe)
    
    # Initialize the error
    squared_distances = np.zeros(num_samples)
    
    # For each point on the ellipse, find the closest point on the spline
    for i, ellipse_point in enumerate(ellipse_points):
        # Calculate distances to all points on the spline
        distances = np.sum((spline_points - ellipse_point)**2, axis=1)
        # Take the minimum distance
        squared_distances[i] = np.min(distances)
    
    # Calculate the square root of the mean of squared distances
    error = np.sqrt(np.mean(squared_distances))
    return error

def save_plot(curve, a, b, output_dir, filename="optimized_ellipse_final.png"):
    """
    Saves a comparison between the optimized spline and the real ellipse.
    
    Args:
        curve: Optimized spline.
        a, b: Semi-axes of the ellipse.
        output_dir: Directory where to save the image.
        filename: Image filename.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the spline
    t_best = np.linspace(curve.t[0], curve.t[-1] - 1e-10, 100)
    best_points = curve(t_best)
    ax.plot(best_points[:, 0], best_points[:, 1], 'g-', linewidth=2, label='Optimized Spline')
    
    # Show control points
    ax.plot(curve.points[:, 0], curve.points[:, 1], 'mo', markersize=8, label='Control Points')
    # Number the points
    for i, (x, y) in enumerate(curve.points):
        ax.text(x, y, f' {i+1}', fontsize=12)
    
    # Draw the ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)
    ax.plot(ellipse_x, ellipse_y, 'r--', linewidth=1.5, label='Real Ellipse')
    
    # Calculate and show the error
    error = calculate_error(curve, a, b)
    ax.set_title(f'Optimized Spline for Ellipse (a={a}, b={b}, Error: {error:.6f})')
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    # Add point information
    info_text = "Optimized points:\n"
    for i, (x, y) in enumerate(curve.points):
        info_text += f"Point {i+1}: ({x:.4f}, {y:.4f})\n"
    
    # Place informative text
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Save the image
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Image saved to: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Apply optimized points to ellipse spline')
    parser.add_argument('--a', type=float, default=4,
                        help='Semi-major axis of the ellipse')
    parser.add_argument('--b', type=float, default=5,
                        help='Semi-minor axis of the ellipse')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')
    
    args = parser.parse_args()
    
    # Configure output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    # Generate filename with timestamp if not specified
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_ellipse_final_{timestamp}.png"
    else:
        filename = args.output
    
    # Create optimized curve
    optimized_curve = create_optimized_curve(args.a, args.b)
    
    # Create original curve (for reference)
    original_curve = create_original_curve(args.a, args.b)
    
    # Calculate errors
    opt_error = calculate_error(optimized_curve, args.a, args.b)
    orig_error = calculate_error(original_curve, args.a, args.b)
    
    # Print comparative information
    print(f"Original spline vs. Optimized spline:")
    print(f"Original error: {orig_error:.6f}")
    print(f"Optimized error: {opt_error:.6f}")
    print(f"Improvement: {(orig_error - opt_error) / orig_error * 100:.2f}%")
    
    # Save image
    filepath = save_plot(optimized_curve, args.a, args.b, output_dir, filename)
    
    print(f"\nOptimized spline image created!")
    print(f"You can use this image for your applied geometry assignment.")

if __name__ == "__main__":
    main() 