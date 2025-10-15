"""
Main script for Assignment 2: Parametric Spline Curves in 2D.

This script allows running examples of 2D parametric spline curves.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from assignment_2.curves.spline2d.curve import Spline2D, ellipse_example


def main():
    """
    Main function to run 2D spline curve examples.
    """
    parser = argparse.ArgumentParser(description='2D Parametric Spline Curves')
    parser.add_argument('--example', choices=['ellipse', 'custom'], 
                        default='ellipse', help='Example to run')
    parser.add_argument('--points', nargs='+', 
                        help='Custom points in format x1,y1 x2,y2 ...')
    parser.add_argument('--show_derivatives', action='store_true',
                        help='Show tangent vectors along the curve')
    parser.add_argument('--save_only', action='store_true',
                        help='Only save the plot as image without displaying it')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assignment_2", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.example == 'ellipse':
        print("Running ellipse example...")
        curve = ellipse_example(save_only=args.save_only, output_dir=output_dir, timestamp=timestamp)
    
    elif args.example == 'custom':
        if not args.points:
            print("Error: For custom example, you must provide points using --points")
            return
        
        try:
            # Parse points from command line
            points = []
            for point_str in args.points:
                x, y = map(float, point_str.split(','))
                points.append((x, y))
            
            points = np.array(points)
            
            # Create and plot the curve
            curve = Spline2D(points)
            fig, ax = plt.subplots(figsize=(10, 8))
            curve.plot(show_derivatives=args.show_derivatives, ax=ax)
            plt.tight_layout()
            
            # Save the plot
            filename = f"custom_spline_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Plot saved to: {filepath}")
            
            if not args.save_only:
                plt.show()
            
        except ValueError as e:
            print(f"Error parsing points: {e}")
            print("Points should be in format x1,y1 x2,y2 ...")
            return


def generate_custom_ellipse_points(a=4, b=5, num_points=5, random_seed=None):
    """
    Generate custom points on an ellipse with specified parameters.
    
    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        num_points (int): Number of points to generate.
        random_seed (int, optional): Seed for random number generator.
        
    Returns:
        tuple: (points, parameter_values)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random parameter values between 0 and 2π
    u = np.sort(np.random.uniform(0, 2*np.pi, num_points))
    
    # Generate points on the ellipse
    points = np.array([(a*np.cos(ui), b*np.sin(ui)) for ui in u])
    
    return points, u


if __name__ == "__main__":
    main() 